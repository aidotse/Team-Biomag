import os

import numpy as np
import tensorflow as tf
import imageio
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy

import config
import init

import dataset
import stardist_blocks as sd
import tiled_copy

def visualize(original, augmented):
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1,2,2)
    plt.title('Augmented image')
    plt.imshow(augmented)
    plt.show()


def get_network():
    unet_input = Input(shape=config.net_input_shape)
    unet_out = sd.unet_block(n_filter_base=64)(unet_input)
    fluo_channels = Conv2D(3, (1, 1), name='fluo_channels', activation='sigmoid')(unet_out)
    
    model = Model(unet_input, fluo_channels)
    model.summary(line_length=130)

    '''

    Weighed loss:

    Network input: (b, h, w, 3)
    b images are in the batch, each of them has hxw pixels and 3 channels

    The default MeanSquaredError will reduce the mean over the whole batch.

    '''
    def channelwise_loss(y_true, y_pred):
        
        total_loss = 0.
        
        weights = [.5, .2, .3]
        #weights = [1., 1., 1.]

        for ch in [0, 1, 2]:
            total_loss += weights[ch] * MeanSquaredError()(y_true[..., ch], y_pred[..., ch])
            '''
            total_loss += weights[ch] * BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(
                        y_true[..., ch], 
                        y_pred[..., ch]
                    )
            '''


        # The first channel is the nuclei
        # Most of the pixels below the intensity 600 are the part of the background and correlates with the cyto.
        # Therefore we concentrate on the >600 ground truth pixels. (600 ~ .1 after normalization)
        #nuclei_weight = .8
        #nuclei_thresh = .1
        
        #nuclei_weight_tensor = nuclei_weight*tf.cast(y_true[..., 0] > nuclei_thresh, tf.float32) + (1.-tf.cast(y_true[..., 0] <= nuclei_thresh, tf.float32))
        
        '''
        nuclei_loss = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(
            y_true[..., 0], 
            y_pred[..., 0]
        )
        '''
        
        #total_loss += tf.math.reduce_mean(nuclei_loss * nuclei_weight_tensor)
        #total_loss += tf.math.reduce_mean(nuclei_loss)

        return total_loss

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train(sequences, model):
    train, val = sequences
    mcp_save = tf.keras.callbacks.ModelCheckpoint(
        '%s/model-{epoch:04d}-{val_loss:.8f}.h5' % config.output_dir, 
        save_best_only=True, 
        monitor='val_loss', 
        mode='min')
    callbacks = []
    
    if not config.readonly:
        callbacks += [mcp_save]

    model.fit(
        train, 
        validation_data=val, 
        epochs=config.epochs,
        callbacks=callbacks, 
        initial_epoch=config.initial_epoch, 
        steps_per_epoch=len(train))

    return model

def predict_tiled(x, y_channels, tile_sizes):
    ys, xs = np.shape(x)[1], np.shape(x)[2]
    (y_src, y_src_crop, y_target), (x_src, x_src_crop, x_target) = tiled_copy.get_tiles(ys, tile_sizes[0]), tiled_copy.get_tiles(xs, tile_sizes[1])
    y_shape = np.shape(x)[:3] + (y_channels,)
    stitched_y = np.zeros(y_shape, x.dtype)

    for y_idx in range(len(y_src)):
        for x_idx in range(len(x_src)):
            print('Predicting tile: y=%d, x=%d' % (y_idx, x_idx))
            src_crop = (slice(None), slice(*y_src[y_idx]), slice(*x_src[x_idx]), slice(None)) 
            src_tile_crop = (slice(None), slice(*y_src_crop[y_idx]), slice(*x_src_crop[x_idx]), slice(None))
            target_crop = (slice(None), slice(*y_target[y_idx]), slice(*x_target[x_idx]), slice(None))
            x_tile_predict = model.predict(x[src_crop])

            stitched_y[target_crop] = x_tile_predict[src_tile_crop]
    
    return stitched_y

def test(sequence, model=None, save=False, tile_sizes=None):
    """
    If the model is set, it predicts the image using the model passed and shows the result.
    """
    for idx, (x, y) in enumerate(sequence):
        batch_element = 0
        plot_layout = 140

        x_sample, y_sample = x[batch_element], y[batch_element]
        z_pos = np.shape(x_sample)[-1]//2
        x_im, y_im = x_sample[..., z_pos], y_sample

        if model is not None:
            plot_layout = 240

            if tile_sizes is not None:
                y_pred = predict_tiled(x, 3, tile_sizes)
            else:
                y_pred = model.predict(x)
                plt.imshow(x[0, ..., 3])
                plt.show()
                for ch in range(3):
                    plt.imshow(y_pred[0, ..., ch])
                    plt.show()
            
            y_pred_sample = y_pred[batch_element]

            plt.subplot(plot_layout + 6, title='Predicted fluorescent (red)')
            plt.imshow(y_pred_sample[..., 0])

            plt.subplot(plot_layout + 7, title='Predicted Fluorescent (green)')
            plt.imshow(y_pred_sample[..., 1])

            plt.subplot(plot_layout + 8, title='Predicted Fluorescent (blue)')
            plt.imshow(y_pred_sample[..., 2])

            if save and not config.readonly:
                imageio.imwrite(os.path.join(config.output_dir, '%d_bright.tif' % idx), x_im)
                imageio.imwrite(os.path.join(config.output_dir, '%d_true.tif' % idx), y_sample)
                imageio.imwrite(os.path.join(config.output_dir, '%d_pred.tif' % idx), y_pred_sample)

        plt.subplot(plot_layout + 1, title='Input Brightfield@Z=%d' % z_pos)
        plt.imshow(x_im)

        plt.subplot(plot_layout + 2, title='GT Fluorescent (red)')
        plt.imshow(y_im[..., 0])

        plt.subplot(plot_layout + 3, title='GT Fluorescent (green)')
        plt.imshow(y_im[..., 1])
        
        plt.subplot(plot_layout + 4, title='GT Fluorescent (blue)')
        plt.imshow(y_im[..., 2])

        #plt.show()
        #plt.savefig('%d.png' % idx)

        if save and False:
            bright = (x_sample[..., 1]*255).astype(np.uint8)
            fluo = (y_sample*255).astype(np.uint8)
            
            imageio.imwrite(os.path.join(config.output_dir, '%d_bright.tif' % idx, bright))
            imageio.imwrite(os.path.join(config.output_dir, '%d_fluo.tif' % idx, fluo))


            imageio.volwrite('%s/pred-%d.tif' % (config.output_dir, idx), y_pred_sample)
            imageio.volwrite('%s/true-%d.tif' % (config.output_dir, idx), y_im)
            imageio.volwrite('%s/input-%d.tif' % (config.output_dir, idx), np.transpose(x_sample, (2, 0, 1)))


if __name__ == '__main__':
    # Leave out wells for validation
    lo_ws = ['D04']

    train_sequence = dataset.get_dataset(
        config.data_dir, 
        train_=True, 
        sample_per_image=config.train_samples_per_image, 
        random_subsample_input=True, 
        seed=config.seed, 
        filter_fun=lambda im: dataset.info(im)[1] not in lo_ws)
    
    val_sequence = dataset.get_dataset(
        config.data_dir, 
        train_=False, 
        sample_per_image=config.val_samples_per_image, 
        random_subsample_input=True, 
        seed=config.seed, 
        resetseed=True,
         filter_fun=lambda im: dataset.info(im)[1] in lo_ws)

    print('Length of the train sequence: %d' % len(train_sequence))
    print('Length of the val sequence: %d' % len(val_sequence))

    model = get_network()

    if config.init_weights is not None:
        print('Loading weights:', config.init_weights)
        model.load_weights(config.init_weights)

    if config.train == True:
        model = train((train_sequence, val_sequence), model)
    
    test(val_sequence, model)
    #test(train_sequence)
