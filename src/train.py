import os
from statistics import mean

import numpy as np
import tensorflow as tf
import imageio
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from skimage import transform

import config
import init

import dataset
import stardist_blocks as sd
import tiled_copy
import misc



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
    unet_block_out, skip_layers, concat_layers = sd.unet_block(3, n_filter_base=64)(unet_input)
    fluo_channels = Conv2D(3, (1, 1), name='fluo_channels', activation='relu')(unet_block_out)

    # We can add another branch to the output of the middle upsample layer and supervise the 
    # CP nuclei segmentation... The loss should be modified to use bce for the 4th channel.
    if config.include_nuclei_channel:

        concat_m2_f = 256
        concat_m2 = concat_layers[-2]
        skip_0 = skip_layers[0]
        conv_kernel = (3,3)
        pool = (2,2)
        padding = "same"

        l = Conv2D(concat_m2_f//(2**1), conv_kernel, padding=padding)(concat_m2)
        l = Conv2D(concat_m2_f//(2**2), conv_kernel, padding=padding)(l)
        l = UpSampling2D(pool)(l)
        l = Concatenate()([l, skip_0])
        l = Conv2D(concat_m2_f//(2**2), conv_kernel, padding=padding)(l)
        l = Conv2D(concat_m2_f//(2**2), conv_kernel, padding=padding)(l)
        nuclei_seg = Conv2D(1, (1,1), name='nuclei_seg', activation='sigmoid')(l)
        unet_output = Concatenate(-1, name='result')([fluo_channels, nuclei_seg])
    else:
        unet_output = fluo_channels

    model = Model(unet_input, unet_output)
    model.summary(line_length=130)

    '''

    Weighed loss:

    Network input: (b, h, w, 3)
    b images are in the batch, each of them has hxw pixels and 3 channels

    The default MeanSquaredError will reduce the mean over the whole batch.

    '''

    def channelwise_loss(y_true, y_pred):
        weights = [1., .0, .0]

        total_loss = 0.
        for ch_id in [0, 1, 2]:
            ch_mse = MeanSquaredError()(y_true[..., ch_id], y_pred[..., ch_id])
            #ch_bce = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true[..., ch], y_pred[..., ch])
            total_loss += weights[ch_id] * ch_mse
        return total_loss

    model.compile(optimizer='adam', loss='mean_squared_error')
    #model.compile(optimizer='adam', loss=channelwise_loss)
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

    sequence.return_meta = True
    mse_per_image = {}
    n_fluo_channels = 3

    mse_all = {mag: {i: [] for i in range(n_fluo_channels)} for mag in config.magnifications}

    for x, y, meta in sequence:
        print()

        batch_element = 0
        plot_layout = 140

        x_sample, y_sample = x[batch_element], y[batch_element]
        z_pos = np.shape(x_sample)[-1]//2
        x_im, y_im = x_sample[..., z_pos], y_sample

        magnification = misc.magnification_level(meta[batch_element][0])
        stat = dataset.load_stats()

        if model is not None:
            plot_layout = 240

            if tile_sizes is not None:
                print('Tile sizes: ', tile_sizes)
                y_pred = predict_tiled(x, n_fluo_channels, tile_sizes)
            else:
                y_pred = model.predict(x)
                for ch in range(n_fluo_channels):
                    plt.imshow(y_pred[batch_element, ..., ch])
            
            y_pred_sample = y_pred[batch_element]

            if config.visualize:
                plt.subplot(plot_layout + 6, title='Predicted fluorescent (red)')
                plt.imshow(y_pred_sample[..., 0])
                
                plt.subplot(plot_layout + 7, title='Predicted Fluorescent (green)')
                plt.imshow(y_pred_sample[..., 1])
                
                plt.subplot(plot_layout + 8, title='Predicted Fluorescent (blue)')
                plt.imshow(y_pred_sample[..., 2])

            if config.save and not config.readonly:
                filename = os.path.basename(meta[batch_element][0])
                print('Predicted filename: %s, mag: %s' % (filename, magnification))
                # AssayPlate_Greiner_#655090_D04_T0001F012L01A04Z07C04.tif
                # AssayPlate_Greiner_#655090_D04_T0001F012L01   [len_stem]
                # {ACTION}
                # Z07                                           [len_stem+3:len_stem+6]
                # {CHANNEL}
                # .tif                                          [len_stem+9:]
                im_id = filename[:len('AssayPlate_Greiner_#655090_D04_T0001F012')]
                len_stem = len('AssayPlate_Greiner_#655090_D04_T0001F012L01')
                filename = list(filename)
                out_filename_pattern = \
                    filename[:len_stem] + ['A%.2d'] + filename[len_stem+3:len_stem+6] + ['C%.2d'] + filename[len_stem+9:]
                out_filename_pattern = ''.join(out_filename_pattern)
                print(out_filename_pattern)
                
                # Save visualization results
                vis_subdir = os.path.join(config.output_dir, config.experiment_id, 'visual', magnification)

                os.makedirs(os.path.join(vis_subdir), exist_ok=True)

                imageio.imwrite(
                    os.path.join(vis_subdir, im_id + '_bright.tif'), x_im)

                imageio.imwrite(
                    os.path.join(vis_subdir, im_id + '_true.tif'), y_sample)
                
                imageio.imwrite(
                    os.path.join(vis_subdir, im_id + '_out.tif'), y_pred_sample)
                
                imageio.imwrite(
                    os.path.join(vis_subdir, im_id + '_out.tif'), y_pred_sample)

                diff = (y_sample-y_pred_sample)**2
                imageio.imwrite(
                    os.path.join(vis_subdir, im_id + '_diff.tif'), diff)

                mse = np.sum(diff)/np.size(diff)
                stat_key = '%s/%s/%s'
                mse_per_image[stat_key % (magnification, im_id, 'all')] = mse

                for ch_id in range(n_fluo_channels):
                    diff_ch = (y_sample[..., ch_id]-y_pred_sample[..., ch_id])**2
                    mse_ch = np.sum(diff_ch)/np.size(diff_ch)
                    mse_per_image[stat_key % (magnification, im_id, str(ch_id))] = mse_ch
                    mse_all[magnification][ch_id].append(mse_ch)

                # Save raw results
                result_subdir = os.path.join(config.experiment_id, 'results', magnification)
                os.makedirs(os.path.join(config.output_dir, result_subdir), exist_ok=True)

                for channel_id in range(3):
                    y_result =  y_pred_sample[..., channel_id]*65535.0 # TODO: call unnormalize here!
                    if config.upscale_result is not None:
                        y_result = transform.resize(y_result, config.upscale_result, anti_aliasing=False, order=1) #order=1 -> bilinear
                    
                    y_result = y_result.astype(np.uint16)
                    imageio.imwrite(os.path.join(config.output_dir, result_subdir, out_filename_pattern % (channel_id+1, channel_id+1)), y_result)
        # End of prediction code

        if config.visualize:
            plt.subplot(plot_layout + 1, title='Input Brightfield@Z=%d' % z_pos)
            plt.imshow(x_im)

            plt.subplot(plot_layout + 2, title='GT Fluorescent (red)')
            plt.imshow(y_im[..., 0])
            
            plt.subplot(plot_layout + 3, title='GT Fluorescent (green)')
            plt.imshow(y_im[..., 1])

            plt.subplot(plot_layout + 4, title='GT Fluorescent (blue)')
            plt.imshow(y_im[..., 2])

            plt.show()
            #plt.savefig('%d.png' % idx)

            plt.subplot(141, title='Inp, unnorm')
            bright_stat = stat[magnification]['mean'][-1], stat[magnification]['std'][-1]
            fluo_stat = stat[magnification]['mean'][:3], stat[magnification]['std'][:3]
            
            n_x = dataset.standardize_bright(x_sample, *bright_stat, inverse=True)
            plt.imshow(n_x[..., z_pos])

            plt.subplot(142, title='Inp as it got')
            plt.imshow(x_sample[..., z_pos])

            plt.subplot(143, title='Fluo unnorm')
            n_y = dataset.standardize_fluo(y_sample, *fluo_stat, inverse=True)
            plt.imshow(n_y[..., 0])

            plt.subplot(144, title='Fluo as it got')
            plt.imshow(y_sample[..., 0])

            #imageio.volwrite('y.tif', y_sample)
            #imageio.volwrite('y_unnorm.tif', n_y)

            plt.show()

        # End of loop

    if model is not None and config.save and not config.readonly:
        final_result = {mag: {ch: mean(mse_all[mag][ch]) for ch in range(n_fluo_channels)} for mag in config.magnifications}

        experiment_dir = os.path.join(config.output_dir, config.experiment_id)

        misc.put_json(os.path.join(experiment_dir, 'mse_per_image.json'), mse_per_image)
        misc.put_json(os.path.join(experiment_dir, 'mse_final.json'), final_result)
        misc.put_json(os.path.join(experiment_dir, 'mse_all.json'), mse_all)

    sequence.return_meta = False

    # End of function

if __name__ == '__main__':
    # Leave out wells for validation
    lo_ws = ['D04']

    train_sequence = dataset.get_dataset(
            config.data_dir,                                        # Data directory, it has 3 subdirs: 20x, 40x, 60. 
        train_=True,                                                # If it will be used for training. (No augmentation for val)
        sample_per_image=config.train_samples_per_image,            # How many random crops should be extracted per FOV
        random_subsample_input=config.train_subsample,              # Do we want to extract crops / or want to train w/ the full images?
        seed=config.seed,                                           # ...
        filter_fun=lambda im: dataset.info(im)[1] not in lo_ws)     # Filter the data: the info() returns (experiment, well, field)
    
    val_sequence = dataset.get_dataset(
        config.data_dir, 
        train_=False, 
        sample_per_image=config.val_samples_per_image, 
        random_subsample_input=config.val_subsample, 
        seed=config.seed, 
        resetseed=True,                                             # The seed is reset after every epoch so we work with the same validation set.
        filter_fun=lambda im: dataset.info(im)[1] in lo_ws)

    print('Length of the train sequence: %d' % len(train_sequence))
    print('Length of the val sequence: %d' % len(val_sequence))

    model = get_network()

    if config.init_weights is not None:
        print('Loading weights:', config.init_weights)
        model.load_weights(config.init_weights)

    #test(val_sequence)                                             # If noting is passed but a sequence, then it shows each image (if the visualization turned on in the config.)

    if config.train == True:
        model = train((train_sequence, val_sequence), model)
    
    test(val_sequence, model, save=True, tile_sizes=config.predict_tile_size)
    #test(train_sequence)
