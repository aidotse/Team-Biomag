from glob import glob
from collections import defaultdict
import os
import math
from random import Random

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
import imageio
from skimage import transform
import matplotlib.pyplot as plt

import config
import init
import stardist_blocks as sd
import tiled_copy

norm = {
    '20x': {'low': {0: 37, 1: 7, 2: 258, 3: 828}, 'high': {0: 2652, 1: 2096, 2: 1345, 3: 5953}},
    '40x': {'low': {0: 25, 1: 23, 2: 6, 3: 218}, 'high': {0: 2507, 1: 2438, 2: 1682, 3: 1635}},
    '60x': {'low': {0: 23, 1: 23, 2: 27, 3: 141}, 'high': {0: 2000, 1: 2059, 2: 1904, 3: 1339}}
}


class AZSequence(Sequence):

    def __init__(self, X, y, batch_size, sample_per_image=1, train_=True, random_subsample_input=True, seed=None, resetseed=False):
        self.rand_instance = Random()
        self.seed = seed
        self.resetseed = resetseed
        
        if self.seed is not None:
            self.rand_instance.seed(self.seed)

        self.x, self.y = X, y
        self.batch_size = batch_size
        self.sample_per_image = sample_per_image
        self.train = train_
        self.random_subsample_input = random_subsample_input

    def __len__(self):
        return math.ceil(len(self.x)*self.sample_per_image / self.batch_size)

    def on_epoch_end(self):
        if self.resetseed is True:
            self.rand_instance.seed(self.seed)

    def get_random_crop(self, image_shape, crop_shape):
        randmax = image_shape-np.array(list(crop_shape))
        topleft = np.array([self.rand_instance.randrange(r) for r in randmax])
        return tuple(slice(s, e) for (s, e) in zip(topleft, topleft+crop_shape))

    @staticmethod
    def read_stack(slice_paths, train_, random_subsample=None):
        # The same x-y crop will be applied to each brightflield slice and even on the fluo targets.
        for idx, im_path in enumerate(slice_paths):
            slice_ = imageio.imread(im_path).astype(np.float32)

            if config.target_size is not None and np.shape(slice_) != config.target_size:
                # Resize
                #slice_ = transform.resize(slice_, config.target_size)
                
                # Crop from top-left
                global_crop = tuple(slice(None, s) for s in config.target_size)
                slice_ = slice_[global_crop]

            """
            if train_:
                slice_ = slice_[:config.splity, :]
            else:
                slice_ = slice_[config.splity:, :]
            """

            if random_subsample is not None:
                slice_ = slice_[random_subsample]

            if idx == 0:
                xy_shape = (len(slice_paths),) + np.shape(slice_)
                image = np.zeros(xy_shape, slice_.dtype)

            image[idx] = slice_
        return image

    @staticmethod
    def augment(image, rotate_angle: int, fliplr_tf: bool, flipud_tf: bool) -> np.ndarray:
        # AZSequence.count += 1
        # print(f"{AZSequence.count} - rotate={rotate_angle}, flip LR={fliplr_tf}, flip UD={flipud_tf}")
        if rotate_angle != 0:
            image = transform.rotate(image, angle=rotate_angle)
        if fliplr_tf:
            image = np.fliplr(image)
        if flipud_tf:
            image = np.flipud(image)

        # if image_in.shape[-1] > 3:
        #     visualize(image_in[..., 3], image[..., 3])
        # else:
        #     visualize(image_in, image)

        return image

    def __getitem__(self, idx):
        def normalize(im, low, high):
            """ Threshold and divide """
            histo = False
            if histo:
                im[im > high] = high
                im = im / high
            else:
                im /= 65535

            plt.imshow(im)
            plt.show()

            return im

        image_idx = idx // self.sample_per_image
        batch_x = self.x[image_idx * self.batch_size:(image_idx + 1) *
        self.batch_size]
        batch_y = self.y[image_idx * self.batch_size:(image_idx + 1) *
        self.batch_size]
        
        #print('Constructing batch:')

        batch_x_images = []
        batch_y_images = []

        """
        if self.train:
            random_subsample = self.get_random_crop((config.splity, 2554), config.sample_crop[:2])
        else:
            random_subsample = self.get_random_crop((2154-config.splity, 2554), config.sample_crop[:2])
        """

        random_subsample = self.get_random_crop((2154, 2554), config.sample_crop[:2])

        def magnification_level(path):
            return os.path.basename(os.path.dirname(path))


        if not self.random_subsample_input:
            random_subsample = None

        rotate_tf = np.random.uniform() < config.rotate_p
        if rotate_tf and config.augment:
            rotate_angle = self.rand_instance.choice([90, 180, 270])
        else:
            rotate_angle = 0
        fliplr_tf = np.random.uniform() < config.fliplr_p
        flipud_tf = np.random.uniform() < config.flipud_p

        import pprint
        pp = pprint.PrettyPrinter(indent=4)

        for batch_elem in batch_x:
            mag_level = magnification_level(batch_elem[0])
            image = self.read_stack(batch_elem, self.train, random_subsample)
            image = np.transpose(image, (1, 2, 0))
            
            # Normalize to [0. 1.]
            for z in range(len(batch_elem)):
                print('Image:', batch_elem[z])
                image[..., z] = normalize(image[..., z], norm[mag_level]['low'][3], norm[mag_level]['high'][3])
                #imageio.imwrite('l/%d-%d.png' % (z, idx), (image[..., z]*255).astype(np.uint8))
            if config.augment and self.train:
                image = self.augment(image, rotate_angle, fliplr_tf, flipud_tf)
            batch_x_images.append(image)

        for batch_elem in batch_y:
            mag_level = magnification_level(batch_elem[0])
            image = self.read_stack(batch_elem, self.train, random_subsample)
            image = np.transpose(image, (1, 2, 0))

            # Normalize to [0. 1.]
            for ch in range(len(batch_elem)):
                print('Image:', batch_elem[ch])
                image[..., ch] = normalize(image[..., ch], norm[mag_level]['low'][ch], norm[mag_level]['high'][ch])

            #imageio.imwrite('l/%d-%d.png' % (z, idx), (image*255).astype(np.uint8))

            if config.augment and self.train:
                image = self.augment(image, rotate_angle, fliplr_tf, flipud_tf)
            batch_y_images.append(image)

        return np.array(batch_x_images), np.array(batch_y_images)


def get_images_list(data_dir):
    if config.local_run:
        image_paths_ = glob('%s/*/input/*' % data_dir)
        label_paths_ = glob('%s/*/targets/*' % data_dir)
    else:
        image_paths_ = []
        label_paths_ = []
        magnifications = ['20x', '40x', '60x']

        magnifications = [magnifications[2]]

        for magnification in magnifications:
            glob_im = '%s/%s/*A04*' % (data_dir, magnification)
            print(glob_im)
            image_paths_ += glob(glob_im)
            for ch in ['1', '2', '3']:
                glob_lab = '%s/%s/*A0%s*' % (data_dir, magnification, ch)
                label_paths_ += glob(glob_lab)

    return image_paths_, label_paths_

def info(im_name):
    im_bname = os.path.basename(im_name)
    experiment = im_bname[:len('AssayPlate_Greiner_#655090')]
    well = im_bname[len('AssayPlate_Greiner_#655090_'):len('AssayPlate_Greiner_#655090_')+3]
    field = im_bname[len('AssayPlate_Greiner_#655090_D04_T0001'):len('AssayPlate_Greiner_#655090_D04_T0001')+4]

    return experiment, well, field

def get_dataset(data_dir, train_, sample_per_image=60, random_subsample_input=True, seed=None, resetseed=None, filter_fun=None):    
    image_paths_, label_paths_ = get_images_list(data_dir)
    print('Number of image files discovered:', len(image_paths_))
    print('Number of label files discovered:', len(label_paths_))

    image_paths, label_paths = [], []
    def filter_(paths, filter_fun_):
        result_ = []
        for path_ in paths:
            if filter_fun_(path_):
                result_.append(path_)

        return result_

    image_paths, label_paths = image_paths_, label_paths_

    if filter_fun is not None:
        image_paths = filter_(image_paths_, filter_fun)
        label_paths = filter_(label_paths, filter_fun)

    print('Number of selected files:', len(image_paths))
    print('Number of selected labels:', len(label_paths))

    label_paths.sort()
    image_paths.sort()

    '''
    Image format:   AssayPlate_Greiner_#655090_D04_T0001F006L01A04Z03C04.tif - Only the Z varies
    Label format:   AssayPlate_Greiner_#655090_D04_T0001F006L01A01Z01C01.tif - A and C varies between C01 and C03, Z= 01
    
    Fluo channels:
    C01=nuclei  (Red)
    C02=lipids  (Green)
    C03=cyto    (Blue)

    D04: well
    T0001: timepoint (irrelevant)
    F006: FOV (site)
    L01: timeline (irrelevant)
    A04: action (fluorescents + brightfield)
    Z03: slice number
    C04: same as action


    20x:    48
    40x:    64
    60x:    96

    '''

    def get_im_id(im_path):
        base = os.path.basename(im_path)
        im_id = base[:len('AssayPlate_Greiner_#655090_D04_T0001F006')]
        return im_id

    def get_res(im_path):
        if config.local_run:
            return os.path.basename(
                os.path.dirname(os.path.dirname(im_path)))[:3]
        else:
            return os.path.basename(
                    os.path.dirname(im_path))[:3]

    images, labels = defaultdict(list), defaultdict(list)

    for image in image_paths:
        k = (get_im_id(image), get_res(image))
        images[k].append(image)

    for label in label_paths:
        k = get_im_id(label), get_res(label)
        labels[k].append(label)

    x, y = [], []

    for k in labels.keys():
        #print('Image found:', k)
        x.append(images[k])
        y.append(labels[k])

    return AZSequence(
        x, y, batch_size=1, sample_per_image=sample_per_image, train_=train_, 
        random_subsample_input=random_subsample_input, resetseed=resetseed, seed=seed)

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

    model.compile(optimizer='adam', loss=channelwise_loss)
    return model


def train(sequences, model):
    train, val = sequences
    mcp_save = tf.keras.callbacks.ModelCheckpoint('%s/model-{epoch:04d}-{val_loss:.4f}.h5' % config.output_dir, save_best_only=True, monitor='val_loss', mode='min')
    callbacks = []
    
    if not config.readonly:
        callbacks += [mcp_save]

    model.fit(train, validation_data=val, epochs=400, callbacks=callbacks, initial_epoch=config.initial_epoch, steps_per_epoch=200)

    return model

def predict_tiled(x, y_channels, tile_sizes):
    print(np.shape(x))
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
            
            y_pred_sample = y_pred[batch_element]

            y_pred_sample_normalized = np.zeros_like(y_pred_sample)
            for ch in range(np.shape(y_pred_sample)[-1]):
                y_pred_sample_normalized[..., ch] = y_pred_sample[..., ch] / 65535

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

        plt.show()
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
    if not config.readonly:
        os.makedirs(config.output_dir, exist_ok=True)

    # Leave out wells for validation
    lo_ws = ['B03']

    train_sequence = get_dataset(
        config.data_dir, 
        train_=True, 
        sample_per_image=20, 
        random_subsample_input=True, 
        seed=config.seed, 
        filter_fun=lambda im: info(im)[1] not in lo_ws)
    val_sequence = get_dataset(
        config.data_dir, 
        train_=False, 
        sample_per_image=8, 
        random_subsample_input=True, 
        seed=config.seed, 
        resetseed=True,
         filter_fun=lambda im: info(im)[1] in lo_ws)

    #test_sequence = get_dataset(config.data_dir, train_=False, sample_per_image=1, random_subsample_input=False)

    model = get_network()

    if config.init_weights is not None:
        print('Loading weights:', config.init_weights)
        model.load_weights(config.init_weights)

    if config.train == True:
        model = train((train_sequence, val_sequence), model)

    
    #test(test_sequence, model, tile_sizes=tuple(config.sample_crop[:2]), save=config.save)
    test(train_sequence)
