from glob import glob
from collections import defaultdict
import os
import math
import random

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

class AZSequence(Sequence):

    def __init__(self, X, y, batch_size, sample_per_image=1):
        self.x, self.y = X, y
        self.batch_size = batch_size
        self.sample_per_image = sample_per_image

    def __len__(self):
        return math.ceil(len(self.x)*self.sample_per_image / self.batch_size)

    @staticmethod
    def get_random_crop(image_shape, crop_shape):
        randmax = image_shape-np.array(list(crop_shape))
        topleft = np.array([random.randrange(r) for r in randmax])
        return tuple(slice(s, e) for (s, e) in zip(topleft, topleft+crop_shape))


    @staticmethod
    def read_stack(slice_paths, normalize=False, random_subsample=False):
        # The same x-y crop will be applied to each brightflield slice and even on the fluo targets.
        for idx, im_path in enumerate(slice_paths):
            slice_ = imageio.imread(im_path).astype(np.float32)

            if normalize:
                slice_ = slice_/np.max(slice_)

            if np.shape(slice_) != config.target_size:
                # Resize
                #slice_ = transform.resize(slice_, config.target_size)
                
                # Crop from top-left
                global_crop = tuple(slice(None, s) for s in config.target_size)
                slice_ = slice_[global_crop]

            if random_subsample is not None:
                slice_ = slice_[random_subsample]

            if idx == 0:
                xy_shape =  (len(slice_paths),) + np.shape(slice_)
                image = np.zeros(xy_shape, slice_.dtype)
            
            image[idx] = slice_
        return image


    @staticmethod
    def augment(image, rotate_tf: bool, fliplr_tf: bool, flipud_tf: bool) -> np.ndarray:
        if rotate_tf:
            angle = np.random.choice([90,180,270])
            image = transform.rotate(image, angle=angle)
        if fliplr_tf:
            image = np.fliplr(image)
        if flipud_tf:
            image = np.flipud(image)
        return image


    def __getitem__(self, idx):
        image_idx = idx // self.sample_per_image
        batch_x = self.x[image_idx * self.batch_size:(image_idx + 1) *
        self.batch_size]
        batch_y = self.y[image_idx * self.batch_size:(image_idx + 1) *
        self.batch_size]
        
        #print('Constructing batch:')

        batch_x_images = []
        batch_y_images = []

        random_subsample = AZSequence.get_random_crop(config.target_size, config.sample_crop[:2])

        rotate_tf = np.random.uniform() < config.rotate_p
        fliplr_tf = np.random.uniform() < config.fliplr_p
        flipud_tf = np.random.uniform() < config.flipud_p

        for batch_elem in batch_x:
            image = self.read_stack(batch_elem, True, random_subsample)
            image = np.transpose(image, (1, 2, 0))
            if config.augment:
                image = self.augment(image, rotate_tf, fliplr_tf, flipud_tf)
            batch_x_images.append(image)
            # print('Batch X shape:', np.shape(image))

        for batch_elem in batch_y:
            image = self.read_stack(batch_elem, True, random_subsample)
            image = np.transpose(image, (1, 2, 0))
            if config.augment:
                image = self.augment(image, rotate_tf, fliplr_tf, flipud_tf)
            batch_y_images.append(image)
            # print('Batch y shape:', np.shape(image))

        return np.array(batch_x_images), np.array(batch_y_images)

def get_dataset(data_dir):
    image_paths = glob('%s/*/input/*' % data_dir)
    label_paths = glob('%s/*/targets/*' % data_dir)

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
    '''

    def get_im_id(im_path):
        base = os.path.basename(im_path)
        im_id = base[:len('_D04_T0001F008L01A04Z04C04.tif')]
        return im_id

    def get_res(im_path):
        return os.path.basename(os.path.dirname(os.path.dirname(im_path)))[:3]

    images, labels = defaultdict(list), defaultdict(list)

    for image in image_paths:
        k = (get_im_id(image), get_res(image))
        images[k].append(image)

    for label in label_paths:
        k = get_im_id(label), get_res(label)
        labels[k].append(label)

    x, y = [], []

    for k in labels.keys():
        print('Image found:', k)
        x.append(images[k])
        y.append(labels[k])

    return AZSequence(x, y, batch_size=1, sample_per_image=20)


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
    fluo_channels = Conv2D(3, (1, 1), name='fluo_channels')(unet_out)
    
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
        for ch in [1, 2]:
            total_loss += MeanSquaredError()(y_true[..., ch], y_pred[..., ch])
        
        # The first channel is the nuclei
        # Most of the pixels below the intensity 600 are the part of the background and correlates with the cyto.
        # Therefore we concentrate on the >600 ground truth pixels. (600 ~ .1 after normalization)
        nuclei_weight = .8
        nuclei_thresh = .1
        
        nuclei_weight_tensor = nuclei_weight*tf.cast(y_true[..., 0] > nuclei_thresh, tf.float32) + (1.-tf.cast(y_true[..., 0] <= nuclei_thresh, tf.float32))
        
        nuclei_loss = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(
            y_true[..., 0], 
            y_pred[..., 0]
        )
        
        total_loss += tf.math.reduce_mean(nuclei_loss * nuclei_weight_tensor)

        return total_loss

    model.compile(optimizer='sgd', loss=channelwise_loss)
    return model


def train(sequence, model):
    model.fit(sequence, epochs=200)
    if config.save_checkpoint is not None:
        model.save_weights(config.save_checkpoint)

    return model
'''
If the model is set, it predicts the image using the model passed and shows the result.
'''
def test(sequence, model=None, save=False):
    for idx, (x, y) in enumerate(sequence):
        batch_element = 0
        plot_layout = 120

        x_sample, y_sample = x[batch_element], y[batch_element]
        
        z_pos = np.shape(x_sample)[-1]//2
        x_im, y_im = x_sample[..., z_pos], y_sample

        if model is not None:
            plot_layout = 220
            y_pred = model.predict(x)
            y_pred_sample = y_pred[batch_element]

            y_pred_sample_normalized = np.zeros_like(y_pred_sample)
            for ch in range(np.shape(y_pred_sample)[-1]):
                y_pred_sample_normalized[..., ch] = y_pred_sample[..., ch] / np.max(y_pred_sample[..., ch])

            plt.subplot(plot_layout + 3, title='Predicted fluorescent')
            plt.imshow(y_pred_sample)

            plt.subplot(plot_layout + 4, title='Predicted fluorescent (ch normalized)')
            plt.imshow(y_pred_sample_normalized)

            if save:
                imageio.imwrite(os.path.join(config.output_dir, '%d_pred.tif' % idx), y_pred_sample)

        plt.subplot(plot_layout + 1, title='Brightfield@Z=%d' % z_pos)
        plt.imshow(x_im)

        plt.subplot(plot_layout + 2, title='Fluorescent (merged)')
        plt.imshow(y_im)

        plt.show()

        if save:
            bright = (x_sample[..., 1]*255).astype(np.uint8)
            fluo = (y_sample*255).astype(np.uint8)
            
            imageio.imwrite(os.path.join(config.output_dir, '%d_bright.tif' % idx, bright))
            imageio.imwrite(os.path.join(config.output_dir, '%d_fluo.tif' % idx, fluo))



    model.load_weights(config.save_checkpoint)

if __name__ == '__main__':
    os.makedirs(config.output_dir, exist_ok=True)

    sequence = get_dataset(config.data_dir)

    model = get_network()
    model = train(sequence, model)
    if config.save_checkpoint is not None:
        model.load_weights(config.save_checkpoint)
    
    # A (tranied) model can be passed to see the results.
    test(sequence, model)