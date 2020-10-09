from glob import glob
from collections import defaultdict
import os
import math

import numpy as np
from tensorflow.keras.utils import Sequence
import imageio
import matplotlib.pyplot as plt

import config
import init

class AZSequence(Sequence):

    def __init__(self, X, y, batch_size):
        self.x, self.y = X, y
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    @staticmethod
    def read_stack(slice_paths, normalize=False):
        for idx, im_path in enumerate(slice_paths):
            slice_ = imageio.imread(im_path)
            if idx == 0:
                xy_shape =  (len(slice_paths),) + np.shape(slice_)
                image = np.zeros(xy_shape, np.uint16)
            
            if normalize:
                image[idx] = (slice_/np.max(slice_))*255
            else:
                image[idx] = slice_
        return image

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        
        print('Constructing batch:')

        batch_x_images = []
        batch_y_images = []

        for batch_elem in batch_x:
            image = self.read_stack(batch_elem)
            batch_x_images.append(image)
            print('Batch X shape:', np.shape(image))

        for batch_elem in batch_y:
            image = self.read_stack(batch_elem, True)
            batch_y_images.append(image)
            print('Batch y shape:', np.shape(image))

        return np.array(batch_x_images), np.array(batch_y_images)

def get_dataset(data_dir):
    image_paths = glob('%s/*/input/*' % data_dir)
    label_paths = glob('%s/*/targets/*' % data_dir)

    label_paths.sort()
    image_paths.sort()

    '''
    Image format:   AssayPlate_Greiner_#655090_D04_T0001F006L01A04Z03C04.tif - Only the Z varies
    Label format:   AssayPlate_Greiner_#655090_D04_T0001F006L01A01Z01C01.tif - A and C varies between C01 and C03, Z= 01
    
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

    return AZSequence(x, y, batch_size=1)

if __name__ == '__main__':
    sequence = get_dataset(config.data_dir)

    for x, y in sequence:
        batch_element = 0
        z_pos = len(x[batch_element])//2
        x_im = x[batch_element][z_pos]
        y_im = y[batch_element]

        print(np.shape(x_im))
        print(np.shape(y_im))

        plt.subplot(121, title='Brightfield@Z=%d' % z_pos)
        plt.imshow(x_im)

        plt.subplot(122, title='Fluorescent (merged)')
        plt.imshow(np.transpose(y_im, (1, 2, 0)))

        plt.show()