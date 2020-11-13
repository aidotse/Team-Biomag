import os
import math
from random import Random
from glob import glob
from collections import defaultdict
import pprint

import numpy as np
import imageio
from tensorflow.keras.utils import Sequence
from skimage import transform

import init
import config

import misc


def load_limits():
    print('Loading limits statistics to scale the intensities.')
    norm = {}
    for magnification in config.magnifications:
        norm[magnification] = misc.get_json(config.limits_file % magnification)

    return norm

def load_stats():
    print('Loading std/mean for the datasets...')

    stats = {}
    for magnification in config.magnifications:
        stats[magnification] = misc.get_json('x-stat-%s.json' % magnification)

    return stats

def standardize(rawim, mean, std):
    return (rawim-mean)/std

def unstandardize(stdim, mean, std):
    return (stdim*std)+mean

def standardize_bright(a_bright, mean, std, inverse=False):
    fun = standardize
    if inverse is True:
        fun = unstandardize
    
    bright = a_bright.copy()
    for z in range(np.shape(bright)[-1]):
        bright[..., z] = fun(bright[..., z], mean, std)

    return bright


def standardize_fluo(a_fluo, mean, std, inverse=False):
    fun = standardize
    if inverse is True:
        fun = unstandardize

    fluo = a_fluo.copy()
    for ch in range(np.shape(fluo)[-1]):
        fluo[..., ch] = fun(fluo[..., ch], mean[ch], std[ch])

    return fluo

def normalize(im, low, high, histo = True):
    """
    Thresholds the image then translates to the interval [0, 1]
    """
    '''
    if histo:
        im[im > high] = high
        im[im < low] = low
        im -= low
        im = im / (high-low)
    else:
        im /= np.max(im)

    '''
    return im

def denormalize(im, low, high):
    return im*(high-low)+low


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
        self.norm = load_limits()
        self.stats = load_stats()
        self.return_meta = False

    def __len__(self):
        return math.ceil(len(self.x)*self.sample_per_image / self.batch_size)

    def on_epoch_end(self):
        if not config.readonly:
            os.makedirs(config.output_dir, exist_ok=True)

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
            #slice_ = imageio.imread(im_path)

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

            if config.resize_crop is not None:
                slice_ = transform.resize(slice_, config.resize_crop, anti_aliasing=False, order=1) #order=1 -> bilinear

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

        random_subsample = self.get_random_crop(config.target_size, config.sample_crop[:2])

        if not self.random_subsample_input:
            random_subsample = None

        rotate_tf = np.random.uniform() < config.rotate_p
        if rotate_tf and config.augment:
            rotate_angle = self.rand_instance.choice([0, 180])
        else:
            rotate_angle = 0
        fliplr_tf = np.random.uniform() < config.fliplr_p
        flipud_tf = np.random.uniform() < config.flipud_p

        mag_level = misc.magnification_level(batch_x[0][0])
        bright_stats = self.stats[mag_level]['mean'][-1], self.stats[mag_level]['std'][-1]
        fluo_stats = self.stats[mag_level]['mean'][:3], self.stats[mag_level]['std'][:3]

        for batch_elem in batch_x:
            image = self.read_stack(batch_elem, self.train, random_subsample)
            image = np.transpose(image, (1, 2, 0))

            image = standardize_bright(image, *bright_stats)

            if config.augment and self.train:
                image = self.augment(image, rotate_angle, fliplr_tf, flipud_tf)
            batch_x_images.append(image)

        for batch_elem in batch_y:
            if config.include_nuclei_channel:
                batch_elem_0 = batch_elem[0]
                batch_elem_0 = os.path.basename(batch_elem_0)
                nuclei_path_ = os.path.join(config.cp_nuclei_path, mag_level, batch_elem_0 + 'f')
        
                image = self.read_stack(batch_elem + [nuclei_path_], self.train, random_subsample)
            else:
                image = self.read_stack(batch_elem, self.train, random_subsample)
            image = np.transpose(image, (1, 2, 0))

            if config.include_nuclei_channel:
                image[..., 3] = (image[..., 3] > 0).astype(image.dtype)


            '''
            Test standardization and then invert...

            import matplotlib.pyplot as plt
            plt.subplot(131, title='Before')
            plt.imshow(image[..., 1])

            print('Min max before', np.min(image), np.max(image))

            image = standardize_fluo(image, *fluo_stats)
            plt.subplot(132, title='After stdized')
            plt.imshow(image[..., 1])

            print('Min max after std', np.min(image), np.max(image))

            plt.subplot(133, title='Std inverted')
            inverted = standardize_fluo(image, *fluo_stats, inverse=True)
            plt.imshow(inverted[..., 1])
            print('Min max after std', np.min(inverted), np.max(inverted))
            
            plt.show()

            if config.augment and self.train:
                image = self.augment(image, rotate_angle, fliplr_tf, flipud_tf)
            batch_y_images.append(image)
            '''

        if self.return_meta:
            return np.array(batch_x_images), np.array(batch_y_images), batch_x
        else:
            return np.array(batch_x_images), np.array(batch_y_images)


def get_images_list(data_dir, magnifications):
    image_paths_ = []
    label_paths_ = []

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
    print('Magnifications to use: %s' % config.magnifications)
    image_paths_, label_paths_ = get_images_list(data_dir, config.magnifications)
    print('Number of image files discovered (7/FoV):', len(image_paths_))
    print('Number of label files discovered (3/FoV):', len(label_paths_))

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

    print('Number of image files selected (7/FoV):', len(image_paths))
    print('Number of label files selected (3/FoV):', len(label_paths))

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


    20x:    48 FOV: 8 well, 6 field: 48*7 briht=336+144 fluo
    40x:    64 FOV: 8 well, 8 field: 64*7 bright=448+192 fluo
    60x:    96 FOV: 8 well, 12 field: 96*7 bright=672+288 fluo
    All:    208 FOV: btiht=1456+624 fluo
    '''

    def get_im_id(im_path):
        base = os.path.basename(im_path)
        im_id = base[:len('AssayPlate_Greiner_#655090_D04_T0001F006')]
        return im_id

    def get_res(im_path):
        return os.path.basename(
                os.path.dirname(im_path))[:3]

    images, labels = defaultdict(list), defaultdict(list)

    print('Dataset inputs: train=', train_)
    for image in image_paths:
        print('Adding:\t', image)
        k = (get_im_id(image), get_res(image))
        images[k].append(image)

    print('Dataset labels: train=', train_)
    for label in label_paths:
        print('Adding:\t', label)
        k = get_im_id(label), get_res(label)
        labels[k].append(label)

    x, y = [], []

    for k in labels.keys():
        x.append(images[k])
        y.append(labels[k])

    return AZSequence(
        x, y, batch_size=1, sample_per_image=sample_per_image, train_=train_, 
        random_subsample_input=random_subsample_input, resetseed=resetseed, seed=seed)
