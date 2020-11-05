import math
from random import Random

import imageio
from tensorflow.keras.utils import Sequence
import numpy as np

import config


def cantor_pairing(a, b):
    return (a + b) * (a + b + 1) // 2 + b if a is not None and b is not None else None

def inv_cantor_pairing(c):
    w = math.floor((math.sqrt(8*c+1)-1)/2)
    t = (w**2+w)/2
    y = c-t
    x = w-y
    return int(x), int(y)


class BaseGenerator(Sequence):
    def __init__(self, n_inputs, batch_size: int, shuffle=True, random_seed=None):
        self.n_inputs = n_inputs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.last_batch = None
        self.indices = None
        self._rn = None
        self.generate_dataset(self.random_seed)

    def __len__(self):
        return len(self.indices) + (self.last_batch is not None)

    def get_indices(self, index):
        return self.indices[index] if self.last_batch is None else self.last_batch

    def __getitem__(self, index):
        return self.get_indices(index)

    def on_epoch_end(self):
        self.generate_dataset()

    def generate_dataset(self, new_random_seed=None):
        self.indices = np.arange(self.n_inputs)
        # self.indices = self.indices[:self.n_inputs - (self.n_inputs % self.batch_size)]
        self._rn = None
        if self.shuffle:
            if self.random_seed is not None:
                self.random_seed = new_random_seed if new_random_seed is not None else self.random_seed + 1
                np.random.seed(self.random_seed)
            else:
                self._rn = np.random.choice(len(self.n_inputs))
            np.random.shuffle(self.indices)
        self.last_batch = self.indices[self.n_inputs - (self.n_inputs % self.batch_size):] \
            if (self.n_inputs % self.batch_size) > 0 else None
        self.indices = self.indices[:self.n_inputs - (self.n_inputs % self.batch_size)].reshape(-1, self.batch_size)



class AugmentationGenerator(BaseGenerator):
    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int, rotate=True, flip_ud=True, flip_lr=True,
                 crop_size=None, crop_stride=1, transform_y=False, data_format="channels_last", shuffle=True,
                 random_seed=None, **kwargs):
        self.add_noise = kwargs["noise"] if "noise" in kwargs else False
        self.add_cosmic_ray = kwargs["cosmic_ray"] if "cosmic_ray" in kwargs else False
        # translation_mask = (int(self.add_noise) << 5) | (int(self.add_cosmic_ray) << 4)\
        #                    | (int(flip_ud) << 3) | int(flip_lr) << 2 | int(rotate) * 3
        translation_mask = (int(flip_ud) << 3) | int(flip_lr) << 2 | int(rotate) * 3
        if flip_lr and flip_ud and rotate:
            translation_mask = translation_mask & ~0b100
        self.translation_params = np.unique(np.arange(32) & translation_mask)
        self.translation_params = np.repeat(self.translation_params, len(x))
        n_inputs = len(self.translation_params)
        self.true_n_inputs = len(x)
        self.x = x
        self.y = y
        self.preprocess_x = None
        self.transform_y = transform_y
        # translation order: noise, cosmic ray, flip_ud, flip_lr, rotate

        self.noise_b_mask = 1<<5
        self.cr_b_mask = 1<<4
        self.f_ud_b_mask = 1<<3
        self.f_lr_b_mask = 1<<2
        self.rot_b_mask = 0b11

        self.rotate = rotate
        self.flip_ud = flip_ud
        self.flip_lr = flip_lr
        self.crop_stride = crop_stride
        self.crop_count = -1
        if type(crop_stride) == int:
            self.crop_stride = [crop_stride]*2
        self.crop_size = crop_size
        if type(crop_size) == int:
            self.crop_size = [crop_size]*2
        self.crop_indices = None
        self.data_format = data_format
        if data_format=="channels_last":
            self.h_dim = 1
            self.w_dim = 2
            self.c_dim = 3
        elif data_format=="channels_first":
            self.c_dim = 1
            self.h_dim = 2
            self.w_dim = 3
        else:
            raise ValueError("data_format must be one of 'channels_first' or 'channels_last'")
        self.initialize_crop_indices(random_seed)
        if crop_size is not None:
            n_inputs *=self.n_crops
        super().__init__(n_inputs, batch_size, shuffle, random_seed)


    @property
    def shape(self):
        return self.x.shape[1:]

    @property
    def n_crops(self):
        return len(self.crop_indices) if self.crop_indices is not None else 0

    @property
    def n_translations(self):
        return len(self.translation_params)

    def __getitem__(self, index):
        idx = super().get_indices(index)
        batch_x = self.x[idx % self.true_n_inputs]
        if self.preprocess_x is not None:
            batch_x = self.preprocess_x(batch_x)
        batch_x = self.translate(batch_x, index, idx)

        batch_y = self.y[idx % self.true_n_inputs]
        if self.transform_y:
            batch_y = self.translate(batch_y, index, idx)
        return batch_x, batch_y

    def translate(self, batch, batch_index, item_idx):
        batch[(self.translation_params[item_idx % self.n_translations] & self.f_ud_b_mask).astype(np.bool)] = \
            np.flip(batch[(self.translation_params[item_idx % self.n_translations] & self.f_ud_b_mask).astype(np.bool)],
                    self.h_dim)
        batch[(self.translation_params[item_idx % self.n_translations] & self.f_lr_b_mask).astype(np.bool)] = \
            np.flip(batch[(self.translation_params[item_idx % self.n_translations] & self.f_lr_b_mask).astype(np.bool)],
                    self.w_dim)
        for i in range(4):
            batch[self.translation_params[item_idx % self.n_translations] & self.rot_b_mask == i] = \
                np.rot90(batch[self.translation_params[item_idx % self.n_translations] & self.rot_b_mask == i], i,
                         (self.h_dim, self.w_dim))
        if self.crop_size is not None:
            slice_arr = self.crop_indices[batch_index // self.n_translations]
            batch = batch[slice_arr]
        return batch

    def generate_dataset(self, new_random_seed=None):
        super().generate_dataset(new_random_seed)
        self.initialize_crop_indices(self.random_seed)

    def initialize_crop_indices(self, random_seed):
        if self.crop_size is not None:
            self.crop_indices = self.RandomIndices(self.shape, self.crop_size, self.crop_stride,
                                                   self.h_dim, self.w_dim, self.c_dim, random_seed)

    class RandomIndices:
        def __init__(self, shape, crop_size, stride, h_dim, w_dim, c_dim, shuffle=True, random_seed=None):
            self.shape = shape
            self.indices = None
            self.crop_size = crop_size
            self.h_dim = h_dim
            self.w_dim = w_dim
            self.c_dim = c_dim
            self.crop_stride = stride
            self.indices = np.transpose(
                np.indices(((self.shape[self.h_dim - 1] - self.crop_size[0]) // self.crop_stride[0]+1,
                            (self.shape[self.w_dim - 1] - self.crop_size[1]) // self.crop_stride[1]+1)), (1, 2, 0)) \
                               .reshape(-1, 2) * self.crop_stride
            if shuffle:
                if random_seed is not None:
                    np.random.seed(random_seed)
                np.random.shuffle(self.indices)

        def __getitem__(self, idx):
            crop_index = self.indices[idx]
            slice_arr = [None] * (len(self.shape)+1)
            slice_arr[0] = slice(None)
            slice_arr[self.h_dim] = slice(crop_index[0], crop_index[0] + self.crop_size[0])
            slice_arr[self.w_dim] = slice(crop_index[1], crop_index[1] + self.crop_size[1])
            slice_arr[self.c_dim] = slice(None)
            return tuple(slice_arr)

        def __len__(self):
            return len(self.indices)


class CPSequence(AugmentationGenerator):

    def __init__(self, X, y, batch_size, train_=True, seed=None, shuffle=True, transform=True):
        self.rand_instance = Random()
        cols = np.genfromtxt(y, np.str, delimiter=",", max_rows=1)
        y = np.loadtxt(y, delimiter=",", usecols=[i for i in range(len(cols))
                                                       if cols[i] != "ImageNumber"
                                                       and not cols[i].startswith("Metadata")], skiprows=1)
        y = (y-y.mean(axis=0, keepdims=True))/y.std(axis=0, keepdims=True)
        self.train = train_

        super().__init__(np.asarray(X).reshape(-1, 3), y, batch_size,
                         rotate=transform, flip_ud=transform, flip_lr=transform,
                         shuffle=shuffle, random_seed=seed)
        self.preprocess_x = lambda batch_x: np.concatenate(
            list(map(lambda p: np.expand_dims(np.transpose(self.read_stack(p, self.train, True),
                                                           (1, 2, 0)), 0),
                     batch_x)), axis=0)

    @staticmethod
    def read_stack(slice_paths, train_, normalize=False, random_subsample=None):
        # The same x-y crop will be applied to each brightflield slice and even on the fluo targets.
        for idx, im_path in enumerate(slice_paths):
            slice_ = imageio.imread(im_path).astype(np.float32)

            if normalize:
                slice_ = slice_ / np.max(slice_)

            if config.target_size is not None and np.shape(slice_) != config.target_size:
                # Resize
                # slice_ = transform.resize(slice_, config.target_size)

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


class U_CPSequence(CPSequence):

    def __init__(self, X, y, f, batch_size, train_=True, seed=None, shuffle=True, transform=True):
        super().__init__(y, f, batch_size, train_, seed, shuffle, transform)
        self.br = np.asarray(X).reshape(-1, 7)

    def __getitem__(self, index):
        batch_y, batch_f = super().__getitem__(index)
        idx = super().get_indices(index)
        batch_x = self.br[idx % self.true_n_inputs]
        batch_x = self.preprocess_x(batch_x)
        batch_x = self.translate(batch_x, index, idx)
        return batch_x, [batch_y, batch_f]
