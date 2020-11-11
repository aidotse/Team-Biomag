import os
from collections import defaultdict

import config
import dataset
import init
import stardist_blocks as sd
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.layers import Input, Dense, Conv2D, Layer
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import tensorflow.keras.backend as K

from generators import CPSequence, U_CPSequence


class NormLayer(Layer):
    def __init__(self, low, high, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(False, name, dtype, dynamic, **kwargs)
        self.low = tf.cast(low, tf.float32)
        self.high = tf.cast(high, tf.float32)
        for i in range(3):
            self.low = tf.expand_dims(self.low, 0)
            self.high = tf.expand_dims(self.high, 0)

    def call(self, inputs, **kwargs):
        return self.normalize(inputs, self.low, self.high)
    
    @staticmethod
    def normalize(inputs, low, high):
        out = tf.identity(inputs)
        out -= low
        out = out / (high-low)
        out = tf.clip_by_value(out, 0., 1.)
        return out


class DenormLayer(Layer):
    def __init__(self, low, high, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(False, name, dtype, dynamic, **kwargs)
        self.low = tf.cast(low, tf.float32)
        self.high = tf.cast(high, tf.float32)
        for i in range(3):
            self.low = tf.expand_dims(self.low, 0)
            self.high = tf.expand_dims(self.high, 0)

    def call(self, inputs, **kwargs):
        return dataset.denormalize(inputs, self.low, self.high)



def CP(input_shape, n_features):
    """ CP feature predictor network. Uses ResNet50 V2.

    :param input_shape: shape of the input image
    :param n_features: number of CP features to predict
    :return: cp_net function to create the network.

    Example:
    input = Input(input_shape)
    cp = CP(n_features, input_shape)(input)
    model = Model(input, cp)
    ...
    """
    def cp_net(input):
        resnet = ResNet50V2(False, None, input_tensor=input, input_shape=input_shape, pooling="avg")
        out = Dense(n_features, name="CP_features")(resnet.output)
        return out

    return cp_net


def Unet(input_layer, **kwargs):
    unet = sd.unet_block(**kwargs)(input_layer)
    fluo_channels = Conv2D(3, (1, 1), name='fluo_channels', activation='sigmoid')(unet)
    return fluo_channels


def U_CP(input_shape, n_features, norm_bounds=None, **kwargs):
    """ U-Net-CP network combined for brightfield to fluorescent generator with CP feature predictor

    :param input_shape:
    :param n_features:
    :param kwargs: keyword arguments for unet_block
    :return:
    """
    input_layer = Input(shape=input_shape, dtype=tf.float32)
    if norm_bounds is not None:
        norm_layer = NormLayer(norm_bounds[0][0], norm_bounds[1][0], name="norm")(input_layer)
    unet = Unet(input_layer if norm_bounds is None else norm_layer, n_filter_base=64, **kwargs)
    if norm_bounds is not None:
        unet = DenormLayer(norm_bounds[0][1:], norm_bounds[1][1:], name="denorm")(unet)
    cp_net = CP(input_shape, n_features)(unet)

    return Model(input_layer, [unet, cp_net])


def filter_fun(wells):
    return lambda im: dataset.info(im)[1] in wells


def get_cp_dataset(im_dir, feature_file, norm_med=False, seed=None, wells=None, use_crop_id=False, batch_size=1):
    """

    :param im_dir: folder containing the input images
    :param feature_file: output .csv file from CP
    :param norm_med: True if input should be normalized (division by median), median to divide features with, or False if
                        no normalization is needed
    :param seed: random seed for the generator
    :param wells: wells to include from the dataset
    :param use_crop_id: True if input images have crop ids on the end of the files ("like '..._xx.tif')
    :return: CPSequence instance (generator)
    """
    fluor_paths_ = os.listdir(im_dir)
    fluor_paths_ = filter(lambda p: p[-10:-7] in ["C%.2d" % i for i in range(1, 4)], fluor_paths_) if use_crop_id\
        else filter(lambda p: p[-7:-4] in ["C%.2d" % i for i in range(1, 4)], fluor_paths_)
    fluor_paths_ = list(map(lambda p: os.path.join(im_dir, p), fluor_paths_))
    def filter_(paths, filter_fun_):
        result_ = []
        for path_ in paths:
            if filter_fun_(path_):
                result_.append(path_)

        return result_

    fluor_paths = fluor_paths_

    if wells is not None:
        fluor_paths = filter_(fluor_paths_, filter_fun(wells))

    fluor_paths.sort()

    def get_im_id(im_path):
        base = os.path.basename(im_path)
        im_id = base[:len('AssayPlate_Greiner_#655090_D04_T0001F006')]
        return im_id

    def get_res(im_path):
        return os.path.basename(
            os.path.dirname(os.path.dirname(im_path)))[:3]

    def get_crop_id(im_path):
        crop_id = im_path[-6:-4]
        return crop_id

    fluors = defaultdict(list)

    for label in fluor_paths:
        k = (get_im_id(label), get_res(label)) if not use_crop_id else (
        get_im_id(label), get_res(label), get_crop_id(label))
        fluors[k].append(label)

    x = []

    for k in fluors.keys():
        #print('Image found:', k)
        x.append(fluors[k])

    return CPSequence(
        x, feature_file, batch_size=batch_size, norm_med=norm_med, seed=seed, used_wells=wells)

def get_u_cp_dataset(im_dir, feature_file, norm_med=False, seed=None, wells=None, use_crop_id=False, batch_size=1):

    """
    :param im_dir: folder containing the input images
    :param feature_file: output .csv file from CP
    :param norm_med: True if input should be normalized (division by median), median to divide features with, or False if
                        no normalization is needed
    :param seed: random seed for the generator
    :param wells: wells to include from the dataset
    :param use_crop_id: True if input images have crop ids on the end of the files ("like '..._xx.tif')
    :return: U_CPSequence instance (generator)
    """
    path_list = os.listdir(im_dir)
    label_paths_ = filter(lambda p: p[-10:-7] in ["C%.2d" % i for i in range(1, 4)], path_list) if use_crop_id\
        else filter(lambda p: p[-7:-4] in ["C%.2d" % i for i in range(1, 4)], path_list)
    label_paths_ = list(map(lambda p: os.path.join(im_dir, p), label_paths_))

    image_paths_ = filter(lambda p: p[-10:-7] == "C04", path_list) if use_crop_id\
        else filter(lambda p: p[-7:-4] == "C04", path_list)
    image_paths_ = list(map(lambda p: os.path.join(im_dir, p), image_paths_))

    def filter_(paths, filter_fun_):
        result_ = []
        for path_ in paths:
            if filter_fun_(path_):
                result_.append(path_)

        return result_

    image_paths, label_paths = image_paths_, label_paths_

    if wells is not None:
        image_paths = filter_(image_paths_, filter_fun(wells))
        label_paths = filter_(label_paths, filter_fun(wells))

    label_paths.sort()
    image_paths.sort()

    def get_im_id(im_path):
        base = os.path.basename(im_path)
        im_id = base[:len('AssayPlate_Greiner_#655090_D04_T0001F006')]
        return im_id

    def get_res(im_path):
        return os.path.basename(
            os.path.dirname(os.path.dirname(im_path)))[:3]

    def get_crop_id(im_path):
        crop_id = im_path[-6:-4]
        return crop_id

    images, labels = defaultdict(list), defaultdict(list)
    for image in image_paths:
        k = (get_im_id(image), get_res(image)) if not use_crop_id else (get_im_id(image), get_res(image), get_crop_id(image))
        images[k].append(image)

    for label in label_paths:
        k = (get_im_id(label), get_res(label)) if not use_crop_id else (get_im_id(label), get_res(label), get_crop_id(label))
        labels[k].append(label)

    x, y = [], []

    for k in labels.keys():
        # print('Image found:', k)
        x.append(images[k])
        y.append(labels[k])

    return U_CPSequence(x, y, feature_file, batch_size=batch_size, norm_med=norm_med, seed=seed, used_wells=wells)

'''
class PredictImageCallback(Callback):
    def __init__(self, input_list, out_folder, bounds=None, frequency=10, log_filename=None, input_size=(512, 512, 1)):

        super().__init__()
        self.input_list = input_list
        self.out_folder = out_folder
        self.frequency = frequency
        self.bounds = bounds

    def on_epoch_end(self, epoch, _=None):
        import gc
        gc.collect()
        if epoch % self.frequency != 0:
            return
        out_paths = [os.path.join(self.out_folder, "%s_%.2d.tif" % (self.input_list[i].split(".")[0], epoch))
                     for i in range(len(self.filename_list))]
        x = np.zeros([len()])
        self.model.predict()
'''

# params
cp_input_shape = (config.sample_crop[0], config.sample_crop[1], 3)
val_ws = ['D04']
train_ws = list(filter(lambda w: w not in val_ws, config.wells))

limits = dataset.load_limits()
low, high = limits[config.magnifications[0]]["low"], limits[config.magnifications[0]]["high"]

# training CP feature predictor
'''
train_sequence = get_cp_dataset(config.cropped_data_dir, config.feature_file_path, True, config.seed, use_crop_id=True, wells=train_ws, batch_size=20)
val_sequence = get_cp_dataset(config.cropped_data_dir, config.feature_file_path, train_sequence.norm_med, config.seed, use_crop_id=True, wells=val_ws, batch_size=20)
cp_input = Input(shape=cp_input_shape)
cp_net = CP(cp_input_shape, config.n_features)(cp_input)
cp_net = Model(cp_input, cp_net)
# cp_net.summary(line_length=120)
cp_net.compile("adam", "mse", "mae")
cb = ModelCheckpoint(config.checkpoint_path)
if not os.path.exists(os.path.dirname(config.checkpoint_path)):
    os.mkdir(os.path.dirname(config.checkpoint_path))
# cp_net.fit(train_sequence, epochs=20, validation_data=val_sequence, callbacks=[cb])
cp_net.save_weights(config.u_cp_weights_path)
'''

# training U-net
'''
train_sequence = dataset.get_dataset(config.data_dir, train_=True, sample_per_image=config.train_samples_per_image,
                                     random_subsample_input=True, seed=config.seed, filter_fun=lambda im: dataset.info(im)[1] in train_ws)
val_sequence = dataset.get_dataset(config.data_dir, train_=False, sample_per_image=config.val_samples_per_image,
                                     random_subsample_input=True, seed=config.seed, filter_fun=lambda im: dataset.info(im)[1] in val_ws, resetseed=True)

u_input = Input((512, 512, 7))
u_net = Unet(u_input, n_depth=3, n_filter_base=64)
u_net = Model(u_input, u_net)
u_net.compile("adam", "mse") 
u_net.summary(line_length=120)
u_net.fit(train_sequence, epochs=1, validation_data=val_sequence)
'''

train_sequence = get_u_cp_dataset(config.cropped_data_dir, config.feature_file_path, True, config.seed, use_crop_id=True, wells=train_ws)
val_sequence = get_u_cp_dataset(config.cropped_data_dir, config.feature_file_path, train_sequence.norm_med, config.seed, use_crop_id=True, wells=val_ws)
cb = ModelCheckpoint(config.checkpoint_path[:-3] + "_ucp.h5")

model = U_CP(config.net_input_shape, config.n_features, norm_bounds=(low, high), n_depth=3)
model.compile("adam", "mse", "mae")
model.load_weights(config.cp_weights_path, True)
model.load_weights(config.u_weights_path, True)
model.summary(line_length=120)
model.fit(train_sequence, epochs=8, validation_data=val_sequence, callbacks=[cb])
