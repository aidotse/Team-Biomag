import os
from collections import defaultdict

import config
import init
import stardist_blocks as sd
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.layers import Input, Dense, Conv2D
from tensorflow.keras import Model

from generators import CPSequence, U_CPSequence


def CP(n_features, input_shape):
    """ CP feature predictor network. Uses ResNet50 V2.

    :param n_features: number of CP features to predict
    :param input_shape: shape of the input image
    :return: cp_net function to create the network.

    Example:
    input = Input(input_shape)
    cp = CP(n_features, input_shape)(input)
    model = Model(input, cp)
    ...
    """
    def cp_net(input):
        resnet = ResNet50V2(False, None, input_tensor=input, input_shape=input_shape, pooling="avg")
        out = Dense(n_features)(resnet.output)
        return out

    return cp_net


def U_CP(input_shape, n_features, **kwargs):
    """ U-Net-CP network combined for brightfield to fluorescent generator with CP feature predictor

    :param input_shape:
    :param n_features:
    :param kwargs: keyword arguments for unet_block
    :return:
    """
    input = Input(shape=input_shape)
    unet = sd.unet_block(n_filter_base=64, **kwargs)(input)
    fluo_channels = Conv2D(3, (1, 1), name='fluo_channels', activation='sigmoid')(unet)
    cp_net = CP(n_features=n_features, input_shape=input_shape)(fluo_channels)

    return Model(input, [fluo_channels, cp_net])


def get_cp_dataset(im_dir, feature_file, train_, seed=None, filter_fun=None, use_crop_id=False):
    fluor_paths_ = os.listdir(im_dir)
    fluor_paths_ = filter(lambda p: p[-10:-7] in ["C%.2d" % i for i in range(1, 4)], fluor_paths_)
    fluor_paths_ = list(map(lambda p: os.path.join(im_dir, p), fluor_paths_))
    def filter_(paths, filter_fun_):
        result_ = []
        for path_ in paths:
            if filter_fun_(path_):
                result_.append(path_)

        return result_

    fluor_paths = fluor_paths_

    if filter_fun is not None:
        fluor_paths = filter_(fluor_paths_, filter_fun)

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
        x, feature_file, batch_size=1, train_=train_, seed=seed)

def get_u_cp_dataset(im_dir, feature_file, train_, seed=None, filter_fun=None, use_crop_id=False):
    path_list = os.listdir(im_dir)
    label_paths_ = filter(lambda p: p[-10:-7] in ["C%.2d" % i for i in range(1, 4)], path_list)
    label_paths_ = list(map(lambda p: os.path.join(im_dir, p), label_paths_))

    image_paths_ = filter(lambda p: p[-10:-7] == "C04", path_list)
    image_paths_ = list(map(lambda p: os.path.join(im_dir, p), image_paths_))

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

    return U_CPSequence(x, y, feature_file, batch_size=1,  train_=train_, seed=seed)


# params
cp_input_shape = (config.sample_crop[0], config.sample_crop[1], 3)

# data

lo_ws = ['B03']

# train_sequence = get_cp_dataset(config.data_dir, train_=True, sample_per_image=20, random_subsample_input=True, seed=config.seed, filter_fun=lambda im: info(im)[1] not in lo_ws)
# val_sequence = get_cp_dataset(config.data_dir, train_=False, sample_per_image=8, random_subsample_input=True, seed=config.seed, resetseed=True, filter_fun=lambda im: info(im)[1] in lo_ws)

# training CP feature predictor
train_sequence = get_cp_dataset(config.data_dir, config.feature_file_path, True, config.seed, use_crop_id=True)
cp_input = Input(shape=cp_input_shape)
cp_net = CP(n_features=config.n_features, input_shape=cp_input_shape)(cp_input)
cp_net = Model(cp_input, cp_net)
cp_net.summary(line_length=120)
cp_net.compile("adam", "mse", ["mse", "mae"])
cp_net.fit(train_sequence, epochs=20)
cp_net.save_weights(config.u_cp_weights_path)


train_sequence = get_u_cp_dataset(config.data_dir, config.feature_file_path, True, config.seed, use_crop_id=True)

model = U_CP(config.net_input_shape, config.n_features)
model.compile("adam", "mse", ["mse", "mae"])
model.load_weights(config.u_cp_weights_path, True)
model.summary(line_length=120)
model.fit(train_sequence, epochs=5)
