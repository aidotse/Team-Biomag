import os
import config

config.magnifications = ['60x']
config.train_samples_per_image = 10
config.val_samples_per_image = 1
config.val_subsample = False

config.train = False
config.readonly = False
config.save = True
config.data_dir = os.environ['AZ_DATA']
config.augment = True

config.resize_crop = (1078, 1278)
config.upscale_result = (2156, 2556)

config.init_weights = 'output/60x/unet_60x_trained_500x.h5'
config.output_dir = 'output'
