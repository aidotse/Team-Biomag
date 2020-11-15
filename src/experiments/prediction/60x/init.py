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

config.resize_crop = (1077, 1277)
config.upscale_result = (2054, 2554)

config.init_weights = 'output/2020-1114_01-47-25/model-0355-0.00000642.h5'
config.output_dir = 'output/2020-1114_01-47-25'
