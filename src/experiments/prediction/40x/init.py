import os
import config

config.magnifications = ['40x']
config.val_samples_per_image = 1
config.val_subsample = False

config.train = False
config.readonly = False
config.save = True
config.data_dir = os.environ['AZ_DATA']
config.augment = True

config.output_dir = 'output'
config.init_weights = 'output/40x/unet_40x_trained_500x.h5'

config.predict_tile_size = (512, 512)
