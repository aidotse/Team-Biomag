import os
import config

config.magnifications = ['20x']
config.val_samples_per_image = 1
config.val_subsample = False

config.train = False
config.readonly = False
config.save = True
config.data_dir = os.environ['AZ_DATA']
config.augment = True

config.output_dir = 'output/2020-1114_02-05-32'
config.init_weights = 'output/2020-1114_02-05-32/model-0162-0.00001119.h5'

config.predict_tile_size = (512, 512)
