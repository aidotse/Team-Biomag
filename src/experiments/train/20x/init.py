import os
import config

config.magnifications = ['20x']

config.epochs = 600
config.train_samples_per_image = 10
config.val_samples_per_image = 1

config.train = True
config.readonly = False
config.save = True
config.data_dir = os.environ['AZ_DATA']
config.augment = True

# Resume a previous train
config.output_dir = 'output/2020-1114_02-05-32'
config.init_weights = 'output/2020-1114_02-05-32/model-0162-0.00001119.h5'

config.initial_epoch = 162
config.predict_tile_size = (512, 512)

