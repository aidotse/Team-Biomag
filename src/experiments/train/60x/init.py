import config

config.magnifications = ['60x']

config.epochs = 600
config.train_samples_per_image = 10
config.val_samples_per_image = 1

config.train = True
config.readonly = False
config.save = True

config.data_dir = os.environ['AZ_DATA']
config.augment = True

# Resize the crop right before giving it to the network
config.resize_crop = (512, 512)

# Random sample this size of crops from the raw image.
config.sample_crop = (1024, 1024, 1)

# Resume a previous train
config.init_weights = 'output/2020-1114_01-47-25/model-0355-0.00000642.h5'
config.output_dir = 'output/2020-1114_01-47-25'
config.initial_epoch = 355

config.predict_tile_size = (512, 512)
