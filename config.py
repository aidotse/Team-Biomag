import os
from datetime import datetime

seed = 42

magnifications = ['20x', '40x', '60x']

train_samples_per_image = 20
val_samples_per_image = 10
epochs = 1

# Will be set by the init script
data_dir = None

# Split the input images into two parts at y=(2154-512-1).
# We sample the patches from y<1641 to the train and y>=1641 to the val.
splity = 1641

# If not none, crop the top-left part to uniform.
target_size = (2154, 2554)

# The input will be sampled with this size.
sample_crop = (512, 512, 1)

# As the network is fully convolutional it can be independent of the input shape.
net_input_shape = (None, None, 7)

train = True

# The outputs will go to the output/$CURRENT-DATE
TRAIN_ID = datetime.now().strftime("%Y-%m%d_%H-%M-%S")
output_dir = os.path.join('output', TRAIN_ID)

init_weights = None

# augmentation on/off and probabilities
augment = True
rotate_p = 0.5
fliplr_p = 0.5
flipud_p = 0.5

readonly = False

# Wether to save the predicted images or not.
save = True

# The file to export the min and max values of images
limits_file = 'x-limits-%s.json'


initial_epoch = 1