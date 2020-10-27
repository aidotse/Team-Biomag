import os
from datetime import datetime

# Will be set by the init script
data_dir = None

# Split the input images into two parts at y=1600.
# We sample the patches from y<1600 to the train and y>1600 to the val.
splity = 1600

# Crop the top-left part of this sizem from the input to have equal size inputs.
target_size = (2154, 2554)

# The input will be sampled with this size.
sample_crop = (256, 256, 1)
net_input_shape = (256, 256, 7)

train = False

# The outputs will go to the output/$CURRENT-DATE
TRAIN_ID = datetime.now().strftime("%Y-%m%d_%H-%M-%S")
output_dir = os.path.join('output', TRAIN_ID)

init_weights = None

# augmentation on/off and probabilities
augment = False
rotate_p = 0.5
fliplr_p = 0.5
flipud_p = 0.5

readonly = True