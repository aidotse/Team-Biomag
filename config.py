import os
from datetime import datetime

seed = 42

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

# Whether to run the code on the local computer, or use the dgx8 format...
local_run = True