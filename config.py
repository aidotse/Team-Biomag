data_dir = None
# Resize input images to this size
target_size = (2154, 2554)

sample_crop = (256, 256, 1)
net_input_shape = (256, 256, 7)

save_checkpoint = 'model.h5'
#save_checkpoint = None
output_dir = 'output'

# augmentation on/off and probabilities
augment = False
rotate_p = 0.5
fliplr_p = 0.5
flipud_p = 0.5
