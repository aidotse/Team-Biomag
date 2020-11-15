# Biomag

## How to run inference, quick guide
Inference is ran using src/predict_all.sh script. The script includes input file path
AZ_DATA
which should be changed to the test data input directory. The script assumes that the input directory includes separately 20x, 40x and 60x subfolders for different magnifications.

Config files for each magnification are located at
src/experiments/prediction/(20x|40x|60x)/init.py

The output path and the model path should be modified into each of those three files in config variables
config.output_dir
config.init_weights

After the modifications, just run
./predict_all.sh

## Setting up
Python 3.7 is recommended.

Add virtual environment in your project root:

```shell script
$ python3 -m venv ./.venv/
```

Activate it:
```shell script
$ source .venv/bin/activate
```

Install packages:
```shell script
$ pip install -r requirements.txt
```

Create an `init.py`  file to init env specific stuff in the project root, then add and modify this example to your needs:
```python

import config

# Which magnification subsets to use
config.magnifications = ['20x', '40x', '60x']

config.epochs = 100
config.train_samples_per_image = 10
config.val_samples_per_image = 1

config.train = False

# Save anything to the disk?
config.readonly = False

# Save the predicted images into the out dir?
config.save = True

config.data_dir = 'data'
config.augment = True

# To load the weights into the model
config.init_weights = None

# If resuming a train then specify the resumed out dir and the start epoch
#config.output_dir = None
#config.initial_epoch = 2

``` 

During training, a subdirectory will be created with the current timestamp in the ```out_dir```. 
The weights will be saved here.
If the ```save``` set to true, then if the ```test``` function saves the predicted images grabbed from the val sequence.
The raw results in the required format will go into the folder ```out/$TIMESTAMP/results``` (currently float 32 each channel).
The results for visualization will go to the ```out/$TIMESTAMP/visual``` subdir along with the jsons containing the MSE for each image.

If you use the GPU for a monitor as well, then you might want to add this snippet (or set environment
 variable `TF_FORCE_GPU_ALLOW_GROWTH=true`):
```python
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf_config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

## Development
Activate your venv:
```shell script
$ source .venv/bin/activate
```

Run training with:
```
```

Make sure to keep `.gitignore` updated if you use an editor that adds new files, or you add 
some data files to the path. Then you can commit and push your changes:
```
$ git commit -a -m "message"
$ git push
```

## Docker
In the `Docker` folder, build the image using `sh build.sh`.
Then the container can be started using `sh run.sh`. 

