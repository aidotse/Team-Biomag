# BIOMAG Adipocyte challenge code

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

Create an `init.py` file in the project root, then add and modify this example to your needs:
```python
import config

config.data_dir = "/home/ervin/Dokumentumok/brc/Adipocyte/images_for_preview"
``` 

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



