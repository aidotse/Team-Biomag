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



