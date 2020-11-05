import os
import json

def magnification_level(path):
    return os.path.basename(os.path.dirname(path))

def get_json(fname):
    with open(fname, 'r') as limitsfile:
        return json.load(limitsfile)


def put_json(fname, data):
    with open(fname, 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)