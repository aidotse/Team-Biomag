import config
import init
from dataset import denormalize, load_limits
import os
import cv2.cv2 as cv2
import numpy as np

# you should set magnifications, output_dir (containing output from the model) in config
limits = load_limits()

for magnification in config.magnifications:
    in_folder = os.path.join(config.output_dir, "visual", magnification)
    out_folder = os.path.join(in_folder, "orig_format")
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    low, high = limits[magnification]["low"], limits[magnification]["high"]
    low, high = np.expand_dims(low[1:], (0, 1)), np.expand_dims(high[1:], (0, 1))
    files = os.listdir(in_folder)
    files = filter(lambda p: p.endswith("out.tif"), files)
    for im_path in files:
        img = cv2.imread(os.path.join(in_folder, im_path), -1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = denormalize(img, low, high).astype(np.uint16)
        new_path = im_path[:-8] + "L01A%.2dZ01C%.2d.tif"
        for i in range(3):
            cv2.imwrite(os.path.join(out_folder, new_path % (i+1, i+1)), img[..., i])
