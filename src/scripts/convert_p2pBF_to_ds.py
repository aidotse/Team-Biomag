import os
from os import listdir
from os.path import isfile, join
from dataset import denormalize, load_limits

import imageio
import numpy as np

magnification = "20x"
input_folder = "/home/koosk/data/images/adipocyte/generated_images/20x-RGB-Z0X-results/prepared/512"
output_folder = "/home/koosk/data/images/adipocyte/generated_images/20x-RGB-Z0X-results/prepared/512/tiff"

files = [f for f in listdir(input_folder) if isfile(join(input_folder, f))]
os.makedirs(output_folder, exist_ok=True)
limits = load_limits()
low, high = limits[magnification]["low"], limits[magnification]["high"]

for file in files:
    img = imageio.imread(join(input_folder, file)).astype(np.float32)/255

    ch4 = denormalize(img, low[3], high[3]).astype(np.uint16)

    bname = file[:-4]
    ch4_out = join(output_folder, bname+".tif")
    imageio.imwrite(ch4_out, ch4, "tif")
