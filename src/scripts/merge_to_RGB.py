import os
from os import listdir
from os.path import isfile, join

import imageio
import numpy as np

ch1_folder = "/home/koosk/data/images/adipocyte/dataset/60x-tiles-renamed/Z01C03"
ch2_folder = "/home/koosk/data/images/adipocyte/dataset/60x-tiles-renamed/Z01C02"
ch3_folder = "/home/koosk/data/images/adipocyte/dataset/60x-tiles-renamed/Z01C01"
output_folder = "/home/koosk/data/images/adipocyte/dataset/60x-tiles-renamed/RGB"

files = [f for f in listdir(ch1_folder) if isfile(join(ch1_folder, f))]
os.makedirs(output_folder, exist_ok=True)
for file in files:
    r = imageio.imread(join(ch1_folder, file)).astype(np.uint8)
    g = imageio.imread(join(ch2_folder, file)).astype(np.uint8)
    b = imageio.imread(join(ch3_folder, file)).astype(np.uint8)
    rgb = np.dstack((r, g, b))
    # out_file = file.replace("L01A03", "L01A01")
    out_file = file
    imageio.imwrite(join(output_folder, out_file), rgb, "png")
