from sklearn.model_selection import train_test_split
import os
from os import listdir
from os.path import isfile, join
import shutil
import imageio
import numpy as np


ch1_folder = "/home/koosk/data/images/adipocyte/dataset/20x-tiles/C03"
ch2_folder = "/home/koosk/data/images/adipocyte/dataset/20x-tiles/C02"
ch3_folder = "/home/koosk/data/images/adipocyte/dataset/20x-tiles/C01"
output_folder = "/home/koosk/data/images/adipocyte/dataset/20x-tiles/RGB"

files = [f for f in listdir(ch1_folder) if isfile(join(ch1_folder, f))]
os.makedirs(output_folder, exist_ok=True)
for file in files:
    r = imageio.imread(join(ch1_folder, file)).astype(np.uint8)
    g_file = file.replace("L01A03", "L01A02")
    g = imageio.imread(join(ch2_folder, g_file)).astype(np.uint8)
    r_file = file.replace("L01A03", "L01A01")
    b = imageio.imread(join(ch3_folder, r_file)).astype(np.uint8)
    rgb = np.dstack((r, g, b))
    out_file = file.replace("L01A03", "L01A01")
    imageio.imwrite(join(output_folder, out_file), rgb, "png")
