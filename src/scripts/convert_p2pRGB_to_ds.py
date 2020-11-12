import os
from os import listdir
from os.path import isfile, join
from dataset import denormalize, load_limits

import imageio
import numpy as np

magnification = "20x"
input_folder = "/home/koosk/data/images/adipocyte/generated_images/fluor/20x/512"
output_folder = "/home/koosk/data/images/adipocyte/generated_images/fluor/20x/512/split"

ch1_postfix = "_A01Z01C01"
ch2_postfix = "_A02Z01C02"
ch3_postfix = "_A03Z01C03"

files = [f for f in listdir(input_folder) if isfile(join(input_folder, f))]
os.makedirs(output_folder, exist_ok=True)
limits = load_limits()
low, high = limits[magnification]["low"], limits[magnification]["high"]

for file in files:
    img = imageio.imread(join(input_folder, file)).astype(np.float32)/255

    ch1 = img[..., 2]
    ch2 = img[..., 1]
    ch3 = img[..., 0]

    ch1 = denormalize(ch1, low[0], high[0]).astype(np.uint16)
    ch2 = denormalize(ch2, low[1], high[1]).astype(np.uint16)
    ch3 = denormalize(ch3, low[2], high[2]).astype(np.uint16)

    bname = file[:-4]
    ch1_out = join(output_folder, bname+ch1_postfix+".tif")
    ch2_out = join(output_folder, bname+ch2_postfix+".tif")
    ch3_out = join(output_folder, bname+ch3_postfix+".tif")
    imageio.imwrite(ch1_out, ch1, "tif")
    imageio.imwrite(ch2_out, ch2, "tif")
    imageio.imwrite(ch3_out, ch3, "tif")


    # r = imageio.imread(join(ch1_folder, file)).astype(np.uint8)
    # g = imageio.imread(join(ch2_folder, file)).astype(np.uint8)
    # b = imageio.imread(join(ch3_folder, file)).astype(np.uint8)
    # rgb = np.dstack((r, g, b))
    # # out_file = file.replace("L01A03", "L01A01")
    # out_file = file
    # imageio.imwrite(join(output_folder, out_file), rgb, "png")