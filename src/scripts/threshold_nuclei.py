import os
from os import listdir
from os.path import isfile, join
import glob

import imageio
import numpy as np
from skimage import img_as_ubyte
from skimage import io
from skimage.filters import threshold_otsu

# input_folder = "/home/koosk/data/images/adipocyte/dataset/20x"
# output_folder = "/home/koosk/data/images/adipocyte/dataset/20x-nuclei_thresholded"
# input_folder = "/home/koosk/data/images/adipocyte/dataset/40x"
# output_folder = "/home/koosk/data/images/adipocyte/dataset/40x-nuclei_thresholded"
input_folder = "/home/koosk/data/images/adipocyte/dataset/60x"
output_folder = "/home/koosk/data/images/adipocyte/dataset/60x-nuclei_thresholded"
# input_folder = "/home/koosk/data/images/adipocyte/nuclei_CP_mask-60x-indexed"
# output_folder = "/home/koosk/data/images/adipocyte/nuclei_CP_mask-60x"

if __name__ == '__main__':
    """
    Threshold nuclei images in folder and save them.
    """
    os.makedirs(output_folder, exist_ok=True)
    # files = listdir(input_folder)
    files = glob.glob(f"{input_folder}/*C01*")
    for file in files:
        input_file = join(input_folder, file)
        if not isfile(input_file):
            continue
        # img = imageio.imread(input_file).astype(np.float32)
        image = imageio.imread(input_file).astype(np.float32)
        image = image / np.max(image) * 255
        image = image.astype(np.uint8)
        # print(np.max(image))
        # print(np.shape(image))
        thresh = threshold_otsu(image)
        # thresh = 0
        binary = image > thresh
        # # binary = np.asarray(binary)
        # print(type(image))
        # print(type(binary))
        # print(np.shape(binary))
        # imageio.imwrite("/home/koosk/data/images/adipocyte/tmp/thresholded.png", image, "png")
        im_bname = os.path.basename(input_file)
        im_name = im_bname[:-4]
        im_ext = im_bname[-3:]
        out_fpath = join(output_folder, im_name + ".png")
        io.imsave(out_fpath, img_as_ubyte(binary))

