import os
import imageio
import numpy as np
from os import listdir
from os.path import isfile, join

# input_folder = "/home/koosk/data/images/adipocyte/tmp/"
# output_folder = "/home/koosk/data/images/adipocyte/tmp/tiles"
# input_folder = "/home/koosk/data/images/adipocyte/dataset/20x"
# output_folder = "/home/koosk/data/images/adipocyte/dataset/20x-tiles"
# input_folder = "/home/koosk/data/images/adipocyte/dataset/40x"
# output_folder = "/home/koosk/data/images/adipocyte/dataset/40x-tiles"
# input_folder = "/home/koosk/data/images/adipocyte/dataset/60x"
# output_folder = "/home/koosk/data/images/adipocyte/dataset/60x-tiles"
# input_folder = "/home/koosk/data/images/adipocyte/dataset/20x-nuclei_thresholded"
# output_folder = "/home/koosk/data/images/adipocyte/dataset/20x-nuclei_thresholded-tiled"
# input_folder = "/home/koosk/data/images/adipocyte/dataset/40x-nuclei_thresholded"
# output_folder = "/home/koosk/data/images/adipocyte/dataset/40x-nuclei_thresholded-tiled"
input_folder = "/home/koosk/data/images/adipocyte/dataset/60x-nuclei_thresholded"
output_folder = "/home/koosk/data/images/adipocyte/dataset/60x-nuclei_thresholded-tiled"
# input_folder = "/home/koosk/data/images/adipocyte/nuclei_mask-20x"
# input_folder = "/home/koosk/data/images/adipocyte/nuclei_mask-20x-tiled"
tile_size = 512


if __name__ == '__main__':
    os.makedirs(output_folder, exist_ok=True)
    files = listdir(input_folder)
    for file in files:
        input_file = join(input_folder, file)
        if not isfile(input_file):
            continue
        img = imageio.imread(input_file).astype(np.float32)
        img = img - np.min(img)
        img = img / np.max(img) * 255
        img = img.astype('uint8')
        im_bname = os.path.basename(input_file)
        imshape = (np.shape(img))
        im_name = im_bname[:-4]
        im_ext = im_bname[-3:]
        tile_counter = 0
        for i in range(0, imshape[0] - tile_size + 1, tile_size):
            for j in range(0, imshape[1] - tile_size + 1, tile_size):
                tile_counter += 1
                img_tile = img[i:i+tile_size, j:j+tile_size]
                tile_fname = join(output_folder, im_name + "-tile-%03d" % tile_counter + ".png")
                imageio.imwrite(tile_fname, img_tile, "png")

