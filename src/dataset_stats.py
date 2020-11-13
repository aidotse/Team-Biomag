import os
from glob import glob
from collections import defaultdict
import numpy as numpy
import matplotlib.pyplot as plt

import imageio
import numpy as np
import pprint
from statistics import median_low, median_high, mean

import config
import init
import misc

pp = pprint.PrettyPrinter(indent=4)

def equalize(ch, lo, hi):
    ch[ch>hi] = hi
    ch[ch<lo] = lo

    ch = ch - lo
    ch = ch / (hi-lo)
    
    ch = ch*255
    ch = ch.astype(np.uint8)
    return ch

def simplify(only_measure=True):
    data_folder = config.data_dir
    out_path = '.'

    """
    Example image name:
    
    AssayPlate_Greiner_#655090_B04_T0001F004L01A03Z01C03.tif
    """

    # Intensity limit tolerance
    tol = 100

    out = {}

    magnifications = ['20x', '40x', '60x']
    for magnification in magnifications:
        ims = defaultdict(list)
        data_path = os.path.join(data_folder, magnification)
        for im in os.listdir(data_path):
            k = im[:len('AssayPlate_Greiner_#655090_D04_T0001F006')]
            ims[k].append(im)

        for k in ims.keys():
            ims[k].sort()

        #import pprint
        #print('Images at mag:', magnification)
        #pp = pprint.PrettyPrinter(indent=4)
        #pp.pprint(ims)

        for k in ims.keys():
            print(k, len(ims[k]))

        # Intensity minimums and maximums for each channel for each image
        lows = defaultdict(list)
        highs = defaultdict(list)
        mins = defaultdict(list)
        maxs = defaultdict(list)

        means = defaultdict(list)
        stds = defaultdict(list)

        # Process each FOV on the particular magnification level

        print('Computing histo limit stats on the mag level: %s' % magnification)
        for idx, im_key in enumerate(ims.keys()):
            merged = []
            # Process fluos

            print('Fluo keys:', ims[im_key][:3])

            fluo_channels = []

            for ch_id, fluo_filename in enumerate(ims[im_key][:3]):
                n, _a, _b, pad = 0, 0, 0, 0
                sort_ch = None
                fluo_path = None
                fluo_im = None

                ####

                fluo_path = os.path.join(data_folder, magnification, fluo_filename)
                fluo_im = imageio.imread(fluo_path)

                fluo_channels.append(fluo_im)

                sort_ch = np.sort(np.reshape(fluo_im, (-1,)))
                n = len(sort_ch)
                pad = round(n/tol)

                _a, _b = sort_ch[pad], sort_ch[-pad]

                lows[ch_id].append(_a)
                highs[ch_id].append(_b)
                mins[ch_id].append(sort_ch[0])
                maxs[ch_id].append(sort_ch[-1])

                means[ch_id].append(np.mean(fluo_im))
                stds[ch_id].append(np.std(fluo_im))
                print('im (fluo): %s, %s low: %s high: %s min: %s max: %s' % (fluo_path, magnification, _a, _b, sort_ch[0], sort_ch[-1]))

            #fluo = np.stack(fluo_channels)
            #means['fluo'].append(np.mean(fluo))
            #stds['fluo'].append(np.std(fluo))

            # Process brightfields
            bright_slices = []
            bright_ch_id = 3
            for bright_filename in ims[im_key][3:]:
                n, _a, _b, pad = 0, 0, 0, 0
                sort_ch = None
                bright_path = None
                bright_im = None

                ####

                bright_path = os.path.join(data_folder, magnification, bright_filename)
                bright_im = imageio.imread(bright_path)
                bright_slices.append(bright_im)

                sort_ch = np.sort(np.reshape(bright_im, (-1,)))

                n = len(sort_ch)
                pad = round(n/tol)

                _a, _b = sort_ch[pad], sort_ch[-pad]

                lows[bright_ch_id].append(_a)
                highs[bright_ch_id].append(_b)
                mins[bright_ch_id].append(sort_ch[0])
                maxs[bright_ch_id].append(sort_ch[-1])
                print('im (bright): %s mag: %s low: %s high: %s min: %s max: %s' % (bright_path, magnification, _a, _b, sort_ch[0], sort_ch[-1]))

            bright = np.stack(bright_slices)
            means[bright_ch_id].append(np.mean(bright))
            stds[bright_ch_id].append(np.std(bright))

            '''
            if idx == 0:
                break
            '''

        # End of the loop through the FOVs

        channels = list(range(4))

        ch_ids = None

        for ch_ids in channels:
            lows[ch_ids].sort()
            highs[ch_ids].sort()

            print('LOWS', ch_ids, lows[ch_ids])
            print('HIGHS', ch_ids, highs[ch_ids])

        ch_id = None
        limits = None
        minmax = None
        limits = {
            'low': [int(median_low(lows[ch_id])) for ch_id in channels], 
            'high': [int(median_high(highs[ch_id])) for ch_id in channels]}

        minmax = {
            'min': [int(min(mins[ch_id])) for ch_id in channels], 
            'max': [int(max(maxs[ch_id])) for ch_id in channels]}


        print('stds, means:', stds, means)

        stat = {
            'std': [mean(stds[ch_id]) for ch_id in channels],
            'mean': [mean(means[ch_id]) for ch_id in channels]
        }

        print('limits, minmax', limits, minmax)
        print('std mean', stat)

        misc.put_json(os.path.join(out_path, config.limits_file) % magnification, limits)
        misc.put_json(os.path.join(out_path, config.stats_file) % magnification, stat)

        if only_measure == False:
            print('Exporting dataset')
            for idx, k in enumerate(ims.keys()):
                print('Fluo channels:')
                merged = []
                for ch_id, im in enumerate(ims[k][:3]):
                    fluo_im = imageio.imread(os.path.join(data_folder, magnification, im))

                    lo = low_med[ch_id]
                    hi = high_med[ch_id]

                    fluo_im = equalize(fluo_im, lo, hi)
                    
                    merged.append(fluo_im)

                merged = np.array(merged).astype(np.uint8)
                merged = np.transpose(merged, (1, 2, 0))

                imageio.imwrite('%s/%s/%s.png' % (out_path, magnification, os.path.splitext(ims[k][0])[0]), merged)
                
                pp.pprint(ims[k][:3])
                print('Bright channels:')
                pp.pprint(ims[k][3:])

                ch_id = 3
                for b in ims[k][3:]:
                    b_im = imageio.imread(os.path.join(data_folder, magnification, b))
                    b_im = equalize(b_im, low_med[ch_id], high_med[ch_id])
                    imageio.imwrite('%s/%s/%s.png' % (out_path, magnification, b[:-4]), b_im)


if __name__ == '__main__':
    simplify()