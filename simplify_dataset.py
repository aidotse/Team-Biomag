import os
from glob import glob
from collections import defaultdict
import numpy as numpy
import matplotlib.pyplot as plt
import imageio
import numpy as np
import pprint
import json

pp = pprint.PrettyPrinter(indent=4)

if __name__ == '__main__':
    def get_med_ch(low, high):
        low_med = {}
        high_med = {}
        for ch in range(4):
            low[ch].sort()
            high[ch].sort()
            low_med[ch] = low[ch][len(low[ch])//2]
            high_med[ch] = high[ch][len(high[ch])//2]

        return low_med, high_med
    
    def equalize(ch, lo, hi):
        ch[ch>hi] = hi
        ch[ch<lo] = lo
    
        ch = ch - lo
        ch = ch / (hi-lo)
        
        ch = ch*255
        ch = ch.astype(np.uint8)
        return ch


    data_folder = '/mnt/Data2/etasnadi/adipocyte-data/data'
    out_path = 'out'

    """
    Example image name:
    
    AssayPlate_Greiner_#655090_B04_T0001F004L01A03Z01C03.tif
    """

    # Intensity limit tolerance
    tol = 100

    ims = defaultdict(list)

    magnifications = ['40x']
    for magnification in magnifications:
        os.makedirs('%s/%s' % (out_path, magnification), exist_ok=True)
        data_path = os.path.join(data_folder, magnification)
        for im in os.listdir(data_path):
            k = im[:len('AssayPlate_Greiner_#655090_D04_T0001F006')]
            ims[k].append(im)

        for k in ims.keys():
            ims[k].sort()
        
        # Intensity minimums and maximums for each channel for each image
        low = defaultdict(list)
        high = defaultdict(list)

        print('Computing histo limit stats on the mag level: %s' % magnification)
        for idx, k in enumerate(ims.keys()):
            merged = []
            for ch_id, im in enumerate(ims[k][:3]):
                ch = imageio.imread(os.path.join(data_folder, magnification, im))
                
                sort_ch = np.sort(np.reshape(ch, (-1,)))
                n = len(sort_ch)
                pad = round(n/tol)
                a = sort_ch[pad]
                b = sort_ch[-pad]

                low[ch_id].append(a)
                high[ch_id].append(b)
            
            for b in ims[k][3:]:
                b_im = imageio.imread(os.path.join(data_folder, magnification, b))

                sort_ch = np.sort(np.reshape(b_im, (-1,)))
                n = len(sort_ch)
                pad = round(n/tol)
                a = sort_ch[pad]
                b = sort_ch[-pad]

                ch_id = 3
                low[ch_id].append(a)
                high[ch_id].append(b)

        low_med, high_med = get_med_ch(low, high)
        print('Computed histo stretch limit (med) for each channel:', low_med, high_med)

        limits = {'low': low_med, 'high': high_med}

        with open(os.path.join(out_path, 'limits-%s.json' % magnification), 'w') as outfile:
            outfile.write(str(limits))

        print('Exporting dataset')
        for idx, k in enumerate(ims.keys()):
            print('Fluo channels:')
            merged = []
            for ch_id, im in enumerate(ims[k][:3]):
                ch = imageio.imread(os.path.join(data_folder, magnification, im))

                lo = low_med[ch_id]
                hi = high_med[ch_id]

                ch = equalize(ch, lo, hi)
                
                merged.append(ch)

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
                imageio.imwrite('%s/%s/%s.png' % (out_path, magnification, b), b_im)