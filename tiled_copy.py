from math import ceil

verbose = False


def get_tiles(x, k, n_tiles=None, overlap_ratio=.2):
    """ Constructs the tiling parameters in 1D.

    Parameters:
    x (list): the input array size
    k (int): the tile sizes intended to crop from the input
    n_tiles (int)=none: the number of requested tiles to use or skipped
    overlap_ratio (float)=.2: if the n_tiles is not specified, it will be determined
        based on the requested overlap ratio between the subsequent input tiles.

    Returns:
    src_tiles (list((int, int))): what to crop from the source
    src_tile_crops (list(int, int)): crop this from the cropped tile
    taret_tiles (list(int, int)): put the cropped source crop here

    Usage:
    for t in range(len(target_tiles)):
        target[slice(*target_tiles[t])] = source[slice(*src_tiles[t])][slice(*src_tile_crops[t])]
    
    Example:
    >>> get_tiles(100, 25)
    ([(0, 25), (15, 40), (30, 55), (45, 70), (60, 85), (75, 100)], 
     [(0, -5), (5, -5), (5, -5), (5, -5), (5, -5), (5, None)], 
     [(0, 20), (20, 35), (35, 50), (50, 65), (65, 80), (80, 100)])
    
    """

    if k > x or k < 1:
        raise ValueError("The tile size should be positive and less than or equal to the input array size.")

    if n_tiles is not None and x > n_tiles*k:
        raise ValueError("The sum size of the tiles is less than the size of the input array.")

    # Compute the number of tiles automatically.
    # If c is the margin we want to drop, then: x = 2*(k-c)+(n-2)*(k-c). Solving for n gives n=x/(k-2*x)-1.
    if n_tiles is None:
        c = k*overlap_ratio
        n_tiles = ceil(x/(k-2*c))-1

    # First, determine the centers for each input tile considering the input array coordinates.
    # The first and the last center is in k/2 distance from the sides.
    mid = x-(k/2)*2
    step = mid/(n_tiles-2+1)
    pad = k/2

    tile_centers = [pad] + [pad + (i+1)*step for i in range(n_tiles-2)] + [x-pad]
    
    # Next, determine each tile centered at the previously computed centers.
    src_tiles = [(round(t-k/2), round(t+(k/2))) for t in tile_centers]
    
    # Now, determine the centers (middle points) of the overlaps of the neigbouring tiles.
    overlap_midpts = []
    for i in range(len(src_tiles)-1):
            next_tile_start, current_tile_end = src_tiles[i+1][0], src_tiles[i][1]
            midpt = round((next_tile_start + current_tile_end)/2)
            overlap_midpts.append(midpt)

    # The target tiles are computed by considering the adjacent midpoints as their endpoints.
    target_tiles = []
    for i in range(len(overlap_midpts)):
        prev_midpt = 0 if i == 0 else overlap_midpts[i-1]
        act_midpt = overlap_midpts[i]
        target_tiles.append((prev_midpt, act_midpt))

    target_tiles += [(overlap_midpts[-1], x)]

    # Determine the croppings from the input to align with the target tiles.
    src_tile_crops = []
    for i in range(len(target_tiles)):
        crop_s = target_tiles[i][0]-src_tiles[i][0]
        crop_e = target_tiles[i][1]-src_tiles[i][1]
        src_tile_crops.append((crop_s, None if crop_e == 0 else crop_e))

    if verbose:
        print('x: %d, #tiles: %s, middle: %d, step: %f' % (x, n_tiles, mid, step))
        print('tile centers:    ', tile_centers)
        print('tiles:           ', src_tiles)
        print('overlap midpts:  ', overlap_midpts)
        print('targets:         ', target_tiles)
        print('src tile crops:  ', src_tile_crops)
    
    return src_tiles, src_tile_crops, target_tiles


def test():
    """ Creates a test array a, and a target b with the same size but filled with zeros.
    The task is to copy the contents of a into b using overlapped tiling.
    To make sure the algorithm works it is a good idea to use prime length arrays, 
    request prime number of tiles and prime length tiles to use.

    """
    # length
    x = 587
    
    # tile size
    k = 97

    # number of tiles
    n_tiles = None

    a = list(range(x))
    b = [0] * len(a)
    print('length of a (source): %d, length of b (target): %d, a==b: %s (before copying)' % (len(a), len(b), a==b))

    src_tiles, src_tile_crops, target_tiles = get_tiles(x, k, n_tiles)

    for i in range(len(target_tiles)):
        t = slice(*target_tiles[i])
        s = slice(*src_tiles[i])
        cr = slice(*src_tile_crops[i])
        if verbose:
            print('Target: %s\t\t\tsource: %s\t\t\tcrop from source: %s' % (t, s, cr))
        b[t] = a[s][cr]

    print('length of a (source): %d, length of b (target): %d, a==b: %s (after copying)' % (len(a), len(b), a==b))


if __name__ == '__main__':
    test()