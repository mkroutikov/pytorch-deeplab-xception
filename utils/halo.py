import cv2 as cv
import json
import numpy as np


def make_halo(mask1, mask2, kernel):
    '''Locates regions were mask1 and mask2 are close to each other

    mask1 - an image of type np.uint8 with 0 backgrouond and 1 foreground
    mask2 - an image of type np.uint8 with 0 backgrouond and 1 foreground

    images mask1 adn mask2 are non-intersecting (but may be close to each other)
    '''

    i1 = cv.dilate(mask1, kernel) > 0
    i2 = cv.dilate(mask2, kernel) > 0
    halo = np.logical_and(i1, i2)
    halo = np.logical_and(halo, mask1 == 0)
    halo = np.logical_and(halo, mask2 == 0)

    return halo.astype(np.uint8)


def halo(w, h, zones, kernel):
    '''
    Creates an image with mask indicating geometrically close zones

    Output is a numpy array of type np.uint32 with background 0 and foreground 1
    '''
    i3 = np.zeros((h,w), np.uint8)

    for index in range(len(zones)):
        i1 = np.zeros((h,w), np.uint8)
        i2 = np.zeros((h,w), np.uint8)
        x0, y0, x1, y1 = zones[index]['bbox']
        cv.rectangle(i1, (x0, y0), (x1, y1), color=1, thickness=-1)
        for idx,z in enumerate(zones):
            xx0, yy0, xx1, yy1 = z['bbox']
            if idx == index:
                continue
            cv.rectangle(i2, (xx0, yy0), (xx1, yy1), color=1, thickness=-1)

        i3 += make_halo(i1, i2, kernel)

    return (i3 > 0).astype(np.uint8)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from ilabs.curate import ic

    dataset = ic.get_dataset('ilabs.vision', 'scarlet200')

    count = 0
    for fname in dataset['train']:
        print(fname)
        if not fname.endswith('.json'):
            continue

        with open(fname) as f:
            meta = json.load(f)

        w, h = meta['size']

        kernel = np.ones((10,10),np.uint8)

        halo_ = halo(w, h, meta['zones'], kernel)

        image_name = fname[:-5] + '.png'
        mask_name = fname[:-5] + '-mask.png'

        image = cv.imread(image_name)

        mask = cv.imread(mask_name)
        mask = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
        mask = (mask < 128).astype(np.uint8)

        print(np.mean(mask), np.mean(halo_))

        plt.figure()
        plt.title('display')
        plt.subplot(131)
        plt.imshow(image)
        plt.subplot(132)
        plt.imshow(mask)
        plt.subplot(133)
        plt.imshow(halo_)
        count += 1
        if count > 5:
            break

    plt.show(block=True)