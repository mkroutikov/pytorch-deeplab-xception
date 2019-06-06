import numpy as np

def blending_masks(offsets, size=513):
    numx = offsets.shape[0]
    numy = offsets.shape[1]
    assert offsets.shape[2] == 2, offsets.shape # expect X and Y there

    # lets check that tiles do not overlap too much
    # or the algorithm for blending would be wrong
    for x in range(numx):
        for y in range(numy-1):
            assert size//2 < offsets[x,y+1,1]-offsets[x,y,1] <= size, (x, y, offsets[x,y+1,1], offsets[x,y,1], size, offsets)
    for y in range(numy):
        for x in range(numx-1):
            assert size//2 < offsets[x+1,y,0]-offsets[x,y,0] <= size, (x, y, offsets[x+1,y,0], offsets[x,y,0], size, offsets)

    masks = np.ones( (numx, numy, size, size), dtype=np.float32)

    for y in range(numy):
        for x in range(numx):
            off = offsets[x,y,0]
            prev_stripe = offsets[x-1,y,0]+size-off if x > 0 else 0
            next_stripe = off+size-offsets[x+1,y,0] if x < numx-1 else 0

            for i in range(prev_stripe):
                weight = (i+1) / (prev_stripe+1)
                masks[x,y,i,:] *= weight

            for i in range(next_stripe):
                weight = 1. - (i+1) / (next_stripe+1)
                masks[x,y,i+size-next_stripe,:] *= weight

    for x in range(numx):
        for y in range(numy):
            off = offsets[x,y,1]
            prev_stripe = offsets[x,y-1,1]+size-off if y > 0 else 0
            next_stripe = off+size-offsets[x,y+1,1] if y < numy-1 else 0

            for i in range(prev_stripe):
                weight = (i+1) / (prev_stripe+1)
                masks[x,y,:,i] *= weight

            for i in range(next_stripe):
                weight = 1. - (i+1) / (next_stripe+1)
                masks[x,y,:,i+size-next_stripe] *= weight

    return masks


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    offsets = np.array([
        [(0, 0), (0, 75)],
        [(60, 0), (60, 75)],
        [(120, 0), (120,75)]
    ], dtype=np.int32)

    size = 100

    masks = blending_masks(offsets, size=size)
    plt.figure()
    plt.title('masks')

    plt.subplot(231)
    plt.imshow(masks[0,0].transpose(1,0))
    plt.subplot(232)
    plt.imshow(masks[1,0].transpose(1,0))
    plt.subplot(233)
    plt.imshow(masks[2,0].transpose(1,0))

    plt.subplot(234)
    plt.imshow(masks[0,1].transpose(1,0))
    plt.subplot(235)
    plt.imshow(masks[1,1].transpose(1,0))
    plt.subplot(236)
    plt.imshow(masks[2,1].transpose(1,0))

    plt.figure()

    width = np.max(offsets[:,:,0]) + size
    height = np.max(offsets[:,:,1]) + size
    pic = np.zeros((width, height), dtype=np.float32)
    for x in range(offsets.shape[0]):
        for y in range(offsets.shape[1]):
            offx, offy = offsets[x,y]
            pic[offx:offx+size,offy:offy+size] += masks[x,y]

    print(np.min(pic), np.max(pic))

    plt.title('cumulative')
    plt.imshow(pic.transpose(1,0))

    plt.show(block=True)