import numpy as np

def blending_masks(offsets, size=513):

    # expect either horizontally-stacked crops, or vertically-striped crops
    if all(x[1] == 0 for x in offsets):
        # horizontally stacked crops: convert to vertical
        offsets = [(y, x) for x,y in offsets]

        return np.transpose(blending_masks(offsets, size=size), (0, 2, 1))

    if not all(x[0] == 0 for x in offsets):
        raise ValueError('I can only generate blending masks for vertically or horizontally stacked crops')

    offsets = [x[1] for x in offsets]  # leave only x-offsets
    offsets.append(offsets[-1]+size)

    masks = np.ones( (len(offsets) - 1, size, size), dtype=np.float32)

    prev = 0
    for k,(po,no) in enumerate(zip(offsets,offsets[1:])):
        left_stripe = prev-po
        right_stripe = po+size-no

        for i in range(left_stripe):
            weight = (i+1) / (left_stripe+1)
            masks[k,i,:] = weight

        for i in range(right_stripe):
            weight = 1. - (i+1) / (right_stripe+1)
            masks[k,i+size-right_stripe,:] = weight

        prev = po + size

    return masks
