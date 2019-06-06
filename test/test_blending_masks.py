import numpy as np
from dataloaders.datasets.blending_masks import blending_masks


def test1():

    '''
    x x x
    x x x
    xyxyxy
    z z z
    z z z
    '''
    masks = blending_masks(
        np.array([[ (0,0), (0,2) ]], dtype=np.int32), size=3)

    np.testing.assert_almost_equal(
        masks,
        np.array([[
            [
                [1., 1., 0.5],
                [1., 1., 0.5],
                [1., 1., 0.5],
            ],
            [
                [0.5, 1., 1.],
                [0.5, 1., 1.],
                [0.5, 1., 1.],
            ]
        ]])
    )

def test2():

    '''
    x x x
    x x x
    xyxyxy
    z z z
    z z z
    '''
    masks = blending_masks(np.array([[(0,0)], [(2,0)]], dtype=np.int32), size=3)

    np.testing.assert_almost_equal(
        masks,
        np.array([
            [[
                [1., 1., 1.],
                [1., 1., 1.],
                [0.5, 0.5, 0.5],
            ]],
            [[
                [0.5, 0.5, 0.5],
                [1., 1., 1.],
                [1., 1., 1.],
            ]]
        ])
    )

def test3():

    '''
    x x x
    x x x
    xyxyxy
    z z z
    z z z
    '''
    masks = blending_masks(np.array(
        [ [(0,0)], [(2,0)], [(4,0)] ], dtype=np.int32), size=3)

    np.testing.assert_almost_equal(
        masks,
        np.array([[
            [
                [1., 1., 0.5],
                [1., 1., 0.5],
                [1., 1., 0.5],
            ],
            [
                [0.5, 1., 0.5],
                [0.5, 1., 0.5],
                [0.5, 1., 0.5],
            ],
            [
                [0.5, 1., 1.],
                [0.5, 1., 1.],
                [0.5, 1., 1.],
            ]
        ]])
    )

def test3():
    offsets = np.array([
        [(0, 0), (0, 75)],
        [(60, 0), (60, 75)],
        [(120, 0), (120,75)]
    ], dtype=np.int32)

    size = 100

    masks = blending_masks(offsets, size=size)

    width = np.max(offsets[:,:,0]) + size
    height = np.max(offsets[:,:,1]) + size
    pic = np.zeros((width, height), dtype=np.float32)
    for x in range(offsets.shape[0]):
        for y in range(offsets.shape[1]):
            offx, offy = offsets[x,y]
            pic[offx:offx+size,offy:offy+size] += masks[x,y]

    np.testing.assert_almost_equal(np.min(pic), 1.)
    np.testing.assert_almost_equal(np.max(pic), 1.)

