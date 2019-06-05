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
    masks = list(blending_masks([ (0,0), (0,2) ], size=3))

    np.testing.assert_almost_equal(
        masks,
        np.array([
            [
                [1., 1., 1.],
                [1., 1., 1.],
                [0.5, 0.5, 0.5],
            ],
            [
                [0.5, 0.5, 0.5],
                [1., 1., 1.],
                [1., 1., 1.],
            ]
        ])
    )

def test2():

    '''
    x x x
    x x x
    xyxyxy
    z z z
    z z z
    '''
    masks = list(blending_masks([ (0,0), (2,0) ], size=3))

    np.testing.assert_almost_equal(
        masks,
        np.array([
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
    masks = list(blending_masks([ (0,0), (2,0), (4,0) ], size=3))

    np.testing.assert_almost_equal(
        masks,
        np.array([
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
        ])
    )