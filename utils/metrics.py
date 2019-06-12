import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def anchor_to_corners(x, y, w, h):
    return x - w/2, y-h/2, x+w/2, y+h/2

def area(x0, y0, x1, y1):
    return (y1 - y0) * (x1 - x0)

def intersection_area( ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
    x0 = max(ax0, bx0)
    x1 = min(ax1, bx1)

    if x1 < x0:
        return 0

    y0 = max(ay0, by0)
    y1 = min(ay1, by1)

    if y1 < y0:
        return 0

    return (y1 - y0) * (x1 - x0)

class RegressionEvaluator:
    def __init__(self):
        self._data = []

    def reset(self):
        self._data.clear()

    def add_batch(self, gt_box, pre_box):

        def iou(x0, y0, w0, h0, x1, y1, w1, h1):
            ax0, ay0, ax1, ay1 = anchor_to_corners(x0, y0, w0, h0)
            bx0, by0, bx1, by1 = anchor_to_corners(x1, y1, w1, h1)

            ia = intersection_area(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1)
            aa = area(ax0, ay0, ax1, ay1)
            ab = area(bx0, by0, bx1, by1)
            return ia / (aa + ab - ia)

        self._data.extend( iou(*box1, *box2) for box1, box2 in zip(gt_box, pre_box))

    def mean_iou(self):
        return sum(self._data) / (len(self._data) + 1e-8)
