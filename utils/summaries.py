import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence
import cv2 as cv
import numpy as np


class TensorboardSummary(SummaryWriter):

    def visualize_image(self, dataset, image, target, output, global_step, title='Sample'):
        bitmap = image.clone().cpu().data
        truth  = decode_seg_map_sequence(torch.squeeze(target, 1).detach().cpu().numpy(), dataset=dataset)
        pred   = decode_seg_map_sequence(torch.max(output, 0)[1].detach().cpu().numpy(), dataset=dataset)

        sample = make_grid([bitmap, truth, pred], 3, normalize=True, scale_each=True)

        self.add_image(title, sample, global_step)

    def visualize_box(self, dataset, image, target, output, global_step, title='Box_Sample'):
        bitmap = image.clone().cpu().data

        bitmap = bitmap.permute(1, 2, 0).numpy()
        height, width = bitmap.shape[:2]

        x, y, w, h = target.cpu().data.tolist()
        x0 = x - w/2
        x1 = x + w/2
        y0 = y - h/2
        y1 = y + h/2

        x0 *= width
        x1 *= width
        y0 *= height
        y1 *= height

        print('Good box: %r %r %r %r' % (x0, y0, x1, y1))
        bmin = bitmap.min()
        bmax = bitmap.max()
        bitmap = (bitmap-bmin) * 255 / (bmax-bmin)
        bitmap = bitmap.astype(np.uint8)
        c = bitmap.copy()
        cv.rectangle(c, (int(x0), int(y0)), (int(x1), int(y1)), color=(0, 255, 0))

        x, y, w, h = output.cpu().data.tolist()
        x0 = x - w/2
        x1 = x + w/2
        y0 = y - h/2
        y1 = y + h/2

        x0 *= width
        x1 *= width
        y0 *= height
        y1 *= height

        print('Predicted box: %r %r %r %r' % (x0, y0, x1, y1))
        cv.rectangle(c, (int(x0), int(y0)), (int(x1), int(y1)), color=(0, 0, 255))
#        cv.imshow('XXX', c)

        bitmap = torch.from_numpy(c).float().permute(2, 0, 1)

        sample = make_grid([bitmap], 1, normalize=True, scale_each=True)

        self.add_image(title, sample, global_step)


if __name__ == '__main__':
    from dataloaders.datasets.scarlet import ScarletSegmentation
    from dataloaders import custom_transforms as tr
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from types import SimpleNamespace
    import logging

    logging.basicConfig(level=logging.INFO)

    args = SimpleNamespace()

    dataset = ScarletSegmentation(args, split='train')

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    summary = TensorboardSummary(logdir='testme')

    for ii, sample in enumerate(dataloader):
        batch_size = sample["image"].shape[0]
        pred = torch.zeros(batch_size, 4)
        for jj in range(batch_size):
            img = sample['image'][jj]
            gt = sample['label'][jj]
            p  = pred[jj]

            summary.visualize_box('scarlet', img, gt, p, jj+1)
        break

    cv.waitKey()


