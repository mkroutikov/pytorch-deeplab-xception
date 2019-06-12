import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence
import cv2 as cv


class TensorboardSummary(SummaryWriter):

    def visualize_image(self, dataset, image, target, output, global_step, title='Sample'):
        bitmap = image.clone().cpu().data
        truth  = decode_seg_map_sequence(torch.squeeze(target, 1).detach().cpu().numpy(), dataset=dataset)
        pred   = decode_seg_map_sequence(torch.max(output, 0)[1].detach().cpu().numpy(), dataset=dataset)

        sample = make_grid([bitmap, truth, pred], 3, normalize=True, scale_each=True)

        self.add_image(title, sample, global_step)

    def visualize_box(self, dataset, image, target, output, global_step, title='Box Sample'):
        bitmap = image.clone().cpu().data

        bitmap = bitmap.permute(1, 2, 0).numpy()
        height, width = bitmap.shape[:2]

        x, y, w, h = target.cpu().data
        x0 = x - w/2
        x1 = x + w/2
        y0 = y - h/2
        y1 = y + h/2

        x0 *= width
        x1 *= width
        y0 *= height
        y1 *= height

        cv.rectangle(bitmap, (x0, y0), (x1, y1), color=(1., 0, 0))

        x, y, w, h = output.cpu().data
        x0 = x - w/2
        x1 = x + w/2
        y0 = y - h/2
        y1 = y + h/2

        x0 *= width
        x1 *= width
        y0 *= height
        y1 *= height

        cv.rectangle(bitmap, (x0, y0), (x1, y1), color=(0., 1., 0))
        bitmap = torch.from_numpy(bitmap).float().permute(2, 0, 1)

        sample = make_grid([bitmap], 1, normalize=True, scale_each=True)

        self.add_image(title, sample, global_step)
