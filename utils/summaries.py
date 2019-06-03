import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence
import numpy as np


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(logdir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        bitmap = image[:3].clone().cpu().numpy()  # take 3 images from the batch
        bmin = np.min(bitmap)
        bmax = np.max(bitmap) + 1.e-8
        bitmap = np.rint((bitmap - bmin) * 255 / (bmax-bmin))
        pred   = decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(), dataset=dataset)
        truth  = decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(), dataset=dataset)

        sample1 = make_grid([bitmap[0], truth[0], pred[0]], 3, normalize=False, range=(0, 255))
        sample2 = make_grid([bitmap[1], truth[1], pred[1]], 3, normalize=False, range=(0, 255))
        sample3 = make_grid([bitmap[2], truth[2], pred[2]], 3, normalize=False, range=(0, 255))

        writer.add_image('Sample 1', sample1, global_step)
        writer.add_image('Sample 2', sample2, global_step)
        writer.add_image('Sample 3', sample3, global_step)
