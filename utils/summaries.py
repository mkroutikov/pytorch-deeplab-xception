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
        bitmap = image.clone().cpu().data
        pred   = decode_seg_map_sequence(torch.max(output, 1)[1].detach().cpu().numpy(), dataset=dataset)
        truth  = decode_seg_map_sequence(torch.squeeze(target, 1).detach().cpu().numpy(), dataset=dataset)

        sample = make_grid([bitmap, truth, pred], 3, normalize=True, scale_each=True)

        writer.add_image('Sample', sample, global_step)
