import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence


class TensorboardSummary(SummaryWriter):

    def visualize_image(self, dataset, image, target, output, global_step, title='Sample'):
        bitmap = image.clone().cpu().data
        truth  = decode_seg_map_sequence(torch.squeeze(target, 1).detach().cpu().numpy(), dataset=dataset)
        pred   = decode_seg_map_sequence(torch.max(output, 0)[1].detach().cpu().numpy(), dataset=dataset)

        sample = make_grid([bitmap, truth, pred], 3, normalize=True, scale_each=True)

        self.add_image(title, sample, global_step)
