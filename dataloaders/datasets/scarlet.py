import numpy as np
import torch
from torch.utils.data import Dataset
from mypath import Path
from tqdm import trange
import os
from pycocotools.coco import COCO
from pycocotools import mask
from torchvision import transforms
from dataloaders import custom_transforms as tr
from PIL import Image, ImageFile
from ilabs.curate import ic
from utils.halo import halo
import logging
import json
import cv2 as cv

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ScarletSegmentation(Dataset):
    NUM_CLASSES = 3

    MASKS = 'scarlet200-masks.pickle'

    def __init__(self,
        args,
        split='train',
    ):
        super().__init__()

        self._dataset = ic.get_dataset('ilabs.vision', 'scarlet200')
        files = list(self._dataset[split])
        images = sorted(f for f in files if f.endswith('.png') and not f.endswith('-mask.png'))

        if not os.path.exists(self.MASKS):
            logging.info('Generating masks')
            masks = [generate_mask(fname) for fname in images]
            torch.save(masks, self.MASKS)
        else:
            masks = torch.load(self.MASKS)
        assert len(images) == len(masks)
        self._images = images
        self._masks = masks
        self.split = split
        self.args = args

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        image = Image.open(self._images[index]).convert('RGB')
        mask  = Image.fromarray(self._masks[index]).convert('L')

        sample = {
            'image': image,
            'label': mask
        }

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'test':
            return self.transform_val(sample)
        else:
            assert False, self.split

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=0xffffff),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)


def generate_mask(fname, kernel_size=(10, 10)):
    assert fname.endswith('.png')

    with open(fname[:-4] + '.json') as f:
        meta = json.load(f)

    w, h = meta['size']
    zones = meta['zones']

    kernel = np.ones(kernel_size, np.uint8)  # FIXME: should we consider non-rectangular brush? -MK

    mask = halo(w, h, zones, kernel)
    mask *= 2  # halo index is now 2

    for z in zones:
        x0, y0, x1, y1 = z['bbox']
        cv.rectangle(mask, (x0, y0), (x1, y1), color=1, thickness=-1)

    return mask


if __name__ == "__main__":
    from dataloaders import custom_transforms as tr
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import matplotlib
    from types import SimpleNamespace

    logging.basicConfig(level=logging.INFO)

    args = SimpleNamespace(
        base_size = 513,
        crop_size = 513,
    )

    dataset = ScarletSegmentation(args, split='train')

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='scarlet200')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)