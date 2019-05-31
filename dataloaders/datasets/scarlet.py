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
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ScarletSegmentation(Dataset):
    NUM_CLASSES = 2

    def __init__(self,
        args,
        split='train',
    ):
        super().__init__()

        self._dataset = ic.get_dataset('ilabs.vision', 'scarlet200')
        files = list(self._dataset[split])
        images = sorted(f for f in files if f.endswith('.png') and not f.endswith('-mask.png'))
        masks  = sorted(f for f in files if f.endswith('-mask.png'))
        assert len(images) == len(masks)
        self._images = images
        self._masks = masks
        self.split = split
        self.args = args

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        image = Image.open(self._images[index]).convert('RGB')
        mask  = Image.open(self._masks[index]).convert('L')
        pix = np.array(mask.getdata()).reshape(mask.size[1], mask.size[0])
        pix = (pix < 128).astype(np.int32)
        mask = Image.fromarray(pix)

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

if __name__ == "__main__":
    from dataloaders import custom_transforms as tr
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import matplotlib
    from types import SimpleNamespace

    matplotlib.use('TkAgg')

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
            segmap = decode_segmap(tmp, dataset='coco')
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