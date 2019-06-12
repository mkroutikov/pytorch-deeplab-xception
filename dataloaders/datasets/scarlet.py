import numpy as np
import torch
from torch.utils.data import Dataset
from mypath import Path
from tqdm import trange, tqdm
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
from dataloaders.datasets.blending_masks import blending_masks
import math
import lxml.etree as et

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ScarletSegmentation(Dataset):
    NUM_CLASSES = 3

    MASKS = 'scarlet300-line-masks-%s.pickle'

    CACHE_BOX = 'scarlet300-box-%s.pickle'

    def __init__(self,
        args,
        split='train',
    ):
        super().__init__()

        self._dataset = ic.get_dataset('ilabs.vision', 'scarlet300')
        files = list(self._dataset[split])
        images = sorted(f for f in files if f.endswith('.png'))

        masks_filename = self.CACHE_BOX % split
        if not os.path.exists(masks_filename):
            print('Generating CACHE for split', split)
            masks = [generate_first_box(fname) for fname in tqdm(images)]
            torch.save(masks, masks_filename)
        else:
            masks = torch.load(masks_filename)
        assert len(images) == len(masks)
        self._images = images
        self._masks = masks
        self.split = split
        self.args = args

        if split == 'train':
            self._transform = transforms.Compose([
                # tr.RandomHorizontalFlip(),
                # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=0xffffff),
                tr.RandomGaussianBlur(),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])
        elif split == 'test':
            self._transform = transforms.Compose([
                # tr.FixScaleCrop(crop_size=self.args.crop_size),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()
            ])
        else:
            raise ValueError('Unknown split: ' + split)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        image = Image.open(self._images[index]).convert('RGB')
        mask  = np.array(self._masks[index], dtype=np.float32)

        sample = {
            'image': image,
            'label': mask
        }

        return self._transform(sample)


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


def generate_border_mask(fname, thickness=5):
    assert fname.endswith('.png')

    with open(fname[:-4] + '.json') as f:
        meta = json.load(f)

    w, h = meta['size']
    zones = meta['zones']

    mask = np.zeros((h,w), np.uint8)
    for z in zones:
        x0, y0, x1, y1 = z['bbox']
        cv.rectangle(mask, (x0, y0), (x1, y0+thickness), color=1, thickness=-1)
        cv.rectangle(mask, (x0, y1-thickness), (x1, y1), color=2, thickness=-1)
        cv.rectangle(mask, (x0, y0), (x0+thickness, y1), color=3, thickness=-1)
        cv.rectangle(mask, (x1-thickness, y0), (x1, y1), color=4, thickness=-1)

    return mask

def generate_volume_mask(fname):
    assert fname.endswith('.png')

    with open(fname[:-4] + '.json') as f:
        meta = json.load(f)

    w, h = meta['size']
    zones = meta['zones']

    mask = np.zeros((h,w), np.uint8)
    for z in zones:
        x0, y0, x1, y1 = z['bbox']
        cv.rectangle(mask, (x0, y0), (x0 + (x1-x0)//2, y1), color=1, thickness=-1)
        cv.rectangle(mask, (x0 + (x1-x0)//2, y0), (x1, y1), color=2, thickness=-1)
        cv.fillConvexPoly(mask, np.array([(x0, y0), ((x0+x1)//2, (y0+y1)//2), (x1, y0)], dtype=np.int32), color=3)
        cv.fillConvexPoly(mask, np.array([(x0, y1), ((x0+x1)//2, (y0+y1)//2), (x1, y1)], dtype=np.int32), color=4)

    return mask

def generate_line_mask(fname):
    assert fname.endswith('.png')

    with open(fname[:-4] + '.xml') as f:
        xml = et.fromstring(f.read())

    w, h = int(xml.attrib['width']), int(xml.attrib['height'])
    mask = np.zeros((h,w), np.uint8)

    for line in xml.findall('.//line'):
        l, t, r, b = (float(line.attrib[x]) for x in 'ltrb')

        topslack = (b-t) / 3
        botslack = (b-t) / 8
        x0, y0, x1, y1 = tuple(int(x) for x in (l, t+topslack, r, b-botslack))

        cv.rectangle(mask, (x0, y0), (x1, y1), color=2, thickness=-1)

        for box in line:
            l, t, r, b = (int(float(box.attrib[x])) for x in 'ltrb')
            cv.rectangle(mask, (l, y0), (r, y1), color=1, thickness=-1)
    return mask


def generate_first_box(fname):
    assert fname.endswith('.png')

    with open(fname[:-4] + '.xml') as f:
        xml = et.fromstring(f.read())

    w, h = int(xml.attrib['width']), int(xml.attrib['height'])

    x0, y0, x1, y1 = 0., 0., 0., 0.
    for block in xml.findall('.//block'):
        x0, y0, x1, y1 = (float(block.attrib[x]) for x in 'ltrb')
        break

    x0 /= w
    x1 /= w
    y0 /= h
    y1 /= h

    return ((x0+x1)/2, (y0+y1)/2, x1-x0, y1-y0)


def image_to_crops(image, cropsize=513, overlap=0.1):
    '''
    Input: a PIL image
    Output: a set of square images of size cropsize x cropsize that cover the original image.
    '''
    width, height = image.size

    if width < height:
        ow = cropsize
        oh = int(height * cropsize / width)
        numcrops = int(math.ceil((oh/cropsize - overlap) / (1 - overlap)))
        delta = (oh-cropsize) / (numcrops - 1)
        offsets = [(0, int(i*delta)) for i in range(numcrops)]
    else:
        ow = int(width * cropsize / height)
        oh = cropsize
        numcrops = int(math.ceil((ow/cropsize - overlap) / (1 - overlap)))
        delta = (ow-cropsize) / (numcrops - 1)
        offsets = [(int(i*delta), 0) for i in range(numcrops)]

    resized = image.resize((ow, oh), Image.BILINEAR)
    crops = [resized.crop( (x, y, x+cropsize, y+cropsize) ) for x,y in offsets]

    return {
        'width': ow,
        'height': oh,
        'offsets': offsets,
        'crops': crops,
    }


def image_to_crops(image, cropsize=513, overlap=0.):
    '''
    Input: a PIL image
    Output: a set of square images of size cropsize x cropsize that cover the original image.
    '''
    width, height = image.size

    numy = int(math.ceil((height/cropsize - overlap) / (1 - overlap)))
    numx = int(math.ceil((width/cropsize - overlap) / (1 - overlap)))

    deltay = (height - cropsize) / (numy - 1)
    deltax = (width - cropsize) / (numx - 1)

    offx = np.array([[int(i*deltax), 0] for i in range(numx)])
    offx = np.expand_dims(offx, axis=1)
    offy = np.array([[0, int(i*deltay)] for i in range(numy)])
    offy = np.expand_dims(offy, axis=0)

    offsets = offx + offy
    assert len(offsets.shape) == 3, offsets.shape

    crops = [image.crop( (x, y, x+cropsize, y+cropsize) ) for x,y in offsets.reshape( (-1, 2) )]

    return crops, offsets


def glue_logits(logits, offsets):
    '''
    logits: array of length len(offsets) of numpy tensors [S, S, N], where S - is the cropsize, N - number of classes

    assume that only two crops can intersect (the neighbors)
    '''
    logits = logits.transpose(0, 2, 3, 1)  # B, L, W, H => B, W, H, L
    cropsize = logits[0].shape[0]
    assert cropsize == logits[0].shape[1]

    N = logits[0].shape[2]

    masks = blending_masks(offsets, size=cropsize)
    masks = masks.reshape( (-1, masks.shape[-2], masks.shape[-1]))

    output_width = offsets[-1,-1,0] + cropsize
    output_height = offsets[-1,-1,1] + cropsize

    big_logits = np.zeros( (output_width, output_height, N) )
    offsets = offsets.reshape( (-1, 2) )
    assert logits.shape[0] == offsets.shape[0]
    for logit, (x, y), mask in zip(logits, offsets, masks):
        mask = np.expand_dims(mask, 2)  # [S, S, 1]
        l = logit * mask  #  [S, S, N]
        big_logits[x:x+cropsize,y:y+cropsize] += l.transpose(1, 0, 2)

    return big_logits


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
            segmap = decode_segmap(tmp, dataset='scarlet')
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