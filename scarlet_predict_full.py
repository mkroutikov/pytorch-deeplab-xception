from dataloaders.datasets.scarlet import image_to_crops, glue_logits
from dataloaders.datasets.blending_masks import blending_masks
from dataloaders.custom_transforms import normalize_rgb_image, normalized_image_to_tensor
from modeling.deeplab import DeepLab
import numpy as np
from PIL import Image
import torch
from torchvision.utils import make_grid


def predict(model, image):
    width, height = image.size

    minibatch = [
        normalized_image_to_tensor(
            normalize_rgb_image(
                image,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        )
    ]
    minibatch = torch.stack(minibatch)

    with torch.no_grad():
        logits = model(minibatch).detach().cpu().numpy()

    logits = logits.squeeze(0).transpose(1, 2, 0)
    mask = np.argmax(logits, axis=2)
    confidence = np.min(np.abs(logits), axis=2)

    # color the mask
    color = np.array([
        [0,   0,   0],  # black
        [255, 0,   0],  # red
        [0,   255, 0],  # green
        [0,   0, 255],  # blue
        [0,   255, 255],  # ??
    ], dtype=np.int32)
    color = np.expand_dims(color, axis=1)
    color = np.expand_dims(color, axis=1)

    mask = np.expand_dims(mask, 2)  # add color dimension
    r = (mask == 0) * color[0]
    g = (mask == 1) * color[1]
    b = (mask == 2) * color[2]
    c = (mask == 3) * color[3]
    d = (mask == 4) * color[4]
    mask = (r + g + b + c + d).astype(np.uint8)

    image = Image.fromarray( mask )
    mask = np.array(image, dtype=np.float32)

    return image


def main(checkpoint_filename, input_image, output_image):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define network
    model = DeepLab(num_classes=3,
                    backbone='resnet',
                    output_stride=16,
                    sync_bn=False,
                    freeze_bn=False)


    checkpoint = torch.load(checkpoint_filename, map_location=device)
    state_dict = checkpoint['state_dict']
    # because model was saved with DataParallel, stored checkpoint contains "module" prefix that we want to strip
    state_dict = {
        key[7:] if key.startswith('module.') else key: val
        for key,val in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model.eval()

    image = Image.open(input_image).convert('RGB')
    mask = predict(model, image)

    mask.save(output_image)

if __name__ == '__main__':

    import fire

    fire.Fire(main)

