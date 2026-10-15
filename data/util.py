"""Image reading and paired augmentation used by CreateDataset."""
from pathlib import Path
import random

import cv2
import numpy as np

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def get_image_paths(dataroot):
    root = Path(dataroot)
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(str(p) for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def read_img(path):
    # Inputs are RGB float32 in [0, 1], as in the original augmented/validation paths.
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def read_nodule_label(path):
    label = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise ValueError(f'Cannot read mask: {path}')
    if not np.isin(label, [0, 1, 255]).all():
        raise ValueError(f'Expected binary mask values 0/1 or 0/255: {path}')
    return (label > 0).astype(np.uint8)


def augment(images):
    hflip, vflip, rot90 = [random.random() < 0.5 for _ in range(3)]
    result = []
    for image in images:
        if hflip:
            image = image[:, ::-1]
        if vflip:
            image = image[::-1]
        if rot90:
            image = image.transpose(1, 0, 2) if image.ndim == 3 else image.T
        result.append(image)
    return result
