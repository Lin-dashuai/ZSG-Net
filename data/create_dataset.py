"""Binary segmentation image/mask pairs matched by filename stem."""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data import util


class CreateDataset(Dataset):
    def __init__(self, img_paths, label_paths, resize, phase, aug=False):
        self.resize, self.phase, self.aug = resize, phase, aug
        self.paths_imgs = util.get_image_paths(img_paths)
        mask_paths = util.get_image_paths(label_paths)
        masks = {Path(path).stem: path for path in mask_paths}
        image_stems = [Path(path).stem for path in self.paths_imgs]
        if not image_stems or len(set(image_stems)) != len(image_stems):
            raise ValueError('Image directory must contain nonempty, unique filename stems.')
        if len(masks) != len(mask_paths) or set(image_stems) != set(masks):
            raise ValueError('Image and mask directories must have matching, unique filename stems.')
        self.paths_label = [masks[stem] for stem in image_stems]

    def __getitem__(self, index):
        path = self.paths_imgs[index]
        image = cv2.resize(util.read_img(path), (self.resize, self.resize))
        label = cv2.resize(util.read_nodule_label(self.paths_label[index]),
                           (self.resize, self.resize), interpolation=cv2.INTER_NEAREST)
        if self.phase == 'train' and self.aug:
            image, label = util.augment([image, label])
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        label = torch.from_numpy(np.ascontiguousarray(label)).long()
        return image, label, path

    def __len__(self):
        return len(self.paths_imgs)
