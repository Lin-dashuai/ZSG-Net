"""Classification loaders extracted from the original my_datasets_sr.py.

List rows are: <image path relative to root> <integer class label>.
The unused coarse-mask input has been removed from the loaders.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MyValDataSet_cls(Dataset):
    def __init__(self, root_path, list_path, val_size=(512, 512)):
        self.root = Path(root_path)
        self.files = []
        for number, line in enumerate(Path(list_path).read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                name, label = line.strip().rsplit(maxsplit=1)
                label = int(label)
                if label not in (0, 1):
                    raise ValueError('Expected label 0 or 1.')
            except ValueError as error:
                raise ValueError(f'Invalid classification list row {number}: {line}') from error
            self.files.append((name, label))
        if not self.files:
            raise ValueError('Classification list is empty.')
        self.crop_h, self.crop_w = val_size

    def __len__(self):
        return len(self.files)

    def transform(self, image):
        return image.resize((self.crop_w, self.crop_h), Image.Resampling.BILINEAR)

    def __getitem__(self, index):
        name, label = self.files[index]
        with Image.open(self.root / name) as source:
            image = self.transform(source.convert('RGB'))
        image = (np.asarray(image, dtype=np.float32) / 255.0).transpose(2, 0, 1)
        return image.copy(), np.int64(label), name


class MyDataSet_cls(MyValDataSet_cls):
    def __init__(self, root_path, list_path, resize=(512, 512)):
        super().__init__(root_path, list_path, resize)
        self.train_augmentation = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0, 0.1), scale=(0.9, 1.1), shear=5.729578),
            transforms.RandomVerticalFlip(p=0.5), transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(), transforms.ToPILImage(), transforms.Resize(resize)])

    def transform(self, image):
        return self.train_augmentation(image)
