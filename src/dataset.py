import os
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensor, ToTensorV2


def get_augmentations(p=0.5):
    imagenet_stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    train_tfms = A.Compose(
        [
            A.Resize(128, 128),
            A.Cutout(p=p),
            A.RandomRotate90(p=p),
            A.Flip(p=p),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50
                    ),
                ],
                p=p,
            ),
            A.OneOf([A.IAAAdditiveGaussianNoise(), A.GaussNoise(),], p=p),
            A.OneOf(
                [
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ],
                p=p,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=p
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.1),
                    A.IAAPiecewiseAffine(p=0.3),
                ],
                p=p,
            ),
            ToTensorV2(),
        ]
    )

    test_tfms = A.Compose([A.Resize(128, 128), ToTensorV2()])
    return train_tfms, test_tfms


class MyDataset(Dataset):
    def __init__(self, df_data, phase="train", data_dir="", transform=None):
        super().__init__()
        self.df = df_data.values
        self.phase = phase
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name, label = self.df[index]
        img_path = os.path.join(self.data_dir, img_name)
        image = cv2.imread(img_path)
        if self.transform is not None:
            image = self.transform(**{"image": np.array(image)})["image"]
        if self.phase in ["train", "eval"]:
            return image, label
        # test
        return image, img_name, label


