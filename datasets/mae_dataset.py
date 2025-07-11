import glob
import numpy as np
from sklearn.model_selection import train_test_split

from monai.data import (
    Dataset,
)

from typing import Optional, Sequence

import torch
import pytorch_lightning as pl

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandShiftIntensityd,
    RandRotate90d,
    Resized,
    ToTensord,
)
from monai.data.utils import pad_list_data_collate

class MAEDataset(Dataset):
    def __init__(self, covid=False, 
                 ild=False, 
                 covid_path=None, 
                 ild_path=None, 
                 test_size = 0.2, 
                 transform=None, 
                 split='train',
                 random_state=42):
        self.files = np.array([], dtype=str)
        self.transform = transform

        if covid and covid_path != None:
            self.files = np.concatenate((self.files, self._get_files(covid_path)))
        if ild and ild_path != None:
            self.files = np.concatenate((self.files, self._get_files(ild_path)))
            
        train_files, val_files = train_test_split(self.files, test_size=test_size, random_state=random_state)

        if split == 'train':
            self.files = train_files
        elif split == 'val':
            self.files = val_files
        else:
            raise ValueError("Invalid split. Use 'train' or 'val'.")

    def _get_files(self, path):
        return np.array(glob.glob(path + '/*.npz'), dtype=str)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        # nifti_image = sitk.ReadImage(path)
        # nifti_array = sitk.GetArrayFromImage(nifti_image)  # Convert to NumPy array (z, y, x)

        sample = {"image": path}
        # print(sample)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample
    
class MAEDatasetNew(Dataset):
    def __init__(self,
                 data_path=None, 
                 test_size = 0.2, 
                 transform=None, 
                 split='train',
                 random_state=42):
        self.files = np.array(glob.glob(data_path + '/*.npz'), dtype=str)
        self.transform = transform

    
        train_files, val_files = train_test_split(self.files, test_size=test_size, random_state=random_state)

        if split == 'train':
            self.files = train_files
        elif split == 'val':
            self.files = val_files
        else:
            raise ValueError("Invalid split. Use 'train' or 'val'.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        # nifti_image = sitk.ReadImage(path)
        # nifti_array = sitk.GetArrayFromImage(nifti_image)  # Convert to NumPy array (z, y, x)
        data = dict(np.load(path, allow_pickle=True))
        # image = data['ct']
        sample = {
            # 'image': np.expand_dims(data['image'],axis=0),
            # 'lung_mask': np.expand_dims(data['lung_mask'],axis=0)
            'image': data['image'],
            'lung_mask': data['lung_mask'],
        }
        # print(sample)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample
    
class MAEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        json_path: str,
        downsample_ratio: Sequence[float],
        batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 4,
        dist: bool = False,
        data_path: str = None,
        split: str = 'train',
        test_size: float = 0.2,
        random_state: int = 42
    ):
        super().__init__()
        self.json_path = json_path
        self.downsample_ratio = downsample_ratio
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.dist = dist
        self.data_path = data_path
        self.split = split
        self.test_size = test_size
        self.random_state = random_state


    def train_transforms(self):
        # SSL training transforms (only for image)
        transforms = Compose(
            [
                Orientationd(keys=["image", "lung_mask"],axcodes="RAS"),  # Ensure consistent orientation
                RandFlipd(keys=["image", "lung_mask"],spatial_axis=[0], prob=0.10),
                RandFlipd(keys=["image", "lung_mask"],spatial_axis=[1], prob=0.10),
                RandFlipd(keys=["image", "lung_mask"],spatial_axis=[2], prob=0.10),
                RandRotate90d(keys=["image", "lung_mask"],prob=0.10, max_k=3),
                # Resized(keys=["image", "lung_mask"], spatial_size=(128,128,128)),
                ToTensord(keys=["image", "lung_mask"]),  # Convert to tensor
            ]
        )
        return transforms

    def val_transforms(self):
        # SSL validation transforms (only for image)
        transforms = Compose(
            [
                Orientationd(keys=["image", "lung_mask"],axcodes="RAS"),
                # Resized(keys=["image", "lung_mask"], spatial_size=(128,128,128)),
                ToTensord(keys=["image", "lung_mask"]),
            ]
        )
        return transforms

    def setup(self, stage: Optional[str] = None):
        self.train_ds = MAEDatasetNew(
                                   data_path=self.data_path,
                                   test_size=self.test_size,
                                   transform=self.train_transforms(),
                                   split='train',
                                   random_state=self.random_state)
        self.val_ds = MAEDatasetNew(
                                data_path=self.data_path,
                                test_size=self.test_size,
                                transform=self.val_transforms(),
                                split='val',
                                random_state=self.random_state)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=pad_list_data_collate,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=pad_list_data_collate,
        )

# class MAEDataModule(pl.LightningDataModule):
#     def __init__(
#         self,
#         json_path: str,
#         downsample_ratio: Sequence[float],
#         batch_size: int = 1,
#         val_batch_size: int = 1,
#         num_workers: int = 4,
#         dist: bool = False,
#         covid: bool = False,
#         ild: bool = False,
#         covid_path: str = None,
#         ild_path: str = None,
#         split: str = 'train',
#         test_size: float = 0.2,
#         random_state: int = 42
#     ):
#         super().__init__()
#         self.json_path = json_path
#         self.downsample_ratio = downsample_ratio
#         self.batch_size = batch_size
#         self.val_batch_size = val_batch_size
#         self.num_workers = num_workers
#         self.dist = dist
#         self.covid = covid
#         self.ild = ild
#         self.covid_path = covid_path
#         self.ild_path = ild_path
#         self.split = split
#         self.test_size = test_size
#         self.random_state = random_state


#     def train_transforms(self):
#         # SSL training transforms (only for image)
#         transforms = Compose(
#             [
#                 Orientationd(keys=["image", "lung_mask"],axcodes="RAS"),  # Ensure consistent orientation
#                 RandFlipd(keys=["image", "lung_mask"],spatial_axis=[0], prob=0.10),
#                 RandFlipd(keys=["image", "lung_mask"],spatial_axis=[1], prob=0.10),
#                 RandFlipd(keys=["image", "lung_mask"],spatial_axis=[2], prob=0.10),
#                 RandRotate90d(keys=["image", "lung_mask"],prob=0.10, max_k=3),
#                 ToTensord(keys=["image", "lung_mask"]),  # Convert to tensor
#             ]
#         )
#         return transforms

#     def val_transforms(self):
#         # SSL validation transforms (only for image)
#         transforms = Compose(
#             [
#                 Orientationd(keys=["image", "lung_mask"],axcodes="RAS"),
#                 CropForegroundd(keys=["image", "lung_mask"],source_key="image"),
#                 ToTensord(keys=["image", "lung_mask"]),
#             ]
#         )
#         return transforms

#     def setup(self, stage: Optional[str] = None):
#         self.train_ds = MAEDatasetNew(covid=self.covid,
#                                    ild=self.ild,
#                                    covid_path=self.covid_path,
#                                    ild_path=self.ild_path,
#                                    test_size=self.test_size,
#                                    transform=self.train_transforms(),
#                                    split='train',
#                                    random_state=self.random_state)
#         self.val_ds = MAEDatasetNew(covid=self.covid,
#                                 ild=self.ild,
#                                 covid_path=self.covid_path,
#                                 ild_path=self.ild_path,
#                                 test_size=self.test_size,
#                                 transform=self.val_transforms(),
#                                 split='val',
#                                 random_state=self.random_state)

#     def train_dataloader(self):
#         return torch.utils.data.DataLoader(
#             self.train_ds,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=True,
#             collate_fn=pad_list_data_collate,
#         )

#     def val_dataloader(self):
#         return torch.utils.data.DataLoader(
#             self.val_ds,
#             batch_size=self.val_batch_size,
#             num_workers=self.num_workers,
#             shuffle=False,
#             collate_fn=pad_list_data_collate,
#         )