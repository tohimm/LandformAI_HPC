from torch.utils.data import Dataset
import os
import numpy as np
from skimage import io
import torch
from torchvision import transforms
import albumentations as A

class GMKDataset(Dataset):
    def __init__(self, img_dir, device, classes, transform=None, augment=False):

        self.classes = classes
        self.device = device
        self.augment = augment

        img_paths = []
        for dir in img_dir:
            for file in os.listdir(dir):
                if ("mask" not in file) & (file.endswith(".tif")):
                    img_paths.append(os.path.join(dir, file))

        self.img_paths = img_paths

        self.transform = transform

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        # Augmentation pipeline: geometric only (no colour jitter — data is
        # geomorphometric floats, not RGB photos). Applied to image + mask together.
        self.aug = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),                              # swap row/col axes
            A.GaussNoise(std_range=(0.001, 0.01), p=0.3),   # small sensor-noise simulation
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        img_path = self.img_paths[idx]
        mask_path = img_path.replace(".tif", "_mask.tif")

        img_arr = np.atleast_3d(io.imread(img_path)) #use only the DTM
        if img_arr.shape[2] == 1:                    # skimage collapses identical bands to 2D, repeat to 3 channels
            img_arr = np.repeat(img_arr, 3, axis=2)
        img_arr = np.where(img_arr < -9999, 0.0, img_arr).astype(np.float32)  # replace nodata (-99999) with 0

        IGNORE_INDEX = 255   # pixels excluded from loss and metrics

        gmk_arr = np.atleast_3d(io.imread(mask_path))
        gmk_arr = gmk_arr[:, :, :1]   # keep only first band (some label rasters are multi-band)
        nodata_mask = (gmk_arr[:, :, 0] == 0)  # original label 0 = no-data (valid range 1-20)
        # Note: must capture nodata BEFORE the subtraction below, because uint8 underflow
        # would silently map 0 → 255 → clip to classes-1 (last valid class), corrupting metrics.
        gmk_arr -= 1    #labels range from 1 to 20; they need to start with 0
        gmk_arr = np.clip(gmk_arr,a_min=0, a_max=self.classes-1).astype(np.uint8)
        gmk_arr[nodata_mask, 0] = IGNORE_INDEX  # re-assign nodata pixels; excluded from loss/metrics

        # Apply augmentation to image + mask jointly (training only)
        if self.augment:
            augmented = self.aug(image=img_arr, mask=gmk_arr[:, :, 0])
            img_arr  = augmented["image"]
            gmk_arr  = augmented["mask"][:, :, np.newaxis]  # restore channel dim

        # put it from HWC to CHW format
        img_tensor = self.transform(img_arr)
        mask_tensor = np.transpose(gmk_arr, (2, 0, 1))

        assert mask_tensor.dtype == np.uint8, "Mask must be np.uint8"
        assert img_tensor.dtype == torch.float32, "Image must be np.float32 not %s" % str(img_tensor.dtype)

        #https://github.com/qubvel-org/segmentation_models.pytorch/discussions/454
        return img_tensor.to(self.device), torch.from_numpy(mask_tensor).long().to(self.device)
