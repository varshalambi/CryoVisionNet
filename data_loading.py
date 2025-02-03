##### data_loading.py #####
import numpy as np
import torch
from monai.data import DataLoader, CacheDataset
from monai.transforms import Compose, EnsureChannelFirstd, Orientationd, NormalizeIntensityd
from config import TRAIN_DATA_DIR, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, NUM_WORKERS, CACHE_RATE

def load_data(train_names, valid_names):
    """
    Load training and validation datasets.
    """
    train_files = [{"image": np.load(f"{TRAIN_DATA_DIR}/train_image_{name}.npy"),
                    "label": np.load(f"{TRAIN_DATA_DIR}/train_label_{name}.npy")}
                   for name in train_names]

    valid_files = [{"image": np.load(f"{TRAIN_DATA_DIR}/train_image_{name}.npy"),
                    "label": np.load(f"{TRAIN_DATA_DIR}/train_label_{name}.npy")}
                   for name in valid_names]

    return train_files, valid_files

def get_dataloaders(train_files, valid_files):
    """
    Create DataLoader instances.
    """
    transforms = Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS")
    ])

    train_ds = CacheDataset(train_files, transform=transforms, cache_rate=CACHE_RATE)
    valid_ds = CacheDataset(valid_files, transform=transforms, cache_rate=CACHE_RATE)

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_ds, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, valid_loader

