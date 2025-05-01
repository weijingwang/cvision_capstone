# data.py

import os, glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split, distributed, SequentialSampler
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ConvertImageDtype
from config import STUDENT_IMG_SIZE, TEACHER_IMG_SIZE  # e.g. 384

# 1) Grab teacher img_size from the config
teacher_img_size = TEACHER_IMG_SIZE

# 2) Define two transforms
hr_transform = Compose([
    Resize((teacher_img_size, teacher_img_size)),
    ToTensor(),
    Normalize([0.5]*3, [0.5]*3),
    ConvertImageDtype(torch.float32),
])
lr_transform = Compose([
    Resize((STUDENT_IMG_SIZE, STUDENT_IMG_SIZE)),
    ToTensor(),
    Normalize([0.5]*3, [0.5]*3),
    ConvertImageDtype(torch.float32),
])

class DualResDataset(Dataset):
    def __init__(self, root_dir):
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.jpg")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        hr = hr_transform(img)   # [3,1536,1536]
        lr = lr_transform(img)   # [3, 384, 384]
        return {"image_hr": hr, "image_lr": lr}

def get_data_loaders(root_dir, batch_size, val_frac=0.1, test_frac=0.1, seed=42):
    full = DualResDataset(root_dir)
    n = len(full)
    n_val  = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        full, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    def collate_fn(batch):
        hrs = torch.stack([b['image_hr'] for b in batch], dim=0)
        lrs = torch.stack([b['image_lr'] for b in batch], dim=0)
        return {"image_hr": hrs, "image_lr": lrs}

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=distributed.DistributedSampler(train_ds),
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        sampler=SequentialSampler(val_ds),
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        sampler=SequentialSampler(test_ds),
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    return train_loader, val_loader, test_loader