import numpy as np
from torch.utils.data import Dataset, DataLoader

class NumpyDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Ensure image is in a format suitable for Albumentations (H, W, C)
        # Often npy files are saved as (N, C, H, W) or (N, H, W, C)
        if self.transform:
            # Albumentations expects (H, W, C)
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, torch.tensor(label, dtype=torch.long)