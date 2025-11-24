import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SolarImageDataset(Dataset):
    def __init__(self, imgPaths, targets, imgSize=(128,128), transform=None):
        assert len(imgPaths) == len(targets)
        self.imgPaths = imgPaths
        self.targets = targets.astype(np.float32)
        self.imgSize = imgSize
        self.transform = transform

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, idx):
        p = self.imgPaths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = img.resize(self.imgSize)
            img = transforms.ToTensor()(img)
        y = torch.tensor(self.targets[idx], dtype=torch.float32).view(1)
        return img, y
