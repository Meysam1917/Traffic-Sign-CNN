import torch
from torch.utils.data import Dataset

class TrafficSignDataset(Dataset):
    def __init__(self, images, labels):
        # images: (N, 32, 32, 3)
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

        # Normalize [0,255] → [0,1]
        self.images = self.images / 255.0

        # Change shape: HWC → CHW
        self.images = self.images.permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
