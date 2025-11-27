
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
    
    def __init__(self, path):
        data = np.loadtxt(path)
        y = data[:, 0].astype(np.int64) - 1
        x = data[:, 1:].astype(np.float32)

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

        self.num_classes = len(np.unique(y))
        self.input_dim = self.x.shape[1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def create_dataloaders(train_path, test_path, batch_size=64, num_workers=0):
    train_dataset = Dataset(train_path)
    test_dataset = Dataset(test_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    meta = {
        "num_classes": train_dataset.num_classes,
        "input_dim": train_dataset.input_dim,
    }

    return train_loader, test_loader, meta
