from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import torch
import os
import h5py
import numpy as np

class DepthDataset(Dataset):
    def __init__(self, root_dir, transform=None, depth_transform=None):
        self.samples = []
        self.transform = transform
        self.depth_transform = depth_transform

        for scene in os.listdir(root_dir):
            scene_dir = os.path.join(root_dir, scene)
            if not os.path.isdir(scene_dir):
                continue
            for fname in os.listdir(scene_dir):
                if fname.endswith('.h5'):
                    self.samples.append(os.path.join(scene_dir, fname))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]

        with h5py.File(file_path, 'r') as f:
            rgb = np.array(f['rgb'])
            depth = np.array(f['depth'])

        if self.transform:
            rgb = self.transform(rgb)
        else:
            rgb = torch.from_numpy(rgb).float() / 255.0

        if self.depth_transform:
            depth = self.depth_transform(depth)
        else:
            depth = torch.from_numpy(depth).float().unsqueeze(0)

        return rgb, depth

def rgb_transform(rgb_np):
    rgb_img = Image.fromarray(np.transpose(rgb_np, (1, 2, 0)).astype(np.uint8))
    
    transform_pipeline = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(), 
    ])
    
    return transform_pipeline(rgb_img)

def depth_transform(depth_np):
    depth_img = Image.fromarray(depth_np.astype(np.float32))
    
    transform_pipeline = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Lambda(lambda x: torch.clamp(x, 0, 10) / 10.0),
    ])
    
    return transform_pipeline(depth_img)
