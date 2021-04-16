import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, image_paths, input_size=256, cropsize=224):
        super().__init__()
        
        self.input_size = input_size
        
        self.image_paths = image_paths
        self.len = len(self.image_paths)

        self.transform_x = T.Compose([T.Resize(input_size, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        p = self.image_paths[index]
        x = Image.open(p)
        x = self.transform_x(x)
        
        return x

