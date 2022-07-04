import os
import numpy as np
import torch
from PIL import Image



class ImageDataset(torch.utils.data.Dataset):
    def __init__(self,image_dir = None, transform = None):
        super().__init__()

        self.image_dir = image_dir
        self.image_names = os.listdir(image_dir)
        self.transform = transform

    def __getitem__(self,idx):
        image_path = os.path.join(self.image_dir,self.image_names[idx])
        img = Image.open(image_path)        
        if self.transform is not None:
            img = self.transform(img)
    

        return img

    def __len__(self):
        return len(self.image_names)



if __name__ == '__main__':
    dataset = ImageDataset(image_dir='/home/tr/Desktop/ML/dataset/teknofest_2022/deneme/deneme/images')
    print("Number of samples: {}".format(len(dataset)))
        


