import os
import numpy as np
import torch
from PIL import Image



class ImageDataset(torch.utils.data.Dataset):
    def __init__(self,root_dir = None, transform = None):
        super().__init__()

        self.rootdir = root_dir
        image_dirs = os.listdir(root_dir)
        self.image_names = []
        for image_dir in image_dirs:
            image_dir_path = os.path.join(root_dir,image_dir)
            images = os.listdir(image_dir_path)
            for image in images:
                self.image_names.append(os.path.join(image_dir_path,image))
         
        self.transform = transform

    def __getitem__(self,idx):
        image_path = self.image_names[idx]
        img = Image.open(image_path)        
        if self.transform is not None:
            img = self.transform(img)
    

        return img

    def __len__(self):
        return len(self.image_names)



if __name__ == '__main__':
    dataset = ImageDataset(root_dir='/home/tr/Desktop/ML/dataset/teknofest_2022/deneme/deneme/images')
    print("Number of samples: {}".format(len(dataset)))
        


