import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import torchvision.transforms.functional as F
import numpy as np
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset, DataLoader

class InfraredDataset(data.Dataset):
    def __init__(self, file_txt,image_size=(224, 224)):
        fh = open(file_txt, 'r')
        imgs = []
        labels = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append(words[0])
            labels.append(int(words[1]))

        self.imgs = imgs
        self.labels = labels
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])

    def __getitem__(self, index):
        ret = {}
        rgb_path = self.imgs[index]

        infrared_path = self.imgs[index].replace('handgesture_rgbdataset_new','handgesture_infrad_dataset_new') 

        rgb = Image.open(rgb_path).convert('RGB')
        infrared = Image.open(infrared_path).convert('RGB')

        rgb = tf.resize(rgb, [self.image_size[0], self.image_size[1]])
        infrared = tf.resize(infrared, [self.image_size[0], self.image_size[1]])
       
        rgb = self.transform(rgb)

        infrared = self.transform(infrared)

        label = self.labels[index]

        return rgb, infrared, label

    def __len__(self):
        return len(self.imgs)


def test_infrared_dataset():
    txt_path = "D:\DataSets\gesturedectection\infrared_rgb_dataset_sorted.txt"
    dataset = InfraredDataset(txt_path)
    data_loader = DataLoader(dataset, batch_size=2,shuffle=True)
    print(len(data_loader))

    rgb, infrared, label = next(iter(data_loader))
    print(rgb.shape,infrared.shape, label)
        
    unloader = transforms.ToPILImage()
    image = rgb.cpu().clone()  # clone the tensor
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.save('rgb.jpg')

    image = infrared.cpu().clone()  # clone the tensor
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.save('infrared.jpg')
    


if __name__ == "__main__":
    test_infrared_dataset()

