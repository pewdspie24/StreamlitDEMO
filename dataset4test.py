import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import glob
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, img):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.img = img
    def __getitem__(self, idx):
        img = self.img.resize((224,224))
        image = self.transform(img)
        return image

    def __len__(self):
        return 1

if __name__ == '__main__':
    DATA_PATH = "./images"
    abc = CustomDataset(DATA_PATH)
    it = abc.__getitem__(0)
    print(it[1])
    img = np.transpose(it[0], (1,2,0))
    plt.imshow(img)
    plt.show()