import torch
from torchvision import models, transforms
from dataset4test import CustomDataset
from torch.utils.data import DataLoader
import torch.nn as nn

class Predicted():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 131)
        self.model = model.to(self.device)
        model.load_state_dict(torch.load("./checkpointA.pth"))

        names = []
        labels = []

        with open("./labels.txt", "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                #For IG1k
                # tmp = line.split(":")
                # labels.append(tmp[0])
                # names.append(tmp[1].split("'")[1].split("'")[0])
                labels.append(idx)
                names.append(line)
        self.lists = dict(zip(labels, names))
        model.eval()
    def predict(self, data):
        testDataset = CustomDataset(data)
        testLoader = DataLoader(testDataset, batch_size = 1)
        res = []
        for image in testLoader:
            input = image.to(self.device)
            predicted_res = self.model(input)
            id = str(torch.argmax(predicted_res, dim=1)).split('[')[1].split(']')[0]
        return self.lists[int(id)]
