# -*- coding: utf-8 -*-

"""
@author:zhangmeixian
@time: 2021-10-22 12:00:00
"""

import sys
import torch
import datetime
import torch.nn as nn
import torch.optim as optim
from utils.log_util import Logger
from torchvision import transforms
from utils.dataset_util import MyDataset
from torch.utils.data import DataLoader
from classifier.AlexNet import AlexNet
from classifier.VGG import vgg

sys.stdout = Logger("./logs/log_{}.txt".format(datetime.datetime.now()))

BATCH_SIZE = 32
EPOCH = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])
train_set = MyDataset("data/dataset/train_set_LogMel", transform=transform)
train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)

test_set = MyDataset("data/dataset/test_set_LogMel", transform=transform)
test_loader = DataLoader(dataset=test_set, batch_size=180, shuffle=False)

test_data_iter = iter(test_loader)
test_image, test_label = test_data_iter.next()


model_name = "vgg16"
net = vgg(model_name=model_name, num_classes=4, init_weights=True)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0002)
save_path = "./AlexNet.pth"
torch.save(net.state_dict(), save_path)

for epoch in range(EPOCH):
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        with torch.no_grad():
            outputs = net(test_image.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            accuracy = (predict_y == test_label.to(device)).sum().item() / test_label.size(0)
            print("[%d, %5d] train_loss: %.3f, test_accuracy: %.3f" %
                  (epoch + 1, step + 1, running_loss / 500, accuracy))
            running_loss = 0.0

print("Finished Training")







