## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class NaimishNet(nn.Module):

    def __init__(self):
        super(NaimishNet, self).__init__()

        # Conv Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4, 4))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1))

        # Pooling Layer
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout Layers
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.6)

        # Fully-connected Layers
        self.fc1 = nn.Linear(in_features=6400, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=500)
        self.fc3 = nn.Linear(in_features=500, out_features=136)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        m.weight = I.uniform(m.weight, a=0, b=1)
        #    elif isinstance(m, nn.Linear):
        #        m.weight = I.xavier_uniform(m.weight, gain=1)

    def forward(self, x):
        x = self.pool(F.elu(self.conv1(x)))
        x = self.drop1(x)

        x = self.pool(F.elu(self.conv2(x)))
        x = self.drop2(x)

        x = self.pool(F.elu(self.conv3(x)))
        x = self.drop3(x)

        x = self.pool(F.elu(self.conv4(x)))
        x = self.drop4(x)

        # Flatten
        x = x.view(x.size(0), -1)

        x = F.elu(self.fc1(x))
        x = self.drop5(x)

        x = F.elu(self.fc2(x))
        x = self.drop6(x)

        x = self.fc3(x)

        return x


class TobisTestNet(nn.Module):

    def __init__(self):
        super(TobisTestNet, self).__init__()

        # Conv Layers and Batch normalisations
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5))
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4))
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.conv3_bn = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2))
        self.conv4_bn = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1))
        self.conv5_bn = nn.BatchNorm2d(512)

        # Pooling Layer
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout Layers
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.6)
        self.drop7 = nn.Dropout(p=0.7)

        # Fully-connected Layers
        self.fc1 = nn.Linear(in_features=2048, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=1000)
        self.fc3 = nn.Linear(in_features=1000, out_features=500)
        self.fc4 = nn.Linear(in_features=500, out_features=136)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = I.uniform(m.weight, a=0, b=1)
            elif isinstance(m, nn.Linear):
                m.weight = I.xavier_uniform(m.weight, gain=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.drop1(x)

        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.drop2(x)

        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.drop3(x)

        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        x = self.drop4(x)

        x = self.pool(F.relu(self.conv5_bn(self.conv5(x))))
        x = self.drop5(x)

        # Flatten
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.drop6(x)

        x = F.relu(self.fc2(x))
        x = self.drop7(x)

        x = F.relu(self.fc3(x))
        x = self.drop7(x)

        x = self.fc4(x)

        return x
