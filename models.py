## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

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
