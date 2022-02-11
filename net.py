import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 5, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 5, stride=2)
        self.conv3 = nn.Conv2d(128, 256, 5, stride=2)

        self.pool = nn.MaxPool2d((2,2))
        # self.dropout = F.dropout(p=0.2)

        self.fc1 = nn.Linear(6400, 3200)
        self.fc2 = nn.Linear(3200, 4)

    def initialize_weights(self, std=0.005):
        # self.conv1.weight.data = self.conv1.weight.data.normal_(mean=0.0, std=0.0220)
        # self.conv2.weight.data = self.conv2.weight.data.normal_(mean=0.0, std=0.008333)
        # self.conv3.weight.data = self.conv3.weight.data.normal_(mean=0.0, std=0.01359)
        # self.fc1.weight.data = self.fc1.weight.data.normal_(mean=0.0, std=0.025)
        # self.fc2.weight.data = self.fc2.weight.data.normal_(mean=0.0, std=0.0353)

        self.conv1.weight.data = self.conv1.weight.data.normal_(mean=0.0, std=std)
        self.conv2.weight.data = self.conv2.weight.data.normal_(mean=0.0, std=std)
        self.conv3.weight.data = self.conv3.weight.data.normal_(mean=0.0, std=std)
        self.fc1.weight.data = self.fc1.weight.data.normal_(mean=0.0, std=std)
        self.fc2.weight.data = self.fc2.weight.data.normal_(mean=0.0, std=std)

        self.conv1.bias.data = nn.init.zeros_(self.conv1.bias.data)
        self.conv2.bias.data = nn.init.zeros_(self.conv2.bias.data)
        self.conv3.bias.data = nn.init.zeros_(self.conv3.bias.data)
        self.fc1.bias.data = nn.init.zeros_(self.fc1.bias.data)
        self.fc2.bias.data = nn.init.zeros_(self.fc2.bias.data)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        return x