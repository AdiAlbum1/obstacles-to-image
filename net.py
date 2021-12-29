import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, 7)
        self.conv2 = nn.Conv2d(128, 512, 7)
        self.conv3 = nn.Conv2d(512, 512, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*64, 128*128)
        self.fc2 = nn.Linear(128*128, 64*64)
        self.fc3 = nn.Linear(64*64, 2)

    def initialize_weights(self, std=0.01):
        self.conv1.weight.data = self.conv1.weight.data.normal_(mean=0.0, std=std)
        self.conv2.weight.data = self.conv2.weight.data.normal_(mean=0.0, std=std)
        self.conv3.weight.data = self.conv3.weight.data.normal_(mean=0.0, std=std)

        self.fc1.weight.data = self.fc1.weight.data.normal_(mean=0.0, std=std)
        self.fc2.weight.data = self.fc2.weight.data.normal_(mean=0.0, std=std)
        self.fc3.weight.data = self.fc3.weight.data.normal_(mean=0.0, std=std)

        self.conv1.bias.data = nn.init.zeros_(self.conv1.bias.data)
        self.conv2.bias.data = nn.init.zeros_(self.conv2.bias.data)
        self.conv3.bias.data = nn.init.zeros_(self.conv3.bias.data)

        self.fc1.bias.data = nn.init.zeros_(self.fc1.bias.data)
        self.fc2.bias.data = nn.init.zeros_(self.fc2.bias.data)
        self.fc3.bias.data = nn.init.zeros_(self.fc3.bias.data)


    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x