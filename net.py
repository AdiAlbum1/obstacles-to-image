import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64*64, 128*128)
        self.fc2 = nn.Linear(128*128, 100*100)
        self.fc3 = nn.Linear(100*100, 80*80)
        self.fc4 = nn.Linear(80*80, 50*50)
        self.fc5 = nn.Linear(50*50, 32*32)
        self.fc6 = nn.Linear(32*32, 2)

    def initialize_weights(self, std=0.01):
        self.fc1.weight.data = self.fc1.weight.data.normal_(mean=0.0, std=std)
        self.fc2.weight.data = self.fc2.weight.data.normal_(mean=0.0, std=std)
        self.fc3.weight.data = self.fc3.weight.data.normal_(mean=0.0, std=std)
        self.fc4.weight.data = self.fc4.weight.data.normal_(mean=0.0, std=std)
        self.fc5.weight.data = self.fc5.weight.data.normal_(mean=0.0, std=std)
        self.fc6.weight.data = self.fc6.weight.data.normal_(mean=0.0, std=std)

        self.fc1.bias.data = nn.init.zeros_(self.fc1.bias.data)
        self.fc2.bias.data = nn.init.zeros_(self.fc2.bias.data)
        self.fc3.bias.data = nn.init.zeros_(self.fc3.bias.data)
        self.fc4.bias.data = nn.init.zeros_(self.fc4.bias.data)
        self.fc5.bias.data = nn.init.zeros_(self.fc5.bias.data)
        self.fc6.bias.data = nn.init.zeros_(self.fc6.bias.data)


    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x