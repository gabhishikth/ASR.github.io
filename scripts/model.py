import torch
from torch import nn


class NeuralNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.layer1 = nn.Linear(3,20)
        self.layer2 = nn.Linear(20,20)
        self.layer3 = nn.Linear(20,10)
        self.output = nn.Linear(10,1)

    def forward(self,x):

        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.sigmoid(self.output(x))

        return x
    