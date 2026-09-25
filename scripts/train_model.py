import torch
from torch import nn,optim

import numpy as np
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import NeuralNetwork


train_loader,test_loader = get_dataloaders()

model = NeuralNetwork()

loss_fn = nn.BCELoss()

optimizer = optim.SGD(model.parameters(),lr=0.1)

num_epochs = 100
loss_values = []

for epoch in range(num_epochs):

    for X,y in train_loader:

        optimizer.zero_grad()

        pred = model(X)

        loss = loss_fn(pred,y.unsqueeze(-1))

        loss.backward()

        optimizer.step()

        loss_values.append(loss.item())

    print("Epoch:",epoch,"Loss:",loss.item())

torch.save(model.state_dict(),"models/circle_model.pth")

print("Model saved successfully")
step = range(len(loss_values))

fig, ax = plt.subplots(figsize=(8,5))
plt.plot(step, np.array(loss_values))
plt.title("Step-wise Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()