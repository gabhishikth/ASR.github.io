import torch
from model import NeuralNetwork
from dataset import get_dataloaders


train_loader,test_loader = get_dataloaders()

model = NeuralNetwork()

model.load_state_dict(torch.load("models/circle_model.pth"))

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for X,y in test_loader:

        outputs = model(X)

        predicted = (outputs > 0.5).float()

        total += y.size(0)

        correct += (predicted.squeeze() == y).sum().item()

accuracy = 100 * correct / total

print("Test Accuracy:",accuracy,"%")