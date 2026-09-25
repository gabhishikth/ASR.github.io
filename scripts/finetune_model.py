import torch
from torch import nn, optim
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Dataset class
class Data(Dataset):

    def __init__(self, X, y):

        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):

        return self.X[index], self.y[index]

    def __len__(self):

        return self.len


# Neural network architecture
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


# Create NEW dataset (different noise for fine tuning)
X, y = make_circles(
    n_samples=5000,
    noise=0.1,
    random_state=42
)

# ADD THE THIRD FEATURE
X = np.hstack([X, X[:,0:1]**2])

X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.33,
        random_state=42
)

train_data = Data(X_train, y_train)

train_loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True
)


# Load pretrained model
model = NeuralNetwork()

model.load_state_dict(torch.load("models/circle_model.pth"))

print("Pretrained model loaded")


# Loss function
loss_fn = nn.BCELoss()


# Optimizer (smaller learning rate for fine tuning)
optimizer = optim.SGD(model.parameters(), lr=0.01)


# Fine tuning training loop
for epoch in range(20):

    for X,y in train_loader:

        optimizer.zero_grad()

        pred = model(X)

        loss = loss_fn(pred, y.unsqueeze(-1))

        loss.backward()

        optimizer.step()

    print("Epoch:", epoch, "Loss:", loss.item())


# Save fine tuned model
torch.save(model.state_dict(), "circle_model_finetuned.pth")

print("Fine tuning completed and model saved")