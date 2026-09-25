import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


class Data(Dataset):

    def __init__(self,X,y):

        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self,index):

        return self.X[index], self.y[index]

    def __len__(self):

        return self.len


def get_dataloaders(batch_size=64):

    X,y = make_circles(
        n_samples=10000,
        noise=0.05,
        random_state=26
    )

    # change input shape (feature engineering)
    X_new = np.hstack([X, X[:,0:1]**2])

    X_train,X_test,y_train,y_test = train_test_split(
        X_new,
        y,
        test_size=0.33,
        random_state=26
    )

    train_data = Data(X_train,y_train)
    test_data = Data(X_test,y_test)

    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=batch_size)

    return train_loader,test_loader