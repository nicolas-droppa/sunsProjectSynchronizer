from dataUtilities import *
from grapthUtilities import *
from utilities import *

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = X_train.shape[1]

        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    checkForCuda()

    data = loadDataset("zadanie1-data.csv", showInfo=False)

    if data is None:
        print(f"\nData not loaded, exiting...")
        exit(1)

    data = removeColumn(data, "duration", showInfo=True)
    data = dropColumnsWithTooManyNaN(data, threshold=0.25, showInfo=True)
    data = removeOutliersWrapper(data, showInfo=True)

    #plotColumnHistograms(data, bins=50, showInfo=True)

    dataRows, dataColumns = getDataCount(data, showInfo=True)

    x, y = preprocessDataset(data, showInfo=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(x)

    #plotColumnHistograms(pd.DataFrame(X, columns=x.columns), bins=50, showInfo=True)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.4, random_state=42, stratify=y_temp
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    y_val = torch.tensor(y_val.values, dtype=torch.long)
    y_test = torch.tensor(y_test.values, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)  # SHUFFLE FALSE LEBO VALIDACNE
    test_dl = DataLoader(test_ds, batch_size=32)  # SHUFLLE FALSE LEBO TESTOVACIE

    model = MLP()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(50):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                preds = model(xb)
                predicted = torch.argmax(preds, dim=1)
                correct += (predicted == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch + 1}, Train Loss: {loss.item():.4f}, Val Acc: {val_acc:.2f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_dl:
            preds = model(xb)
            predicted = torch.argmax(preds, dim=1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)

    test_acc = correct / total
    print(f"\nFinal Test Accuracy: {test_acc:.2f}")
    drawConfusionMatrix(model, test_dl, title="Test Set")
    drawConfusionMatrix(model, train_dl, title="Train Set")