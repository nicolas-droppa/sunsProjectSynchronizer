import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torchvision import transforms
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch import amp

IMG_SIZE = (128, 128)
BATCH_SIZE = 4
EPOCHS = 50

def evaluate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, rmse, r2

# -------------------------
# Dataset pre predspracované tensory
# -------------------------
class SolarDataset(Dataset):
    def __init__(self, X, Y):
        """
        X: list of tensors alebo numpy array s obrázkami [N, 3, H, W]
        Y: numpy array s cieľovými hodnotami [N,1]
        """
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.clone().detach()

        y = self.Y[idx]
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        else:
            y = y.clone().detach()

        return x, y.view(-1)

# -------------------------
# CNN Model
# -------------------------
class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2,2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*32*32,128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128,1)

    def forward(self,x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -------------------------
# Funkcia na predspracovanie a uloženie obrázkov
# -------------------------
def preprocess_and_save_images(X_paths, save_path="X_tensors.pt"):
    if os.path.exists(save_path):
        print(f"{save_path} already exists, loading it...")
        torch.load(save_path, weights_only=True)
    print("Processing images...")
    all_imgs = []
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])
    for p in X_paths:
        img = Image.open(p).convert('RGB')
        img = transform(img)
        all_imgs.append(img)
    all_imgs = torch.stack(all_imgs)
    torch.save(all_imgs, save_path)
    return all_imgs

# -------------------------
# Tréning
# -------------------------
def train_model(X_train_paths, Y_train, X_val_paths, Y_val, batch_size=BATCH_SIZE, epochs=EPOCHS, device='cpu'):
    # --- Scaler ---
    scaler = StandardScaler()
    Y_train_scaled = scaler.fit_transform(Y_train.reshape(-1,1))
    Y_val_scaled = scaler.transform(Y_val.reshape(-1,1))

    # --- Predspracovanie obrázkov ---
    X_train = preprocess_and_save_images(X_train_paths, "X_train.pt")  # CPU tensor
    X_val = preprocess_and_save_images(X_val_paths, "X_val.pt")  # CPU tensor

    Y_train_scaled = torch.tensor(Y_train_scaled, dtype=torch.float32)  # CPU tensor
    Y_val_scaled = torch.tensor(Y_val_scaled, dtype=torch.float32)  # CPU tensor

    # --- Dataset a DataLoader ---
    train_dataset = SolarDataset(X_train, Y_train_scaled)
    val_dataset = SolarDataset(X_val, Y_val_scaled)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- Model, loss, optimizer ---
    model = CNNRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scaler_amp = amp.GradScaler()

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            with amp.autocast(device_type='cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, targets)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # --- Save best model ---
        if val_loss < best_val_loss:
            print("Saving model...")
            torch.save(model.state_dict(), f"./model.pt")
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                all_targets.append(targets)
                all_outputs.append(outputs)

        all_targets = torch.cat(all_targets, dim=0)
        all_outputs = torch.cat(all_outputs, dim=0)
        mse, mae, rmse, r2 = evaluate_metrics(all_targets, all_outputs)
        print(f"Validation Metrics -> MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

    return model, scaler
