# main.py
from data import *
from graph import showCorrelationMatrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import torch
from train import train_model

IMG_FOLDER = "dataCombined/originals"

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    processSolarData("data", rebuildCombinedData=False, showInfo=False)
    data = loadDataset("dataCombined/out_data.csv", False)
    if data is None:
        print("Data not loaded, exiting...")
        exit(1)

    columnsToDrop = ["SunLatitude", "SunLongitude", "PressureTemp", "HumidityTemp",
                     "BodyTemperatureAvg", "SunAzimuth", "SunZenith"]
    columnsToDrop = [col for col in columnsToDrop if col in data.columns]
    data = data.drop(columns=columnsToDrop)

    X = [os.path.join(IMG_FOLDER, fname) for fname in data["PictureName"].values]
    Y = data["Irradiance"].values

    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    model, scaler = train_model(X_train, Y_train, X_val, Y_val, batch_size=16, epochs=50, device=device)
