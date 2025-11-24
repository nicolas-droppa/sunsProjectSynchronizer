import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

from config import Config
from utils import setSeed, saveResultsTable, computeMetrics
from graph import showTrainingCurves, showResiduals
from data import processSolarData, loadDataset, showDatasetOverview, getDataCount
from dataset import SolarImageDataset
from model import SimpleCNNRegressor
from train import trainOneEpoch, evaluate

def runExperiment(cfg, imgPathsAll, targetsAll, hyperparamsName="default"):
    os.makedirs(cfg.checkpointDir, exist_ok=True)
    print("Device:", cfg.device)

    # train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        imgPathsAll, targetsAll, test_size=(cfg.valRatio + cfg.testRatio), random_state=cfg.seed)
    valFrac = cfg.valRatio / (cfg.valRatio + cfg.testRatio)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - valFrac), random_state=cfg.seed)

    # scale Y
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(np.array(y_train).reshape(-1, 1)).reshape(-1)
    y_val_scaled = scaler.transform(np.array(y_val).reshape(-1, 1)).reshape(-1)
    y_test_scaled = scaler.transform(np.array(y_test).reshape(-1, 1)).reshape(-1)

    transformTrain = transforms.Compose([
        transforms.Resize(cfg.imgSize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transformEval = transforms.Compose([
        transforms.Resize(cfg.imgSize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = SolarImageDataset(X_train, y_train_scaled, imgSize=cfg.imgSize, transform=transformTrain)
    val_ds = SolarImageDataset(X_val, y_val_scaled, imgSize=cfg.imgSize, transform=transformEval)
    test_ds = SolarImageDataset(X_test, y_test_scaled, imgSize=cfg.imgSize, transform=transformEval)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batchSize, shuffle=True, num_workers=cfg.numWorkers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batchSize, shuffle=False, num_workers=cfg.numWorkers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg.batchSize, shuffle=False, num_workers=cfg.numWorkers, pin_memory=True)

    model = SimpleCNNRegressor(useBatchnorm=cfg.useBatchnorm, useDropout=cfg.useDropout, dropoutP=cfg.dropoutP).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learningRate, weight_decay=(cfg.l2Lambda if cfg.useL2 else 0.0))
    criterion = torch.nn.MSELoss()
    scalerAmp = torch.amp.GradScaler(enabled=cfg.device.startswith("cuda"))

    bestValLoss = float("inf")
    bestModelPath = None
    trainLosses, valLosses = [], []
    trainMetricsHist = {"MSE": [], "MAE": [], "RMSE": [], "R2": []}
    valMetricsHist = {"MSE": [], "MAE": [], "RMSE": [], "R2": []}

    for epoch in range(1, cfg.numEpochs + 1):
        trLoss, trMetrics = trainOneEpoch(model, train_loader, optimizer, criterion, cfg, scalerAmp)
        valLoss, valMetrics, _, _ = evaluate(model, val_loader, criterion, cfg)

        trainLosses.append(trLoss)
        valLosses.append(valLoss)
        for k in trainMetricsHist:
            trainMetricsHist[k].append(trMetrics[k])
            valMetricsHist[k].append(valMetrics[k])

        print(f"Epoch {epoch}/{cfg.numEpochs} - Train Loss: {trLoss:.4f} - Val Loss: {valLoss:.4f}")
        print(f"  Train -> {trMetrics}")
        print(f"  Val   -> {valMetrics}")

        # checkpoint best
        if valLoss < bestValLoss:
            bestValLoss = valLoss
            bestModelPath = os.path.join(cfg.checkpointDir, f"best_{hyperparamsName}.pth")
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler": scalerAmp,
                "epoch": epoch
            }, bestModelPath)
            print(f"Saved best model -> {bestModelPath}")

    # load best
    if bestModelPath:
        ckpt = torch.load(bestModelPath, map_location=cfg.device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best model from {bestModelPath}")

    # final evaluation
    _, trainMetricsScaled, trainYTrueS, trainYPredS = evaluate(model, train_loader, criterion, cfg)
    _, testMetricsScaled, testYTrueS, testYPredS = evaluate(model, test_loader, criterion, cfg)

    trainYTrue = scaler.inverse_transform(np.array(trainYTrueS).reshape(-1, 1)).reshape(-1)
    trainYPred = scaler.inverse_transform(np.array(trainYPredS).reshape(-1, 1)).reshape(-1)
    testYTrue = scaler.inverse_transform(np.array(testYTrueS).reshape(-1, 1)).reshape(-1)
    testYPred = scaler.inverse_transform(np.array(testYPredS).reshape(-1, 1)).reshape(-1)

    trainMetricsFinal = computeMetrics(trainYTrue, trainYPred)
    testMetricsFinal = computeMetrics(testYTrue, testYPred)

    # plots
    showTrainingCurves(trainLosses, valLosses, trainMetricsHist, valMetricsHist)
    showResiduals(trainYTrue, trainYPred, title="Train Residuals")
    showResiduals(testYTrue, testYPred, title="Test Residuals")

    summary = {
        "hyperparams": hyperparamsName,
        "train_MSE": trainMetricsFinal["MSE"],
        "train_MAE": trainMetricsFinal["MAE"],
        "train_RMSE": trainMetricsFinal["RMSE"],
        "train_R2": trainMetricsFinal["R2"],
        "test_MSE": testMetricsFinal["MSE"],
        "test_MAE": testMetricsFinal["MAE"],
        "test_RMSE": testMetricsFinal["RMSE"],
        "test_R2": testMetricsFinal["R2"]
    }
    return summary


if __name__ == "__main__":
    cfg = Config()
    setSeed(cfg.seed)
    os.makedirs(cfg.checkpointDir, exist_ok=True)

    processSolarData("data", rebuildCombinedData=False, showInfo=False)
    data = loadDataset("dataCombined/out_data.csv", False)
    if data is None:
        print("Data not loaded, exiting...")
        exit(1)

    showDatasetOverview(data, showInfo=False)
    getDataCount("dataCombined", showInfo=False)

    columnsToDrop = ["SunLatitude", "SunLongitude", "PressureTemp", "HumidityTemp",
                     "BodyTemperatureAvg", "SunAzimuth", "SunZenith"]
    columnsToDrop = [col for col in columnsToDrop if col in data.columns]
    data = data.drop(columns=columnsToDrop)

    imgPathsAll = [os.path.join(cfg.imgFolder, os.path.basename(p)) for p in data["PictureName"].values]
    targetsAll = data["Irradiance"].values.astype(np.float32)

    goodIdx = [i for i, p in enumerate(imgPathsAll) if os.path.exists(p)]
    imgPathsAll = [imgPathsAll[i] for i in goodIdx]
    targetsAll = targetsAll[goodIdx]

    experiments = [
        {"name": "cfgA", "batchSize": 16, "learningRate": 1e-3, "dropoutP": 0.3, "useL1": False, "useL2": True},
        {"name": "cfgB", "batchSize": 32, "learningRate": 5e-4, "dropoutP": 0.4, "useL1": True, "useL2": True},
        {"name": "cfgC", "batchSize": 8, "learningRate": 1e-3, "dropoutP": 0.2, "useL1": False, "useL2": False},
        {"name": "cfgD", "batchSize": 16, "learningRate": 2e-3, "dropoutP": 0.5, "useL1": True, "useL2": False},
        {"name": "cfgE", "batchSize": 24, "learningRate": 8e-4, "dropoutP": 0.35, "useL1": False, "useL2": True},
    ]

    results = []
    for exp in experiments:
        cfg.batchSize = exp["batchSize"]
        cfg.learningRate = exp["learningRate"]
        cfg.dropoutP = exp["dropoutP"]
        cfg.useL1 = exp["useL1"]
        cfg.useL2 = exp["useL2"]

        print("\n====================")
        print("Running experiment:", exp["name"])
        summary = runExperiment(cfg, imgPathsAll, targetsAll, hyperparamsName=exp["name"])
        results.append(summary)

    saveResultsTable(results, cfg.resultsCsv)
    print("All experiments finished.")
