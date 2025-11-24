import torch
import numpy as np
from utils import l1Penalty, computeMetrics

def trainOneEpoch(model, loader, optimizer, criterion, cfg, scaler_amp=None):
    model.train()
    runningLoss = 0.0
    allPreds, allTargets = [], []

    for imgs, targets in loader:
        imgs, targets = imgs.to(cfg.device), targets.to(cfg.device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=cfg.device.startswith("cuda")):
            outputs = model(imgs)
            targets = targets.view(-1,1)
            loss = criterion(outputs, targets)
            if cfg.useL1:
                loss += l1Penalty(model, cfg.l1Lambda, cfg.device)

        if scaler_amp:
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            loss.backward()
            optimizer.step()

        runningLoss += loss.item() * imgs.size(0)
        allPreds.append(outputs.detach().cpu().numpy())
        allTargets.append(targets.detach().cpu().numpy())

    epochLoss = runningLoss / len(loader.dataset)
    allPreds = np.concatenate([p.flatten() for p in allPreds])
    allTargets = np.concatenate([t.flatten() for t in allTargets])
    metrics = computeMetrics(allTargets, allPreds)
    return epochLoss, metrics

def evaluate(model, loader, criterion, cfg):
    model.eval()
    runningLoss = 0.0
    allPreds, allTargets = [], []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(cfg.device), targets.to(cfg.device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            runningLoss += loss.item() * imgs.size(0)
            allPreds.append(outputs.cpu().numpy())
            allTargets.append(targets.cpu().numpy())

    epochLoss = runningLoss / len(loader.dataset)
    allPreds = np.concatenate([p.flatten() for p in allPreds])
    allTargets = np.concatenate([t.flatten() for t in allTargets])
    metrics = computeMetrics(allTargets, allPreds)
    return epochLoss, metrics, allTargets, allPreds
