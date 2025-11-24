import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------- Seed ----------
def setSeed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------- L1 penalty ----------
def l1Penalty(model, l1Lambda: float, device: str):
    l1 = torch.tensor(0.0, device=device)
    for p in model.parameters():
        l1 += torch.sum(torch.abs(p))
    return l1Lambda * l1


# ---------- Metrics ----------
def computeMetrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2}


def saveResultsTable(results, csvPath):
    df = pd.DataFrame(results)
    df.to_csv(csvPath, index=False)
    print(f"Saved experiments summary to '{csvPath}'")
