import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from graph import showResiduals
from utils import computeMetrics
from sklearn.preprocessing import StandardScaler


def evalImageDataFusion(df_data, df_features):
    # merge features with picture names
    merged = df_data.merge(df_features, on="PictureName")

    numeric_cols = merged.select_dtypes(include=['number']).columns.tolist()

    drop_cols = ["Irradiance", "IrradianceNotCompensated"]
    X_cols = [c for c in numeric_cols if c not in drop_cols]

    X = merged[X_cols].astype("float32")
    y = merged["Irradiance"].astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=25,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_s, y_train)

    y_train_pred = rf.predict(X_train_s)
    y_test_pred = rf.predict(X_test_s)

    train_metrics = computeMetrics(y_train, y_train_pred)
    test_metrics = computeMetrics(y_test, y_test_pred)

    print("\n=== Random Forest Fusion Results ===")
    print("Train:", train_metrics)
    print("Test:", test_metrics)

    showResiduals(y_train, y_train_pred, title="Fusion RF - Train Residuals")
    showResiduals(y_test, y_test_pred, title="Fusion RF - Test Residuals")

    return train_metrics, test_metrics

