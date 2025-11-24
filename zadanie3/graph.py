import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def showCorrelationMatrix(dataFrame, showInfo=False):
    """
    Prints correlation matrix for numeric columns.
    Args:
        dataFrame (pd.DataFrame): Dataset
        showInfo (bool): Whether to display the matrix
    """
    if showInfo:
        corr = dataFrame.corr(numeric_only=True)
        cmap = LinearSegmentedColormap.from_list("matrix_corr", ["red", "white", "green"])

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=-1,
            vmax=1,
            center=0,
            linewidths=0.5,
            cbar_kws={"label": "Correlation Matrix"}
        )
        plt.title("Correlation Matrix for Numeric Columns")
        plt.tight_layout()
        plt.show()

def showTrainingCurves(train_losses, val_losses, train_metrics, val_metrics):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 5))

    # loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Loss (train vs val)")
    plt.legend()
    plt.grid(True)

    # R2
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_metrics["R2"], label="Train R2")
    plt.plot(epochs, val_metrics["R2"], label="Val R2")
    plt.xlabel("Epoch")
    plt.ylabel("R2")
    plt.title("R2 (train vs val)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def showResiduals(y_true, y_pred, title="Residuals"):
    resid = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, resid, s=8, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (True - Pred)")
    plt.title(title)
    plt.grid(True)
    plt.show()