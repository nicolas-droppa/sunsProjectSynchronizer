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


def showScatterPlot(df, x_col, y_col, showInfo=False):
    """
    Draws a scatter plot for two numeric DataFrame columns.

    Args:
        df (pd.DataFrame): Dataset
        x_col (str): Column for X axis
        y_col (str): Column for Y axis
        showInfo (bool): Whether to show the plot
    """
    if showInfo:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=x_col, y=y_col, s=20, alpha=0.6)
        plt.title(f"Scatter Plot: {x_col} vs {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def showDistribution(df, col, showInfo=False):
    """
    Draws a histogram + KDE distribution plot for one numeric column.

    Args:
        df (pd.DataFrame): Dataset
        col (str): Column to visualize
        showInfo (bool): Whether to show the plot
    """
    if showInfo:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=30, alpha=0.7)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.grid(alpha=0.3)
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