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


def showResiduals(Y_true, Y_pred, title="Residuals"):
    residuals = Y_true - Y_pred
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.scatter(Y_true, Y_pred)
    plt.xlabel("Skutočné hodnoty")
    plt.ylabel("Predikované hodnoty")
    plt.title(f"Predikované vs Skutočné ({title})")

    plt.subplot(1,2,2)
    plt.scatter(Y_pred, residuals)
    plt.xlabel("Predikované hodnoty")
    plt.ylabel("Reziduály")
    plt.title(f"Reziduály ({title})")

    plt.show()