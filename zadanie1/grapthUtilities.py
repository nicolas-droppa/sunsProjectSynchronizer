import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def drawConfusionMatrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Draws a confusion matrix with green for correct predictions
    and red for misclassifications.
    Works with scikit-learn style models.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Build custom green-white-red colormap
    color_matrix = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                color_matrix[i, j] = cm[i, j]
            else:
                color_matrix[i, j] = -cm[i, j]

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("CMColorMap", ["#ff4d4d", "#ffffff", "#4dff4d"])

    im = ax.imshow(color_matrix, interpolation='nearest', cmap=cmap)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of samples")

    classes = ["Not Subscribed", "Subscribed"]
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted values")
    ax.set_ylabel("True values")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, cm[i, j],
                ha='center', va='center',
                color='black', fontsize=12, fontweight='bold'
            )

    plt.tight_layout()
    plt.show()


def plotColumnHistograms(df, bins=50, showInfo=True):
    """
    Plot histograms for all numeric columns to visualize extremes.

    Args:
        df (pd.DataFrame): Dataset
        bins (int): Number of bins in histogram
        showInfo (bool): Whether to print debug info
    """
    numericColumns = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numericColumns:
        plt.figure(figsize=(8, 4))
        plt.hist(df[col], bins=bins, color='skyblue', edgecolor='black')
        plt.title(f"Histogram of '{col}'")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        if showInfo:
            print(f"Plotted histogram for column '{col}'")