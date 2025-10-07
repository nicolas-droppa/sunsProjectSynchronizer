import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

import seaborn as sns

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


def showCorrelationMatrix(dataFrame, showInfo=False):
    """
    Prints correlation matrix for numeric columns
    Args:
        dataFrame (pd.DataFrame): Dataset
        showInfo (bool): Whether to display the matrix
    """
    if showInfo:
        corr = dataFrame.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Matrix for numeric columns")
        plt.show()


def showBoxplot(dataFrame, column, showInfo=False):
    """
    Plots a boxplot for the specified numeric column.
    Args:
        dataFrame (pd.DataFrame): Dataset
        column (str): Name of column to plot
        showInfo (bool): Whether to display the plot
    """
    if showInfo:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=dataFrame[column])
        plt.title(f"Boxplot for {column}")
        plt.show()


def showBoxplotWrapper(dataFrame, columns, showInfo=False):
    """
    Draws boxplots for selected columns.
    Args:
        dataFrame (pd.DataFrame): Dataset
        columns (list): List of column names
        showInfo (bool): Whether to show debug info
    """
    if showInfo:
        for column in columns:
            if column in dataFrame.columns:
                showBoxplot(dataFrame, column, showInfo)

            else:
                if showInfo:
                    print(f"Column '{column}' does not exist, skipping...")


def showBarChart(dataFrame, column, showInfo=False):
    """
    Plots a bar chart for the specified categorical column.
    Args:
        dataFrame (pd.DataFrame): Dataset
        column (str): Name of column to plot
        showInfo (bool): Whether to display the plot
    """
    if showInfo:
        plt.figure(figsize=(8, 4))
        dataFrame[column].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title(f"Bar Chart for {column}")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()


def showPieChart(dataFrame, column, showInfo=False):
    """
    Plots a pie chart for column.
    Args:
        dataFrame (pd.DataFrame): Dataset
        column (str): Column name
        showInfo (bool): Whether to display the plot
    """
    if showInfo:
        plt.figure(figsize=(6, 6))
        dataFrame[column].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        plt.title(f"Pie Chart for {column}")
        plt.ylabel("")  # Hide default y-label
        plt.tight_layout()
        plt.show()


def showCategoricalWrapper(dataFrame, columns, showInfo=False):
    """
    Draws bar and pie charts for selected columns.
    Args:
        dataFrame (pd.DataFrame): Dataset
        columns (list): List of column names
        showInfo (bool): Whether to show debug info
    """
    if showInfo:
        for column in columns:
            if column in dataFrame.columns:
                showBarChart(dataFrame, column, showInfo)
                showPieChart(dataFrame, column, showInfo)
            else:
                if showInfo:
                    print(f"Column '{column}' does not exist, skipping...")