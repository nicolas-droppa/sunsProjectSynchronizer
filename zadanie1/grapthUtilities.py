import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap


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


def showHistogram(dataFrame, column, bins=50, showInfo=False):
    """
    Plots a histogram for column.
    Args:
        dataFrame (pd.DataFrame): Dataset
        column (str): Column name
        bins (int): Number of bins
        showInfo (bool): Whether to show debug info
    """
    if showInfo:
        plt.figure(figsize=(8, 4))
        plt.hist(dataFrame[column], bins=bins, color="skyblue", edgecolor="black")
        plt.title(f"Histogram for {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


def showHistogramWrapper(dataFrame, columns=None, bins=50, showInfo=False):
    """
    Draws histograms for specified columns
    Args:
        dataFrame (pd.DataFrame): Dataset
        columns (list or None): List of columns
        bins (int): Number of bins
        showInfo (bool): Whether to show debug info
    """
    if showInfo:
        if columns is None:
            columns = dataFrame.select_dtypes(include=["int64", "float64"]).columns

        for column in columns:
            if column in dataFrame.columns:
                showHistogram(dataFrame, column, bins=bins, showInfo=showInfo)
            else:
                print(f"Column '{column}' does not exist, skipping...")


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
                print(f"Column '{column}' does not exist, skipping...")


def showBarChart(dataFrame, column, showInfo=False):
    """
    Plots a bar chart for column.
    Args:
        dataFrame (pd.DataFrame): Dataset
        column (str): Column name
        showInfo (bool): Whether to display the plot
    """
    if showInfo:
        plt.figure(figsize=(8, 4))
        dataFrame[column].value_counts().plot(kind="bar", color="skyblue", edgecolor="black")
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
        dataFrame[column].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90, colors=plt.cm.Paired.colors)
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
                print(f"Column '{column}' does not exist, skipping...")


def showDependencyGraph(data, column1, column2, agg="mean", showInfo=True):
    """
    Displays barchart showing dependency between a categorical and numerical column.
    """
    grouped = data.groupby(column1)[column2].agg(agg).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=grouped.index, y=grouped.values, color="skyblue")
    plt.xticks(rotation=45)
    plt.title(f"{agg.capitalize()} of {column2} by {column1}")
    plt.xlabel(column1)
    plt.ylabel(f"{agg.capitalize()} of {column2}")
    plt.tight_layout()
    plt.show()

    if showInfo:
        print(grouped)


def showBoxRelation(data, column1, column2):
    """
    Displays boxplot to visualize distribution of numerical values across categories.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=column1, y=column2, data=data, palette="pastel")
    plt.xticks(rotation=45)
    plt.title(f"Distribution of {column2} by {column1}")
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.tight_layout()
    plt.show()


def showScatterRelation(data, column1, column2, hue=None):
    """
    Displays a scatter plot showing relationship between two columns.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=column1, y=column2, data=data, hue=hue, alpha=0.7)
    plt.title(f"Relationship between {column1} and {column2}")
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.tight_layout()
    plt.show()


def showHeatmapRelation(data, column1, column2):
    """
    Displays a heatmap showing cross-tabulation between two columns.
    """
    crosstab = pd.crosstab(data[column1], data[column2])
    plt.figure(figsize=(8, 6))
    sns.heatmap(crosstab, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(f"Relationship between {column1} and {column2}")
    plt.xlabel(column2)
    plt.ylabel(column1)
    plt.tight_layout()
    plt.show()


def showLineRelation(data, column1, column2, agg="mean"):
    """
    Displays a line plot showing how the average of numerical column changes by category or time.
    """
    grouped = data.groupby(column1)[column2].agg(agg).reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=column1, y=column2, data=grouped, marker="o")
    plt.xticks(rotation=45)
    plt.title(f"{agg.capitalize()} {column2} by {column1}")
    plt.xlabel(column1)
    plt.ylabel(f"{agg.capitalize()} of {column2}")
    plt.tight_layout()
    plt.show()


def showTrainingValidationLoss(trainLosses, validationLosses, showInfo=True):
    """
    Plots the train and validation loss graph
    Args:
        trainLosses (list or np.ndarray): List of training losses
        validationLosses (list or np.ndarray): List of validation losses
        showInfo (bool): Whether to display the plot
    """
    if showInfo:
        plt.figure(figsize=(8, 5))
        plt.plot(trainLosses, label="Training Loss", linewidth=1)
        plt.plot(validationLosses, label="Validation Loss", linewidth=1)
        plt.title("Training vs Validation Loss graph")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()


def showTrainingValidationAccuracy(trainAccuracies, validationAccuracies, showInfo=True):
    """
    Plots the train and validation accuracy graph
    Args:
        trainAccuracies (list or np.ndarray): List of training accuracies
        validationAccuracies (list or np.ndarray): List of validation accuracies
        showInfo (bool): Whether to display the plot
    """
    if showInfo:
        plt.figure(figsize=(8, 5))
        plt.plot(trainAccuracies, label="Training Accuracy", linewidth=1)
        plt.plot(validationAccuracies, label="Validation Accuracy", linewidth=1)
        plt.title("Training vs Validation Accuracy graph")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()
