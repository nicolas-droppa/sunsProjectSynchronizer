import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


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