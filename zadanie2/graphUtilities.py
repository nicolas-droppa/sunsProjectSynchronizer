import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.tree import plot_tree


def showPredictionComparison(y_true, y_pred):
    """
    Shows prediction comparison plot

    Args:
        y_true: real values
        y_pred: predicted values
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color="skyblue", edgecolor="black")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel("Real values")
    plt.ylabel("Predicted values")
    plt.title("Real vs Predicted Comparison")
    plt.tight_layout()
    plt.show()


def showDecisionTree(clf, feature_names=None):
    """
    Draws a decision tree for a classifier.

    Args:
        clf: Fitted DecisionTree model
    """
    plt.figure(figsize=(16, 10))
    plot_tree(
        clf,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        proportion=True,
        impurity=True
    )
    plt.title("Decision Tree Regression")
    plt.tight_layout()
    plt.show()


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
        if dataFrame[column].dtype == 'bool':
            data = dataFrame[column].astype(int)  # True -> 1, False -> 0
        else:
            data = dataFrame[column]


        plt.hist(data, bins=bins, color="skyblue", edgecolor="black")
        plt.title(f"Histogram for {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()


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
    sns.boxplot(x=column1, y=column2, data=data, color="lightblue")
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