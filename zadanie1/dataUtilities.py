from allowedData import *

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def loadDataset(csvPath, showInfo=False):
    """
    Loads dataset from the specified CSV file
    Args:
        csvPath (string): path to the CSV file
        showInfo (bool): prints debug information
    Returns:
        data (pandas DataFrame): Loaded dataset
    """
    try:
        loadedData = pd.read_csv(csvPath, sep=';')

        if showInfo:
            print("Loaded data:")
            print(loadedData.head())
            print("\nColumn types:\n", loadedData.dtypes)

        return loadedData
    except FileNotFoundError:
        print(f"Could not find the file: {csvPath}")
        return None


def removeOutliers(loadedData, column, lowerThreshold=None, upperThreshold=None, allowedValues=None,
                   showInfo=False):
    """
    Removes outliers from a column.
    For numeric columns: keep values within lowerThreshold and upperThreshold
    For non-numeric columns: keep values in allowedValues list

    Args:
        loadedData (pd.DataFrame): original data
        column (str): column name to check
        lowerThreshold (float, optional): minimum allowed value
        upperThreshold (float, optional): maximum allowed value
        allowedValues (list, optional): list of allowed values
        showInfo (bool): prints removed values and count

    Returns:
        data (pandas DataFrame): cleaned data
    """
    mask = pd.Series(True, index=loadedData.index)

    if lowerThreshold is not None:
        mask &= loadedData[column] >= lowerThreshold
    if upperThreshold is not None:
        mask &= loadedData[column] <= upperThreshold
    if allowedValues is not None:
        mask &= loadedData[column].isin(allowedValues)

    count_removed = (~mask).sum()

    if showInfo:
        removed_rows = loadedData.index[~mask]
        for i in removed_rows:
            print(f"Removed outlier at row {i}, column '{column}': {loadedData.at[i, column]}")
        print(f"Total outliers removed in column '{column}': {count_removed}")

    cleanedData = loadedData[mask].copy()
    return cleanedData


def preprocessDataset(loadedData, showInfo=False):
    """
    Preprocesses data
    Args:
        loadedData (pd.DataFrame): Loaded data
        showInfo (bool): Print debug info after processing
    Returns:
        x (pd.DataFrame): Processed feature matrix
        y (pd.Series): Processed target vector
    """
    y = loadedData['subscribed'].copy()
    y = y.map({'no': 0, 'yes': 1})

    x = loadedData.drop(columns=['subscribed'])
    #x = loadedData

    columns = x.select_dtypes(include=['object']).columns
    for column in columns:
        le = LabelEncoder()
        x[column] = le.fit_transform(x[column])

    if showInfo:
        print("\nColumn data types:")
        print(x.dtypes)
        print("\nTarget value counts:")
        print(y.value_counts())
        print("\nNaN values after preprocessing:")
        print(x.isna().sum())

    return x, y


def dropColumnsWithTooManyNaN(dataUncleaned, threshold, showInfo=False):
    """
    Removes columns from DataSet that have more than threshold of NaN values.
    Args:
        dataUncleaned (pd.DataFrame): input data
        threshold (float): fraction of NaN above which column will be dropped
        showInfo (bool): whether to print removed columns

    Returns:
        cleaned data
    """
    ColumnDropCount = []

    for col in dataUncleaned.columns:
        nan_count = dataUncleaned[col].isna().sum()
        nan_fraction = nan_count / len(dataUncleaned)

        if nan_fraction > threshold:
            ColumnDropCount.append(col)
            if showInfo:
                print(f"Dropping column '{col}' with {nan_count} NaN values ({nan_fraction:.2%})")

    return dataUncleaned.drop(columns=ColumnDropCount)


def removeOutliersWrapper(dataUncleaned, showInfo=False):
    """
    Wrapper for removeOutliers
    Args:
        dataUncleaned (pd.DataFrame): Uncleaned data
        showInfo (bool): Print debug info after processing
    Returns:
        Cleaned data
    """
    columnsToClean = [
        ('age', {'lowerThreshold': 17, 'upperThreshold': 100}),
        ('job', {'allowedValues': allowedJobs}),
        ('marital', {'allowedValues': allowedMarital}),
        ('education', {'allowedValues': allowedEducation}),
        ('default', {'allowedValues': allowedDefault}),
        ('housing', {'allowedValues': allowedHousing}),
        ('loan', {'allowedValues': allowedLoan}),
        ('contact', {'allowedValues': allowedContact}),
        ('month', {'allowedValues': allowedMonth}),
        ('day_of_week', {'allowedValues': allowedDayOfWeek}),
        ('duration', {'lowerThreshold': 0, 'upperThreshold': float('inf')}),
        ('campaign', {'lowerThreshold': 0, 'upperThreshold': float('inf')}),
        ('pdays', {'lowerThreshold': 0, 'upperThreshold': float('inf')}),
        ('previous', {'lowerThreshold': 0, 'upperThreshold': float('inf')}),
        ('poutcome', {'allowedValues': allowedPoutcome}),
        ('emp.var.rate', {'lowerThreshold': -1000, 'upperThreshold': 1000}),
        ('cons.price.idx', {'lowerThreshold': 0, 'upperThreshold': 1000}),
        ('cons.conf.idx', {'lowerThreshold': -1000, 'upperThreshold': 0}),
        ('euribor3m', {'lowerThreshold': 0, 'upperThreshold': 100}),
        ('nr.employed', {'lowerThreshold': 1000, 'upperThreshold': 100000}),
        ('subscribed', {'allowedValues': allowedSubscribe})
    ]

    for columnName, params in columnsToClean:
        if columnName in dataUncleaned.columns:
            dataUncleaned = removeOutliers(dataUncleaned, columnName, showInfo=showInfo, **params)
        elif showInfo:
            print(f"Column '{columnName}' does not exist, skipping...")

    return dataUncleaned


def removeColumn(dataUncleaned, columnName, showInfo=False):
    """
    Remove a single column from the dataset
    Args:
        dataUncleaned (pd.DataFrame): Uncleaned data
        columnName (str): Name of column to be removed
        showInfo (bool): Print debug info
    Returns:
        pd.DataFrame: Cleaned data
    """
    newData = dataUncleaned.copy()

    if columnName in newData.columns:
        newData.drop(columns=[columnName], inplace=True)
        if showInfo:
            print(f"Column '{columnName}' was dropped by choice.")
    else:
        if showInfo:
            print(f"Column '{columnName}' does not exist, skipping...")

    return newData


def getDataCount(dataToCount, showInfo=False):
    """
    Returns number of rows and columns in dataset
    Args:
        dataToCount (pd.DataFrame): Dataset
        showInfo (bool): Print debug info
    Returns:
        tuple: (num_rows, num_columns)
    """
    rows, cols = dataToCount.shape
    if showInfo:
        print(f"Dataset has [{rows}] rows and [{cols}] columns.")
    return rows, cols


def showDatasetOverview(dataFrame, showInfo=False):
    """
    Prints a concise overview of dataset.
    Args:
        dataFrame (pd.DataFrame): Dataset
    """
    if showInfo:
        print("\nDATASET OVERVIEW")
        print(f"Rows: {dataFrame.shape[0]}, Columns: {dataFrame.shape[1]}")

        print("\nCOLUMNS")
        for i, col in enumerate(dataFrame.columns, start=1):
            print(f"[{i}] {col}")

        print("\nCOLUMN DETAILS")
        overview = []
        for col in dataFrame.columns:
            overview.append({
                "Column": col,
                "Type": dataFrame[col].dtype,
                "Missing": dataFrame[col].isna().sum(),
                "Unique": dataFrame[col].nunique(),
                "Sample": dataFrame[col].iloc[0] if dataFrame.shape[0] > 0 else None
            })

        import pandas as pd
        overview_df = pd.DataFrame(overview)
        print(overview_df.to_string(index=False))

        print("\nBASIC STATS (NUMERIC)")
        print(dataFrame.describe().T[["mean", "std", "min", "max"]])

        print("\nDUPLICATES")
        print(f"Duplicate rows: {dataFrame.duplicated().sum()}")


def removeOutliersIQR(data, column, k=1.5, showInfo=False):
    """
    Removes outliers from column with IQR.
    Args:
        data (pd.DataFrame): Dataset
        column (str): Column to check for outliers
        k (float): Multiplier for IQR (default 1.5)
        showInfo (bool): Prints removed values and count
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in data.columns:
        if showInfo:
            print(f"Column '{column}' does not exist, skipping...")
        return data

    if not np.issubdtype(data[column].dtype, np.number):
        if showInfo:
            print(f"Column '{column}' is not numeric, skipping IQR outlier removal.")
        return data

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR

    mask = (data[column] >= lower_bound) & (data[column] <= upper_bound)
    count_removed = (~mask).sum()

    if showInfo:
        removed_rows = data.index[~mask]
        for i in removed_rows:
            print(f"Removed outlier at row {i}, column '{column}': {data.at[i, column]}")
        print(f"Total outliers removed in column '{column}' using IQR: {count_removed}")

    return data[mask].copy()


def removeOutliersIQRWrapper(data, columns, k=1.5, showInfo=False):
    """
    Wrapper to remove outliers for columns using IQR.
    Args:
        data (pd.DataFrame): Dataset
        columns (list, optional): List of columns to clean
        k (float): Multiplier for IQR (default 1.5)
        showInfo (bool): Print debug info
    Returns:
        pd.DataFrame: Cleaned data
    """
    for col in columns:
        data = removeOutliersIQR(data, col, k=k, showInfo=showInfo)

    return data


def getPredictionsAndLabels(model, dataloader):
    """
    Generates predictions and true labels from model
    Args:
        model (torch.nn.Module): Trained PyTorch model
        dataloader (torch.utils.data.DataLoader): DataLoader providing batches of (inputs, labels)
    Returns:
        tuple: (numpy.ndarray, numpy.ndarray)
            y_true - true labels
            y_pred - predicted labels
    """
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for xb, yb in dataloader:
            preds = model(xb)
            predicted = torch.argmax(preds, dim=1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(yb.numpy())

    return np.array(all_labels), np.array(all_preds)