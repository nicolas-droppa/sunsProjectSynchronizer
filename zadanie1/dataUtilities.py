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