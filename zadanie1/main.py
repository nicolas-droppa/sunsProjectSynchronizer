from allowedData import *
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


def checkForCuda():
    """
    Checks for torch version and whether CUDA is available
    """
    print("Torch version:", torch.__version__)
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")


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
        # True: keep, False: removed
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
        ('age', {'lowerThreshold': 18, 'upperThreshold': 100}),
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


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = X_train.shape[1]

        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    # checkForCuda()

    data = loadDataset("zadanie1-data.csv", showInfo=False)

    if data is None:
        print(f"\nData not loaded, exiting...")
        exit(1)

    data = removeColumn(data, "duration", showInfo=True)
    data = dropColumnsWithTooManyNaN(data, threshold=0.25, showInfo=True)
    data = removeOutliersWrapper(data, showInfo=True)

    dataRows, dataColumns = getDataCount(data, showInfo=True)

    #printColumnNames(data, showInfo=True)

    x, y = preprocessDataset(data, showInfo=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(x)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.4, random_state=42, stratify=y_temp
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    y_val = torch.tensor(y_val.values, dtype=torch.long)
    y_test = torch.tensor(y_test.values, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)  # SHUFFLE FALSE LEBO VALIDACNE
    test_dl = DataLoader(test_ds, batch_size=32)  # SHUFLLE FALSE LEBO TESTOVACIE

    model = MLP()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(50):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                preds = model(xb)
                predicted = torch.argmax(preds, dim=1)
                correct += (predicted == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch + 1}, Train Loss: {loss.item():.4f}, Val Acc: {val_acc:.2f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_dl:
            preds = model(xb)
            predicted = torch.argmax(preds, dim=1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)

    test_acc = correct / total
    print(f"\nFinal Test Accuracy: {test_acc:.2f}")

