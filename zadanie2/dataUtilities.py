import pandas as pd
from sklearn.preprocessing import StandardScaler


def encodeCategorical(dataFrame, maxUniqueValues=20, showInfo=True):
    """
    Encodes categorical columns using one-hot encoding for columns with limited unique values.
    Args:
        dataFrame (pd.DataFrame): Input dataset
        maxUniqueValues (int): Maximum number of unique values to encode
        showInfo (bool): If True, prints info about encoding
    Returns:
        pd.DataFrame: Dataset with encoded categorical columns
    """
    cleanedData = dataFrame.copy()
    categoricalCols = cleanedData.select_dtypes(include=['object', 'category']).columns

    for col in categoricalCols:
        numUnique = cleanedData[col].nunique()
        if numUnique <= maxUniqueValues:
            if showInfo:
                print(f"Encoding column '{col}' with {numUnique} unique values")
            dummies = pd.get_dummies(cleanedData[col], prefix=col)
            cleanedData = pd.concat([cleanedData.drop(col, axis=1), dummies], axis=1)
        else:
            if showInfo:
                print(f"Skipping column '{col}' ({numUnique} , has too many unique vals: {maxUniqueValues})")

    return cleanedData


def removeNaN(dataFrame, showInfo=False):
    """
    Removes rows containing NaN values from the dataset.
    Args:
        dataFrame (pd.DataFrame): Dataset
        showInfo (bool): prints info about removed rows
    Returns:
        pd.DataFrame: Dataset without NaN values
    """
    cleanedData = dataFrame.copy()
    before = len(cleanedData)
    cleanedData = cleanedData.dropna()
    after = len(cleanedData)

    if showInfo:
        print(f"Removed {before - after} rows containing NaN values")

    return cleanedData


def scaleData(dataFrame, numericColumns, showInfo=True):
    """
    Scales numeric columns of a DataFrame using StandardScaler.
    Args:
        dataFrame (pd.DataFrame): Input DataFrame
        numericColumns (list): List of numeric columns
        showInfo (bool): If True, prints information about scaling
    Returns:
        pd.DataFrame: DataFrame with scaled numeric columns
    """
    cleanedData = dataFrame.copy()

    if showInfo:
        print(f"Scaling columns: {list(numericColumns)}")

    scaler = StandardScaler()
    cleanedData[numericColumns] = scaler.fit_transform(cleanedData[numericColumns])

    return cleanedData

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


def cleanData(dataFrame, showInfo=False):
    """
    Cleans dataset by enforcing thresholds for numeric columns and filtering allowed weathers.

    Args:
        dataFrame (pd.DataFrame): Dataset
        showInfo (bool): prints debug information

    Returns:
        pd.DataFrame: Cleaned dataset
    """
    allowedWeathers = ['clear', 'cloudy', 'light rain/snow', 'heavy rain/snow']

    columnsToClean = [
        ('month', {'lowerThreshold': 1, 'upperThreshold': 12}),
        ('hour', {'lowerThreshold': 0, 'upperThreshold': 23}),
        ('holiday', {'lowerThreshold': 0, 'upperThreshold': 1}),
        ('weekday', {'lowerThreshold': 0, 'upperThreshold': 6}),
        ('workingday', {'lowerThreshold': 0, 'upperThreshold': 1}),
        ('temperature', {'lowerThreshold': -40, 'upperThreshold': 40}),
        ('humidity', {'lowerThreshold': 0, 'upperThreshold': 100}),
        ('windspeed', {'lowerThreshold': 0, 'upperThreshold': 110}),
        ('count', {'lowerThreshold': 0, 'upperThreshold': float('inf')})
    ]

    cleanedData = dataFrame.copy()

    for columnName, params in columnsToClean:
        if columnName in cleanedData.columns:
            lower = params['lowerThreshold']
            upper = params['upperThreshold']
            before = len(cleanedData)
            cleanedData = cleanedData[(cleanedData[columnName] >= lower) & (cleanedData[columnName] <= upper)]
            after = len(cleanedData)
            if showInfo:
                print(f"Cleaned '{columnName}': removed {before - after} rows outside [{lower}, {upper}]")
        elif showInfo:
            print(f"Column '{columnName}' does not exist, skipping...")

    if 'weather' in cleanedData.columns:
        before = len(cleanedData)
        cleanedData = cleanedData[cleanedData['weather'].isin(allowedWeathers)]
        after = len(cleanedData)
        if showInfo:
            print(f"Filtered 'weather': removed {before - after} rows with disallowed weather types")

    return cleanedData


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