import pandas as pd

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