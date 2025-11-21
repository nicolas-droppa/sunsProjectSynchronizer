import torch

def checkForCuda():
    print("Torch version:", torch.__version__)
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")


def dropColumns(dataFrame, columns, showInfo=False):
    """
    Drops columns from a pandas DataFrame
    Args:
        dataFrame: pandas DataFrame
        columns: list of columns
        showInfo (bool): prints debug information
    Returns:
        data (pandas DataFrame): cleaned DataFrame
    """

    data = dataFrame.copy()

    for column in columns:
        if column in data.columns:
            data.drop(columns=[column], inplace=True)
            if showInfo:
                print(f"Column '{column}' was dropped by choice.")
        else:
            if showInfo:
                print(f"Column '{column}' does not exist, skipping...")

    return data
