import os
import shutil
import pandas as pd
from PIL import Image


def listArchives(baseDir):
    """
    Lists all archive folders within the given base directory.
    Args:
        baseDir (str): Path to the base directory (e.g., 'data')
    Returns:
        list: Archive folder names
    """
    return [
        f for f in os.listdir(baseDir)
        if f.startswith("Solar_data-") and os.path.isdir(os.path.join(baseDir, f))
    ]


def getAllPngPaths(archivePath):
    """
    Collects all PNG file paths within 'original' subfolders of the archive.
    Args:
        archivePath (str): Path to one archive (e.g., 'data/Solar_data-...-001')
    Returns:
        list: Absolute paths to all PNG files
    """
    pngPaths = []
    for root, dirs, files in os.walk(archivePath):
        if os.path.basename(root) == "original":
            for file in files:
                if file.lower().endswith(".png"):
                    pngPaths.append(os.path.join(root, file))
    return pngPaths


def getAllCsvPaths(archivePath):
    """
    Collects all out_data.csv file paths within the archive.
    Args:
        archivePath (str): Path to one archive
    Returns:
        list: Absolute paths to CSV files
    """
    csvPaths = []
    for root, dirs, files in os.walk(archivePath):
        if "out_data.csv" in files:
            csvPaths.append(os.path.join(root, "out_data.csv"))
    return csvPaths


def resizeAndCopyPngs(pngPaths, targetDir, size=(128, 128)):
    """
    Resizes and copies PNG files to a target directory, preserving names.
    Args:
        pngPaths (list): List of PNG file paths
        targetDir (str): Destination directory for resized images
        size (tuple): Target size (width, height)
    """
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)

    count = 0
    for src in pngPaths:
        try:
            with Image.open(src) as img:
                resized = img.resize(size)
                dstName = os.path.basename(src)
                dstPath = os.path.join(targetDir, dstName)

                # Handle potential duplicate names
                if os.path.exists(dstPath):
                    base, ext = os.path.splitext(dstName)
                    i = 1
                    while True:
                        newName = f"{base}_{i}{ext}"
                        newPath = os.path.join(targetDir, newName)
                        if not os.path.exists(newPath):
                            dstPath = newPath
                            break
                        i += 1

                resized.save(dstPath)
                count += 1
        except Exception as e:
            print(f"Failed to process {src}: {e}")

    print(f"Processed {count} PNG files into '{targetDir}'.")


def mergeCsvFiles(csvPaths, outputCsv):
    """
    Merges multiple CSV files into a single CSV.
    Args:
        csvPaths (list): List of CSV file paths
        outputCsv (str): Path to output CSV file
    """
    allFrames = []
    for path in csvPaths:
        try:
            df = pd.read_csv(path)
            allFrames.append(df)
        except Exception as e:
            print(f"Failed to read {path}: {e}")

    if allFrames:
        merged = pd.concat(allFrames, ignore_index=True)
        merged.to_csv(outputCsv, index=False)
        print(f"Merged {len(allFrames)} CSV files into '{outputCsv}'.")
    else:
        print("No CSV files were merged.")


def combineArchives(baseDir, combinedDir, archives, overwrite=False):
    """
    Combines all archives into one folder with resized PNGs and merged CSV.
    Args:
        baseDir (str): Path to base directory containing archives
        combinedDir (str): Path to target combined directory
        archives (list): List of archive folder names
        overwrite (bool): If True, clears and rebuilds the combined folder
    """
    if overwrite:
        if os.path.exists(combinedDir):
            shutil.rmtree(combinedDir)
        os.makedirs(combinedDir)
        print(f"Rebuilt combined directory: {combinedDir}")
    else:
        if not os.path.exists(combinedDir):
            os.makedirs(combinedDir)
            print(f"Created combined directory: {combinedDir}")
        else:
            print(f"Using existing combined directory: {combinedDir}")

    pngPaths = []
    csvPaths = []

    for archive in archives:
        archiveSolarPath = os.path.join(baseDir, archive, "Solar_data")
        if not os.path.isdir(archiveSolarPath):
            print(f"Skipping {archive} — missing Solar_data folder")
            continue

        pngPaths.extend(getAllPngPaths(archiveSolarPath))
        csvPaths.extend(getAllCsvPaths(archiveSolarPath))

    originalsDir = os.path.join(combinedDir, "originals")
    outputCsv = os.path.join(combinedDir, "out_data.csv")

    resizeAndCopyPngs(pngPaths, originalsDir, size=(128, 128))
    mergeCsvFiles(csvPaths, outputCsv)


def processSolarData(baseDir="data", rebuildCombinedData=False, showInfo = False):
    """
    Main process that validates archives and optionally combines them.
    Args:
        baseDir (str): Path to the base data directory
        rebuildCombinedData (bool): If True, rebuilds the combined dataset
        showInfo (bool): If True, shows debug info
    """
    combinedDir = os.path.join(os.path.dirname(baseDir), "dataCombined")
    archives = listArchives(baseDir)

    if not archives:
        if showInfo:
            print(f"No archives found in '{baseDir}'.")
        return

    if showInfo:
        print(f"Found {len(archives)} archive(s): {', '.join(archives)}")

    if rebuildCombinedData:
        combineArchives(baseDir, combinedDir, archives, overwrite=True)
    else:
        if showInfo:
            print("Skipping data combination step.")


def getDataCount(basePath="dataCombined", showInfo=False):
    """
    Counts PNG images and rows in CSV files within the combined data folder.
    Args:
        basePath (str): Path to the combined data folder.
        showInfo (bool): If True, prints debug information.
    Returns:
        dict: Dictionary with counts for PNGs and total CSV rows.
    """

    pngCount = 0
    totalCsvRows = 0
    csvFiles = []

    if not os.path.exists(basePath):
        if showInfo:
            print(f"Path '{basePath}' does not exist.")
        return {"png": 0, "csv_rows": 0}

    for root, dirs, files in os.walk(basePath):
        for file in files:
            if file.lower().endswith(".png"):
                pngCount += 1
            elif file.lower().endswith(".csv"):
                csvPath = os.path.join(root, file)
                csvFiles.append(csvPath)
                try:
                    df = pd.read_csv(csvPath)
                    totalCsvRows += len(df)
                except Exception as e:
                    if showInfo:
                        print(f"Could not read '{file}': {e}")

    result = {"png": pngCount, "csv_rows": totalCsvRows}

    if showInfo:
        print(f"  PNG images : {pngCount}")
        print(f"  CSV rows   : {totalCsvRows}\n")

    return result


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
        loadedData = pd.read_csv(csvPath, sep=',')

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
    Prints a concise overview of a dataset.

    Args:
        dataFrame (pd.DataFrame): Dataset to analyze.
        showInfo (bool): If True, prints detailed information.
    """
    if not showInfo:
        return

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
    overview_df = pd.DataFrame(overview)
    print(overview_df.to_string(index=False))

    print("\nBASIC STATS (NUMERIC)")
    numeric_df = dataFrame.select_dtypes(include="number")
    if numeric_df.empty:
        print("No numeric columns to describe.")
    else:
        desc = numeric_df.describe().T
        cols_to_show = [c for c in ["mean", "std", "min", "max"] if c in desc.columns]
        print(desc[cols_to_show])

    print("\nDUPLICATES")
    print(f"Duplicate rows: {dataFrame.duplicated().sum()}")
