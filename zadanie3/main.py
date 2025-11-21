from data import *
from graph import showCorrelationMatrix

if __name__ == "__main__":
    # Creates dataCombined/ folder
    # with subfolder originals/
    # and combines all csvs into one -> out_data.csv
    processSolarData("data", rebuildCombinedData=False, showInfo=False)

    counts = getDataCount(showInfo=False)

    data = loadDataset("dataCombined/out_data.csv", False)

    if data is None:
        print("\nData not loaded, exiting...")
        exit(1)

    showDatasetOverview(data, showInfo=False)

    showCorrelationMatrix(data, showInfo=False)

    columnsToDrop = [
        "SunLatitude", "SunLongitude",  # constants
        "PressureTemp", "HumidityTemp", "BodyTemperatureAvg",  # strong corelation
        "SunAzimuth", "SunZenith"  # weak corelation
    ]

    columnsToDrop = [col for col in columnsToDrop if col in data.columns]

    data = data.drop(columns=columnsToDrop)

    showDatasetOverview(data, showInfo=True)