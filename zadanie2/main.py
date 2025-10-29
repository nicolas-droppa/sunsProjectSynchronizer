from dataUtilities import *
from graphUtilities import *
from utilities import dropColumns

#checkForCuda()

if __name__ == '__main__':
    # soonbinuj mesiac a hodniny s date a potom odstran mesiac a hodiny
    data = loadDataset("z2_data_1y.csv", True)

    if data is None:
        print("\nData not loaded, exiting...")
        exit(1)

    showDatasetOverview(data, True)
    data = dropColumns(data, ['instant'], True)
    showBarChart(data, 'month', True)
    showPieChart(data, 'month', True)
    showBarChart(data, 'hour', True)
    showPieChart(data, 'hour', True)
    showBarChart(data, 'holiday', True)
    showPieChart(data, 'holiday', True)
    showBarChart(data, 'weekday', True)
    showPieChart(data, 'weekday', True)
    showBarChart(data, 'workingday', True)
    showPieChart(data, 'workingday', True)
    showBarChart(data, 'temperature', True)
    showPieChart(data, 'temperature', True)
    showBarChart(data, 'humidity', True)
    showPieChart(data, 'humidity', True)
    showBarChart(data, 'windspeed', True)
    showPieChart(data, 'windspeed', True)
    showBarChart(data, 'count', True)
    showPieChart(data, 'count', True)
