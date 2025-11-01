from sklearn.model_selection import train_test_split

from dataUtilities import *
from graphUtilities import *
from utilities import dropColumns

#checkForCuda()

if __name__ == '__main__':
    data = loadDataset("z2_data_1y.csv", True)

    if data is None:
        print("\nData not loaded, exiting...")
        exit(1)

    data = dropColumns(data, ['instant'], True)
    data = removeNaN(data, True)
    data = cleanData(data, True)

    # make date appear in correlation matrix
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['date'] = data['date'].map(lambda x: x.toordinal() if pd.notnull(x) else None)

    # showCorrelationMatrix(data, True)

    # for column in data.columns:
    #     showBarChart(data, column, True)
    #     showPieChart(data, column, True)

    # for column in data.columns:
    #     showHistogram(data, column, 50, True)

    data = scaleData(
        data,
        numericColumns=data.select_dtypes(include=['int64', 'float64']).columns,
        showInfo=True
    )

    # for column in data.columns:
    #     showHistogram(data, column, 50, True)

    """
    showDependencyGraph(data, "hour", "count", agg="mean")
    showBoxRelation(data, "month", "date")
    showScatterRelation(data, "humidity", "windspeed", hue="count")
    showLineRelation(data, "month", "count", agg="mean")
    """

    data = dropColumns(data, ['month'], True)
    data = encodeCategorical(data, 20, True)

    X = data.drop(columns=['count'])
    y = data['count']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Shape X_train:", X_train_scaled.shape)
    print("Shape X_test:", X_test_scaled.shape)
    print("Shape y_train:", y_train.shape)
    print("Shape y_test:", y_test.shape)