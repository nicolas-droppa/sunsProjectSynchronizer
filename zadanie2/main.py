from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

import numpy as np

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
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # DECISION TREE REGRESSOR
    """
    clf = DecisionTreeRegressor(max_depth=3, min_samples_leaf=2, random_state=42)
    clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print("=== Train set ===")
    print(f"R2: {r2_score(y_train, y_pred_train):.3f}")
    print(f"RMSE: {rmse_train:.3f}\n")
    print("=== Test set ===")
    print(f"R2: {r2_score(y_test, y_pred_test):.3f}")
    print(f"RMSE: {rmse_test:.3f}\n")

    showPredictionComparison(y_test, y_pred_test)
    showDecisionTree(clf, feature_names=list(X.columns))
    """