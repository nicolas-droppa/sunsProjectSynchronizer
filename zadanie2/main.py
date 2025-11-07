from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.ensemble import RandomForestRegressor

from sklearn.decomposition import PCA

import numpy as np

from dataUtilities import *
from graphUtilities import *
from utilities import dropColumns

import plotly.express as px

#checkForCuda()

if __name__ == '__main__':
    # <editor-fold desc="Data loading and cleaning">
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

    if 'hour' in data.columns:
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data = dropColumns(data, ['hour'], True)

    if 'weekday' in data.columns:
        data['weekday_sin'] = np.sin(2 * np.pi * data['weekday'] / 7)
        data['weekday_cos'] = np.cos(2 * np.pi * data['weekday'] / 7)
        data = dropColumns(data, ['weekday'], True)

    # </editor-fold>

    # <editor-fold desc="EDA - graphs">
    # showCorrelationMatrix(data, True)

    # for column in data.columns:
    #     showBarChart(data, column, True)
    #     showPieChart(data, column, True)

    # for column in data.columns:
    #     showHistogram(data, column, 50, True)

    # for column in data.columns:
    #     showHistogram(data, column, 50, True)

    """
    showDependencyGraph(data, "hour", "count", agg="mean")
    showBoxRelation(data, "month", "date")
    showScatterRelation(data, "humidity", "windspeed", hue="count")
    showLineRelation(data, "month", "count", agg="mean")
    """
    # </editor-fold>

    """
    data = dropColumns(data, ['weekday', 'weekday_sin', 'weekday_cos', 'holiday', 'workingday', 'weather',
                              'humidity'], True)
    """

    data = dropColumns(data, ['month'], True)
    data = encodeCategorical(data, 20, True)

    data = dropColumns(data, ['weather_sin', 'weather_cos', 'weather_clear', 'weather_cloudy',
                              'weather_heavy rain/snow', 'windspeed'], True)

    X = data.drop(columns=['count'])
    y = data['count']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_std = scaler.fit_transform(X)

    # <editor-fold desc="=== Decision Tree Regressor ===">
    """
    clf = DecisionTreeRegressor(max_depth=13, min_samples_leaf=3, random_state=42)
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

    showResiduals(y_test, y_pred_test)

    showPredictionComparison(y_test, y_pred_test)
    showDecisionTree(clf, feature_names=list(X.columns))
    """
    # </editor-fold>

    # <editor-fold desc="=== Ensemble - BAGGING ( RANDOM FOREST REGRESSOR ) ===">

    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print("=== Train set ===")
    print(f"R2: {r2_score(y_train, y_pred_train):.3f}")
    print(f"RMSE: {rmse_train:.3f}\n")
    print("=== Test set ===")
    print(f"R2: {r2_score(y_test, y_pred_test):.3f}")
    print(f"RMSE: {rmse_test:.3f}\n")

    showResiduals(y_test, y_pred_test)

    importance_df = pd.DataFrame({
        'Feature': X.columns if isinstance(X, pd.DataFrame) else [f"X{i}" for i in range(X.shape[1])],
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("=== Feature Importances ===")
    for i, row in importance_df.iterrows():
        print(f"{row['Feature']:25} {row['Importance']:.6f}")

    plotFeatureImportances(rf_model, feature_names=list(X.columns), topColumnCount=10)

    # </editor-fold>

    # <editor-fold desc="=== Model SVM ===">
    """
    svm_model = SVR(kernel='rbf', C=100, gamma=0.5, epsilon=0.1)

    # trénovanie modelu
    svm_model.fit(X_train, y_train)

    # predikcie
    y_pred_train = svm_model.predict(X_train)
    y_pred_test = svm_model.predict(X_test)

    # vyhodnotenie presnosti
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print("=== Train set ===")
    print(f"R2: {r2_score(y_train, y_pred_train):.3f}")
    print(f"RMSE: {rmse_train:.3f}\n")

    print("=== Test set ===")
    print(f"R2: {r2_score(y_test, y_pred_test):.3f}")
    print(f"RMSE: {rmse_test:.3f}\n")

    # zobrazenie reziduí a porovnania predikcií
    showResiduals(y_test, y_pred_test)
    showPredictionComparison(y_test, y_pred_test)
    """
    # </editor-fold>

    # <editor-fold desc="=== 3D Plot ===">
    """
    fig = px.scatter_3d(
        data,
        x='hour',
        y='weekday',
        z='temperature',
        color='count',
        color_continuous_scale='Viridis',
        title='3D Scatter – Before Dimensionality Reduction',
        opacity=0.8,
        size_max=8
    )

    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        scene=dict(
            xaxis_title='Hour',
            yaxis_title='Weekday',
            zaxis_title='Temperature'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()
    """
    # </editor-fold>

    # <editor-fold desc="=== Dimension reduction ===">
    """
    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X_std)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance explained:", np.sum(pca.explained_variance_ratio_))

    fig = px.scatter_3d(
        x=X_reduced[:, 0],
        y=X_reduced[:, 1],
        z=X_reduced[:, 2],
        color=y,  # y je count
        color_continuous_scale='Viridis',
        title='3D Scatter – After PCA Reduction',
        opacity=0.8
    )
    fig.show()
    """
    # </editor-fold>

    # <editor-fold desc="=== PCA Data reduction ===">
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X_std)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance explained:", np.sum(pca.explained_variance_ratio_))

    fig = px.scatter_3d(
        x=X_reduced[:, 0],
        y=X_reduced[:, 1],
        z=X_reduced[:, 2],
        color=y,  # y je count
        color_continuous_scale='Viridis',
        title='3D Scatter – After PCA Reduction',
        opacity=0.8
    )
    fig.show()