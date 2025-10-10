from dataUtilities import *
from grapthUtilities import *
from utilities import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

if __name__ == '__main__':
    data = loadDataset("zadanie1-data.csv", showInfo=False)

    if data is None:
        print("\nData not loaded, exiting...")
        exit(1)

    showDatasetOverview(data, showInfo=True)
    showCorrelationMatrix(data, showInfo=True)

    #showDependencyGraph(data, "job", "duration", agg="mean")
    #showBoxRelation(data, "emp.var.rate", "euribor3m")
    #showScatterRelation(data, "emp.var.rate", "nr.employed", hue="subscribed")
    #showHeatmapRelation(data, "emp.var.rate", "cons.price.idx")
    #showLineRelation(data, "emp.var.rate", "pdays", agg="mean")

    numericColumns = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx',
                      'cons.conf.idx', 'euribor3m', 'nr.employed']
    showBoxplotWrapper(data, numericColumns, showInfo=False)

    categoricalColumns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                          'day_of_week', 'poutcome', 'subscribed']
    showCategoricalWrapper(data, categoricalColumns, showInfo=False)

    showHistogramWrapper(data, numericColumns, showInfo=False)

    data = removeColumn(data, "duration", showInfo=False)
    data = removeColumn(data, "nr.employed", showInfo=False)
    data = dropColumnsWithTooManyNaN(data, threshold=0.25, showInfo=False)
    data = removeOutliersWrapper(data, showInfo=False)
    #data = removeOutliersIQRWrapper(data, numericColumns, showInfo=False)
    data = removeOutliersIQR(data, "cons.price.idx", showInfo=False)

    x, y = preprocessDataset(data, showInfo=False)

    # # # histograms after scaling
    # currentColumns = x.columns

    scaler = StandardScaler()
    X = scaler.fit_transform(x)

    #showCorrelationMatrix(data, showInfo=True)

    # # # histograms after scaling
    # X_df = pd.DataFrame(X, columns=currentColumns)
    # showHistogramWrapper(X_df, currentColumns, showInfo=True)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.4, random_state=42, stratify=y_temp
    )

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    # model = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight='balanced')
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Validation Accuracy: {val_acc:.2f}")

    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    print(f"Training Accuracy: {train_acc:.2f}")

    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"Test Accuracy: {test_acc:.2f}")

    drawConfusionMatrix(y_test, test_preds, title="Test Set")
    drawConfusionMatrix(y_train, train_preds, title="Train Set")
