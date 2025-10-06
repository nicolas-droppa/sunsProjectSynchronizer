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

    data = removeColumn(data, "duration", showInfo=True)
    data = dropColumnsWithTooManyNaN(data, threshold=0.25, showInfo=True)
    data = removeOutliersWrapper(data, showInfo=True)

    # plotColumnHistograms(data, bins=50, showInfo=True)

    x, y = preprocessDataset(data, showInfo=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(x)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )

    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    # model = LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced')
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
