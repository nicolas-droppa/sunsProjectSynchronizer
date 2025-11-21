from dataUtilities import *
from grapthUtilities import *
from utilities import *
from neuralNetwork import MLP

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

if __name__ == '__main__':
    data = loadDataset("zadanie1-data.csv", showInfo=False)

    if data is None:
        print("\nData not loaded, exiting...")
        exit(1)

    showDatasetOverview(data, showInfo=False)
    showCorrelationMatrix(data, showInfo=False)

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
    """
    scaler = StandardScaler()+
    X = scaler.fit_transform(x)

    #showCorrelationMatrix(data, showInfo=True)

    # # # histograms after scaling
    # X_df = pd.DataFrame(X, columns=currentColumns)
    # showHistogramWrapper(X_df, currentColumns, showInfo=True)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y    
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    """

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.1, stratify=y,
                                                        random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp,
                                                    random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_val = scaler.transform(x_val)
    X_test = scaler.transform(x_test)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                       SKLEARN MODEL                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    """
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
    """

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                     PERCEPTRON MODEL                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    y_val = torch.tensor(y_val.values, dtype=torch.long)
    y_test = torch.tensor(y_test.values, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

    batchSize = 32
    train_dl = DataLoader(train_ds, batch_size=batchSize, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batchSize)  # SHUFFLE FALSE LEBO VALIDACNE
    test_dl = DataLoader(test_ds, batch_size=batchSize)  # SHUFLLE FALSE LEBO TESTOVACIE

    model = MLP(X_train.shape[1])

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.008)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False)

    trainLosses = []
    validationLosses = []
    trainAccuracies = []
    validationAccuracies = []

    earlyStoppingPatience = 20
    bestValidationLoss = float("inf")
    epochsWithNoImprovement = 0

    for epoch in range(100):
        model.train()
        totalTrainLoss = 0
        correctTrain, totalTrain = 0, 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            totalTrainLoss += loss.item()

            predicted = torch.argmax(preds, dim=1)
            correctTrain += (predicted == yb).sum().item()
            totalTrain += yb.size(0)

        avgTrainLoss = totalTrainLoss / len(train_dl)
        trainLosses.append(avgTrainLoss)
        trainAccuracies.append(correctTrain / totalTrain)

        model.eval()
        totalValidationLoss = 0
        correctValidation, totalValidation = 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                preds = model(xb)
                loss = criterion(preds, yb)
                totalValidationLoss += loss.item()
                predicted = torch.argmax(preds, dim=1)
                correctValidation += (predicted == yb).sum().item()
                totalValidation += yb.size(0)

        avgValidationLoss = totalValidationLoss / len(val_dl)
        validationLosses.append(avgValidationLoss)
        val_acc = correctValidation / totalValidation
        validationAccuracies.append(val_acc)

        print(f"Epoch {epoch + 1}, Train Loss: {avgTrainLoss:.4f}, Val Acc: {val_acc:.2f}")

        #scheduler.step(avgValidationLoss)

        # --------------------------- #
        #        EARLY STOPPING       #
        # --------------------------- #
        if avgValidationLoss < bestValidationLoss:
            bestValidationLoss = avgValidationLoss
            epochsWithNoImprovement = 0
            bestModelState = model.state_dict()
        else:
            epochsWithNoImprovement = epochsWithNoImprovement + 1
            if epochsWithNoImprovement >= earlyStoppingPatience:
                print(f"\nEarly stopping after {epoch + 1} epochs!")
                model.load_state_dict(bestModelState)
                break

    showTrainingValidationAccuracy(trainAccuracies, validationAccuracies)
    showTrainingValidationLoss(trainLosses, validationLosses)

    model.eval()

    correct_train, total_train = 0, 0
    with torch.no_grad():
        for xb, yb in train_dl:
            preds = model(xb)
            predicted = torch.argmax(preds, dim=1)
            correct_train += (predicted == yb).sum().item()
            total_train += yb.size(0)
    train_acc = correct_train / total_train

    correct, total_test = 0, 0
    with torch.no_grad():
        for xb, yb in test_dl:
            preds = model(xb)
            predicted = torch.argmax(preds, dim=1)
            correct += (predicted == yb).sum().item()
            total_test += yb.size(0)
    test_acc = correct / total_test

    print(f"\nFinal Train Accuracy: {train_acc:.3f}")
    print(f"\nFinal Test Accuracy: {test_acc:.3f}")

    y_test_true, y_test_pred = getPredictionsAndLabels(model, test_dl)
    y_train_true, y_train_pred = getPredictionsAndLabels(model, train_dl)

    drawConfusionMatrix(y_test_true, y_test_pred, title="Test Set")
    drawConfusionMatrix(y_train_true, y_train_pred, title="Train Set")
