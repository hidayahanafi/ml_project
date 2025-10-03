import pandas as pd
from train import train_model
from evaluate import evaluate_model

def test_evaluate_model():
    X = pd.DataFrame({"F1": [1, 2, 3, 4], "F2": [4, 3, 2, 1]})
    y = pd.Series([10, 20, 30, 40])

    model, _, _, _, (X_train, X_test, y_train, y_test) = train_model(X, y)

    # Appelle evaluate_model
    evaluate_model(model, X_test, y_test)

    # Pas de retour, mais si aucun exception levée, test passé
