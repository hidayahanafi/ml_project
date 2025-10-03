import pandas as pd
from train import train_model

def test_train_model():
    X = pd.DataFrame({"F1": [1, 2, 3, 4], "F2": [4, 3, 2, 1]})
    y = pd.Series([10, 20, 30, 40])

    model, rmse, mae, r2, (X_train, X_test, y_train, y_test) = train_model(X, y)

    # Vérifie que le modèle est bien entraîné
    assert model is not None
    # Vérifie que les métriques sont des floats
    assert isinstance(rmse, float)
    assert isinstance(mae, float)
    assert isinstance(r2, float)
