import pandas as pd
import numpy as np
from ml_project.train import train_model
from ml_project.evaluate import evaluate_model


def test_evaluate_model():

    X = pd.DataFrame({"FEATURE1": [1, 2, 3, 4], "FEATURE2": [4, 3, 2, 1]})
    y = pd.Series([10, 20, 30, 40])

    model, scaler, X_train_scaled, X_test_scaled, y_train, y_test = train_model(
        X, y, test_size=0.5, random_state=1
    )

    evaluate_model(model, X_test_scaled, y_test, X.columns)
