import os
import pandas as pd
import joblib
from train import train_model
from save import save_model
from load import load_model


def test_save_load_model():
    # Dummy data
    X = pd.DataFrame({"F1": [1, 2], "F2": [3, 4]})
    y = pd.Series([10, 20])

    model, scaler, _, _, _, _ = train_model(X, y, test_size=0.5, random_state=1)

    save_model(
        model, scaler, model_path="test_model.joblib", scaler_path="test_scaler.joblib"
    )

    loaded_model, loaded_scaler = load_model("test_model.joblib", "test_scaler.joblib")

    # Check model prediction consistency
    pred_original = model.predict(X)
    pred_loaded = loaded_model.predict(X)

    assert all(pred_original == pred_loaded)

    os.remove("test_model.joblib")
    os.remove("test_scaler.joblib")
