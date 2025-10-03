import os
import pandas as pd
from train import train_model
from save import save_model
from load import load_model

def test_save_load_model():
    # Données fictives
    X = pd.DataFrame({"F1": [1, 2], "F2": [3, 4]})
    y = pd.Series([10, 20])

    # Entraînement du modèle
    model, rmse, mae, r2, (X_train, X_test, y_train, y_test) = train_model(X, y)

    # Sauvegarde du modèle, scaler et encodeur (None pour ce test)
    save_model(model, None, None)

    # Chargement du modèle en utilisant le chemin fixé par save_model
    loaded_model, loaded_scaler, loaded_encoder = load_model(
        "nba_salary_model_final.joblib",
        "scaler_final.joblib",
        "encoder_final.joblib"
    )

    # Vérification de la cohérence des prédictions
    pred_original = model.predict(X)
    pred_loaded = loaded_model.predict(X)
    assert all(pred_original == pred_loaded)

    # Nettoyage des fichiers générés
    for f in ["nba_salary_model_final.joblib", "scaler_final.joblib", "encoder_final.joblib"]:
        if os.path.exists(f):
            os.remove(f)
