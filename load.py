"""Module pour le chargement du modèle NBA."""

import joblib


def load_model(model_path="nba_salary_model.joblib", scaler_path="scaler.joblib"):
    """
    Charge le modèle et le scaler à partir des fichiers .joblib.

    Args:
        model_path (str): Chemin du modèle sauvegardé.
        scaler_path (str): Chemin du scaler sauvegardé.

    Returns:
        tuple: (model, scaler)
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler
