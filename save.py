"""Module pour la sauvegarde du modèle NBA."""

import joblib


def save_model(
    model, scaler, model_path="nba_salary_model.joblib", scaler_path="scaler.joblib"
):
    """
    Sauvegarde le modèle et le scaler dans des fichiers .joblib.

    Args:
        model: Modèle entraîné.
        scaler: Objet StandardScaler utilisé.
        model_path (str): Chemin de sauvegarde du modèle.
        scaler_path (str): Chemin de sauvegarde du scaler.
    """
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
