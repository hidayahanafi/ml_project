"""
Sauvegarde finale du pipeline NBA.
"""

import joblib


def save_model(model, scaler, encoder):
    """
    Sauvegarde le modèle, le scaler et l'encodeur sur disque.

    Paramètres
    ----------
    model : RandomForestRegressor
    scaler : StandardScaler
    encoder : OneHotEncoder
    """
    joblib.dump(model, "nba_salary_model_final.joblib")
    joblib.dump(scaler, "scaler_final.joblib")
    joblib.dump(encoder, "encoder_final.joblib")
