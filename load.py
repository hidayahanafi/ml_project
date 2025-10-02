"""
Chargement des objets sauvegardés du pipeline NBA.
"""

import joblib


def load_model_objects():
    """
    Charge le modèle, le scaler et l'encodeur depuis le disque.

    Returns
    -------
    model, scaler, encoder
    """
    model = joblib.load("nba_salary_model_final.joblib")
    scaler = joblib.load("scaler_final.joblib")
    encoder = joblib.load("encoder_final.joblib")
    return model, scaler, encoder
