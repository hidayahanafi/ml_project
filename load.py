"""
Module de chargement du modèle NBA.
"""

import joblib


def load_model(model_path, scaler_path=None, encoder_path=None):
    """
    Charge le modèle et éventuellement le scaler et l'encodeur.
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path else None
    encoder = joblib.load(encoder_path) if encoder_path else None
    return model, scaler, encoder
