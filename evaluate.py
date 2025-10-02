"""
Module d'évaluation du modèle NBA.
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from numpy import sqrt


def evaluate_model(model, x_scaled, y):
    """
    Évalue le modèle sur toutes les données.
    """
    y_pred = model.predict(x_scaled)
    rmse = sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"Évaluation finale -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
