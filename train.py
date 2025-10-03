"""
Module d'entraînement du modèle NBA.
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from numpy import sqrt


def train_model(x_scaled, y):
    """
    Entraîne un modèle RandomForest sur les données NBA.
    Retourne le modèle et ses métriques.
    """
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y)

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
    return model, rmse, mae, r2, (x_train, x_test, y_train, y_test)
