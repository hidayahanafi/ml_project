"""Module pour l'entraînement du modèle NBA."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


def train_model(x_data, y_data, test_size=0.2, random_state=42, n_estimators=200):
    """
    Entraîne un RandomForestRegressor sur les données.

    Args:
        x_data (pd.DataFrame): Variables explicatives.
        y_data (pd.Series): Variable cible.
        test_size (float): Proportion des données de test.
        random_state (int): Seed pour reproductibilité.
        n_estimators (int): Nombre d'arbres.

    Returns:
        tuple: (model, scaler, x_train_scaled, x_test_scaled, y_train, y_test)
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_data.columns)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_data.columns)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train_scaled, y_train)

    return model, scaler, x_train_scaled, x_test_scaled, y_train, y_test
