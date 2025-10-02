"""Module pour la préparation des données NBA."""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def prepare_data(file_path):
    """
    Prépare les données pour le modèle à partir du fichier CSV.

    Args:
        file_path (str): Chemin vers le fichier CSV.

    Returns:
        tuple: (x_data, y_data, encoder)
    """
    df = pd.read_csv(file_path)

    cols_to_drop = [
        "Unnamed: 0",
        "Rk",
        "TEAM",
        "AGE",
        "ORB",
        "DRB",
        "PACE",
        "3P%",
        "2P%",
        "eFG%",
        "FT%",
        "FG%",
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore").dropna()

    encoder = LabelEncoder()
    df["POSITION"] = encoder.fit_transform(df["POSITION"])

    x_data = df.drop(columns=["PLAYER", "SALARY_MILLIONS"])
    y_data = df["SALARY_MILLIONS"]

    return x_data, y_data, encoder
