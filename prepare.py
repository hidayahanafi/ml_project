"""
Module de préparation des données pour la prédiction NBA.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def prepare_data(data_path):
    """
    Prépare les données pour l'entraînement du modèle NBA.
    - Encodage des variables catégorielles
    - Standardisation des variables numériques
    """
    data = pd.read_csv(data_path)
    y = data["SALARY_MILLIONS"]
    x_df = data.drop(columns=["SALARY_MILLIONS", "PLAYER", "TEAM", "Rk"])

    # Séparation des colonnes numériques et catégorielles
    num_cols = x_df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = x_df.select_dtypes(include=["object"]).columns

    # Encodage des variables catégorielles
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    if len(cat_cols) > 0:
        x_cat = encoder.fit_transform(x_df[cat_cols])
        x_cat_df = pd.DataFrame(x_cat, columns=encoder.get_feature_names_out(cat_cols))
        x_df = pd.concat(
            [x_df[num_cols].reset_index(drop=True), x_cat_df.reset_index(drop=True)],
            axis=1,
        )
    else:
        x_df = x_df[num_cols]

    # Standardisation
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_df)

    return x_scaled, y, scaler, encoder
