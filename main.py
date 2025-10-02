"""
Pipeline complet NBA
"""

import argparse
import pandas as pd
import joblib
from prepare import prepare_data
from train import train_model
from evaluate import evaluate_model
from save import save_model

DATA_PATH = "nba_2017_nba_players_with_salary.csv"
PREPARED_DATA_FILE = "prepared_data.pkl"


def main():
    """
    Exécute les différentes phases du pipeline
    """
    parser = argparse.ArgumentParser(description="Pipeline NBA RandomForest")
    parser.add_argument(
        "phase", type=str, choices=["prepare", "train", "evaluate", "save", "all"]
    )
    args = parser.parse_args()

    if args.phase in {"prepare", "all"}:
        print("--- Étape 1 : Préparation des données ---")
        x_scaled, y, scaler, encoder = prepare_data(DATA_PATH)
        pd.to_pickle((x_scaled, y, scaler, encoder), PREPARED_DATA_FILE)
        print("Données préparées et sauvegardées.")

    if args.phase in {"train", "all"}:
        print("--- Étape 2 : Entraînement du modèle ---")
        x_scaled, y, scaler, encoder = pd.read_pickle(PREPARED_DATA_FILE)
        model, _, _, _, _ = train_model(x_scaled, y)
        joblib.dump(model, "nba_salary_model.joblib")
        joblib.dump(scaler, "scaler.joblib")
        joblib.dump(encoder, "encoder.joblib")
        print("Modèle, scaler et encodeur sauvegardés.")

    if args.phase in {"evaluate", "all"}:
        print("--- Étape 3 : Évaluation ---")
        model = joblib.load("nba_salary_model.joblib")
        scaler = joblib.load("scaler.joblib")
        encoder = joblib.load("encoder.joblib")
        x_scaled, y, _, _ = pd.read_pickle(PREPARED_DATA_FILE)
        evaluate_model(model, x_scaled, y)

    if args.phase == "save":
        print("--- Étape 4 : Sauvegarde finale ---")
        model = joblib.load("nba_salary_model.joblib")
        scaler = joblib.load("scaler.joblib")
        encoder = joblib.load("encoder.joblib")
        save_model(model, scaler, encoder)
        print("Pipeline complet sauvegardé.")


if __name__ == "__main__":
    main()
