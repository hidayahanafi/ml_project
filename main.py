"""Module principal pour exécuter le pipeline de modélisation NBA par phases."""

import os
import argparse
import joblib
import pandas as pd
from prepare import prepare_data
from train import train_model
from evaluate import evaluate_model
from save import save_model

DATA_PATH = "nba_2017_nba_players_with_salary.csv"
MODEL_FILENAME = "nba_salary_model.joblib"
SCALER_FILENAME = "scaler.joblib"
PREPARED_DATA_FILE = "prepared_data.pkl"
TEMP_MODEL_FILE = "temp_model.joblib"
TEMP_X_TEST_FILE = "temp_X_test.joblib"
TEMP_Y_TEST_FILE = "temp_y_test.joblib"


def main():
    """Exécute le pipeline complet du modèle."""
    parser = argparse.ArgumentParser(description="Exécute le pipeline de modélisation")

    parser.add_argument(
        "phase",
        type=str,
        choices=["prepare", "train", "evaluate", "save", "all"],
        help="Phase à exécuter : 'prepare', 'train', 'evaluate', 'save', ou 'all'.",
    )
    args = parser.parse_args()

    # Étape 1 : Préparation des données
    if args.phase in {"prepare", "all"}:
        print("--- Étape 1: Préparation des données ---")
        if not os.path.exists(DATA_PATH):
            print(f"Erreur : Le fichier de données '{DATA_PATH}' est introuvable.")
            return
        x_data, y_data, _ = prepare_data(DATA_PATH)
        pd.to_pickle((x_data, y_data), PREPARED_DATA_FILE)
        print(f"Données préparées et sauvegardées dans {PREPARED_DATA_FILE}")

    # Étape 2 : Entraînement du modèle
    if args.phase in {"train", "all"}:
        print("\n--- Étape 2: Entraînement du modèle ---")
        if not os.path.exists(PREPARED_DATA_FILE):
            print(
                f"Erreur : '{PREPARED_DATA_FILE}' introuvable. "
                "Exécutez la phase 'prepare' d'abord."
            )
            return
        x_data, y_data = pd.read_pickle(PREPARED_DATA_FILE)
        model, scaler, _, x_test_scaled, _, y_test_data = train_model(x_data, y_data)
        joblib.dump(model, TEMP_MODEL_FILE)
        joblib.dump(scaler, SCALER_FILENAME)
        joblib.dump(x_test_scaled, TEMP_X_TEST_FILE)
        joblib.dump(y_test_data, TEMP_Y_TEST_FILE)
        print("Modèle, scaler et données de test sauvegardés temporairement.")

    # Étape 3 : Évaluation du modèle
    if args.phase in {"evaluate", "all"}:
        print("\n--- Étape 3: Évaluation du modèle ---")
        required_files = [
            TEMP_MODEL_FILE,
            TEMP_X_TEST_FILE,
            TEMP_Y_TEST_FILE,
            PREPARED_DATA_FILE,
        ]
        if not all(os.path.exists(f) for f in required_files):
            print(
                "Erreur : fichiers temporaires ou données préparées introuvables. "
                "Exécutez 'prepare' et 'train' d'abord."
            )
            return
        model = joblib.load(TEMP_MODEL_FILE)
        x_test_scaled = joblib.load(TEMP_X_TEST_FILE)
        y_test_data = joblib.load(TEMP_Y_TEST_FILE)
        x_data, _ = pd.read_pickle(PREPARED_DATA_FILE)
        evaluate_model(model, x_test_scaled, y_test_data, x_data.columns)

    # Étape 4 : Sauvegarde du modèle final
    if args.phase in {"save", "all"}:
        print("\n--- Étape 4: Sauvegarde du modèle ---")
        if not os.path.exists(TEMP_MODEL_FILE):
            print(
                "Erreur : modèle temporaire introuvable. Exécutez la phase 'train' d'abord."
            )
            return
        model = joblib.load(TEMP_MODEL_FILE)
        scaler = joblib.load(SCALER_FILENAME)
        save_model(model, scaler, MODEL_FILENAME, SCALER_FILENAME)
        print(f"Modèle final sauvegardé dans {MODEL_FILENAME}")

    if args.phase == "all":
        print("\n--- Pipeline complet terminé ---")


if __name__ == "__main__":
    main()
