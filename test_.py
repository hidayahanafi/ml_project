# tests/test_pipeline.py
import os
import pandas as pd
from prepare import prepare_data
from train import train_model
from evaluate import evaluate_model

DATA_PATH = "nba_2017_nba_players_with_salary.csv"

def test_prepare_data():
    X, y, encoder = prepare_data(DATA_PATH)
    # Vérifie que X et y ne sont pas vides
    assert not X.empty
    assert len(X) == len(y)
    # Vérifie que la colonne POSITION a été encodée
    assert "POSITION" in X.columns

def test_train_model():
    X, y, _ = prepare_data(DATA_PATH)
    model, scaler, X_train_scaled, X_test_scaled, y_train, y_test = train_model(X, y)
    # Vérifie que le modèle a été entraîné
    assert model is not None
    assert hasattr(model, "predict")

def test_evaluate_model():
    X, y, _ = prepare_data(DATA_PATH)
    model, scaler, X_train_scaled, X_test_scaled, y_train, y_test = train_model(X, y)
    # Ici, on teste juste que la fonction s'exécute sans erreur
    evaluate_model(model, X_test_scaled, y_test, X.columns)
