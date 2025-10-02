"""Module pour l'évaluation du modèle NBA."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def evaluate_model(model, x_test_scaled, y_test, feature_names):
    """
    Évalue le modèle et affiche les métriques et l'importance des features.

    Args:
        model (RandomForestRegressor): Modèle entraîné.
        x_test_scaled (pd.DataFrame): Données de test normalisées.
        y_test (pd.Series): Cible de test.
        feature_names (pd.Index): Noms des features.
    """
    y_pred = model.predict(x_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    importances = model.feature_importances_
    feat_imp = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x="Importance", y="Feature", data=feat_imp, color="skyblue")
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
