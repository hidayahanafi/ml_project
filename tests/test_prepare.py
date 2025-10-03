import os
import pandas as pd
from prepare import prepare_data


def test_prepare_data():
    # Création d'un petit CSV de test
    df = pd.DataFrame(
        {"PLAYER": ["A", "B"], "SALARY_MILLIONS": [10, 20], "POSITION": ["G", "C"]}
    )
    test_file = "dummy.csv"
    df.to_csv(test_file, index=False)

    x_data, y_data, scaler, encoder = prepare_data(test_file)

    # Vérifie que la variable catégorielle POSITION a été encodée
    assert x_data.shape[0] == 2
    assert len(y_data) == 2
    # Vérifie que le scaler et l'encodeur sont bien renvoyés
    assert scaler is not None
    assert encoder is not None

    os.remove(test_file)
