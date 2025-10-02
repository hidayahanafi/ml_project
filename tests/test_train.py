import os
import pandas as pd
from prepare import prepare_data


def test_prepare_data():
    # Create a small dummy CSV
    df = pd.DataFrame(
        {"PLAYER": ["A", "B"], "SALARY_MILLIONS": [10, 20], "POSITION": ["G", "C"]}
    )
    test_file = "dummy.csv"
    df.to_csv(test_file, index=False)

    x_data, y_data, encoder = prepare_data(test_file)

    assert "POSITION" in x_data.columns
    assert len(y_data) == 2
    assert set(encoder.classes_) == {"C", "G"}

    os.remove(test_file)
