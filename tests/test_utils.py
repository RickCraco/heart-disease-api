import pandas as pd
from src.utils import load_data

def test_load_data_valid_csv(tmp_path):

    csv_path = tmp_path / "fake_heart.csv"
    csv_path.write_text("Age,Sex\n58,M\n45,F\n")

    df = load_data(csv_path)

    assert len(df) == 2
    assert list(df.columns) == ["Age", "Sex"]