from src.utils import load_data


def test_load_data_valid_csv(tmp_path):
    """Load a valid CSV file and return a DataFrame with the expected shape."""
    # tmp_path is a pytest fixture: a temporary folder destroyed after the test
    csv_path = tmp_path / "fake_heart.csv"

    # Write a minimal CSV file to disk for this test only
    csv_path.write_text("Age,Sex\n58,M\n45,F\n")

    df = load_data(csv_path)

    assert len(df) == 2
    assert list(df.columns) == ["Age", "Sex"]