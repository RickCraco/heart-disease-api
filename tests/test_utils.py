import pytest
from src.utils import load_data
from src.utils import download_dataset
from unittest.mock import MagicMock, patch



def test_load_data_valid_csv(tmp_path):
    """Load a valid CSV file and return a DataFrame with the expected shape."""
    # tmp_path is a pytest fixture: a temporary folder destroyed after the test
    csv_path = tmp_path / "fake_heart.csv"

    # Write a minimal CSV file to disk for this test only
    csv_path.write_text("Age,Sex\n58,M\n45,F\n")

    df = load_data(csv_path)

    assert len(df) == 2
    assert list(df.columns) == ["Age", "Sex"]


def test_load_data_missing_file(tmp_path):
    """Raise FileNotFoundError when the CSV file does not exist."""
    # Build a path that points to a file we never create
    missing_path = tmp_path / "missing_file.csv"

    # load_data should fail before trying to read the CSV
    with pytest.raises(FileNotFoundError):
        load_data(missing_path)


@patch("src.utils.requests.get")
def test_download_dataset_success(mock_get, tmp_path):
    """
    """
    file_path = tmp_path / "heart_disease.csv"

    mock_response = MagicMock()
    mock_response.content = b"age,sex,target\n63,1,1"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    download_dataset(file_path)

    mock_get.assert_called_once()
    assert file_path.exists()
    assert file_path.read_bytes() == b"age,sex,target\n63,1,1"