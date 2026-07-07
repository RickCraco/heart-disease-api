import pathlib
import requests
import pandas as pd

DATASET_URL = "https://raw.githubusercontent.com/xpy-10/DataSet/refs/heads/main/heart.csv"
DATASET_PATH = pathlib.Path("data/heart_disease.csv")

def download_dataset(file_path: pathlib.Path = DATASET_PATH):
    """
    Download the heart disease dataset and save it to disk.

    If the target file already exists, the download is skipped. Otherwise,
    creates the parent directory, fetches the CSV from ``DATASET_URL``, and
    writes it to ``file_path``. Network errors are caught and printed without
    raising.

    Args:
        file_path (Path): Destination path for the downloaded CSV file.
            Defaults to ``data/heart_disease.csv``.
    """
    # check if the file already exists
    if file_path.exists():
        print(f"File already exist: {file_path}")
        return

    # create the data folder 
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # try to download file from URL
    try:
        print("Downloading file from URL...")
        request = requests.get(DATASET_URL)
        request.raise_for_status()
        content = request.content

        with open(file_path, "wb") as file:
            file.write(content)
            print("File saved successfully")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file... {e}")


def load_data(file_path: pathlib.Path) -> pd.DataFrame:
    """
    Load the dataset from the given path and return a pandas DataFrame.

    Args:
        file_path (Path): path to the dataset CSV file.

    Returns:
        pd.DataFrame: the loaded dataset.
    """
    # check if the file already exists
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    # create df
    df = pd.read_csv(file_path)

    return df

if __name__ == "__main__":
    download_dataset()