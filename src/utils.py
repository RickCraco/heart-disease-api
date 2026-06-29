import pathlib
import requests

DATASET_URL = "https://raw.githubusercontent.com/xpy-10/DataSet/refs/heads/main/heart.csv"

def download_dataset():
    """
    Download the project dataset and saves it
    inside the data\ foulder.
    """
    # check if the file already exists
    file_path = pathlib.Path("data/heart_disease.csv")

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