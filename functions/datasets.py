"""Dataset loaders for anomaly detection benchmarks."""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Seconds before a dataset download request gives up (avoids hanging forever).
_TIMEOUT = 30
_HTTP_OK = 200


def load_dateset(
    file_path: str,
    url: str,
    *,
    save: bool = False,
) -> np.ndarray:
    """Load a numeric dataset from a local file or download it from a URL.

    Args:
        file_path: Path to the local file. Read directly if it exists.
        url: Remote URL to download the dataset from when the file is absent.
        save: Persist the downloaded data to ``file_path`` when True.

    Returns:
        One-dimensional array of floats parsed from the dataset.

    Raises:
        ValueError: If the HTTP response status code is not 200.
    """
    # Check if the file exists
    if Path(file_path).exists():
        # Read the data from the file into a numpy array
        return np.loadtxt(file_path)

    # If the file does not exist, download the data
    response = requests.get(url, timeout=_TIMEOUT)

    if response.status_code == _HTTP_OK:
        lines = response.text.split("\n")
        data = np.array(
            [float(line.strip()) for line in lines if line.strip()],
        )
    else:
        msg = (
            f"Error {response.status_code} while downloading the nprs44. "
            f"Check connection or download and store manually from {url} "
            f"to {file_path}"
        )
        raise ValueError(
            msg,
        )

    if save:
        # Check if the directory exists, if not create it
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving dataset to %s", file_path)
        # Save the data to file_path
        np.savetxt(file_path, data)
    return data


def load_nprs43() -> np.ndarray:
    """Load the NPRS43 time-series benchmark dataset.

    Returns:
        One-dimensional array containing the NPRS43 signal.
    """
    return load_dateset(
        "data/nprs/nprs43.txt",
        "https://www.cs.ucr.edu/~eamonn/discords/nprs43.txt",
        save=True,
    )


def load_nprs44() -> np.ndarray:
    """Load the NPRS44 time-series benchmark dataset.

    Returns:
        One-dimensional array containing the NPRS44 signal.
    """
    return load_dateset(
        "data/nprs/nprs44.txt",
        "https://www.cs.ucr.edu/~eamonn/discords/nprs44.txt",
        save=True,
    )


def load_cats(resample_s: None | int = None) -> pd.DataFrame:
    """Load the CATS multivariate time-series dataset.

    Downloads and caches the dataset as a CSV on first call.

    Args:
        resample_s: Resample period in seconds. No resampling when None.

    Returns:
        DataFrame with a DatetimeIndex and one column per sensor.
    """
    file_path: str = "data/cats/data.csv"
    url = "https://zenodo.org/records/7646897/files/data.parquet"

    if Path(file_path).exists():
        # Read the data from the file into a numpy array
        df = pd.read_csv(file_path, index_col=0)
    else:

        def download_and_read_parquet_with_progress(url: str) -> pd.DataFrame:
            """Download and cache a Parquet file from the given URL.

            Args:
                url: The URL of the Parquet file to download and read.

            Returns:
                DataFrame containing the data from the Parquet file.
            """
            from io import BytesIO

            response = requests.get(url, stream=True, timeout=_TIMEOUT)
            total_size = int(response.headers.get("content-length", 0))
            bytes_downloaded = 0

            buffer = BytesIO()
            for data in response.iter_content(chunk_size=1048576):
                buffer.write(data)
                bytes_downloaded += len(data)
                progress = bytes_downloaded / total_size * 100
                logger.debug(
                    "Downloaded %d/%d bytes (%.2f%%)",
                    bytes_downloaded,
                    total_size,
                    progress,
                )

            # Reset buffer position to the beginning before reading
            buffer.seek(0)

            # Read the Parquet file from the buffer into a pandas DataFrame
            return pd.read_parquet(buffer)

        df = download_and_read_parquet_with_progress(url)

        # Check if the directory exists, if not create it
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    df.index = pd.to_datetime(df.index)
    if resample_s is not None:
        df = df.resample(f"{resample_s}s").median().iloc[resample_s:]

    if not Path(file_path).exists():
        logger.info("Saving dataset to %s", file_path)
        # Save the data to file_path
        df.to_csv(file_path)

    return df


def load_skab(file_path: str = "data/skab") -> dict[str, list[pd.DataFrame]]:
    """Load the SKAB benchmark dataset from a local directory or GitHub.

    Downloads all CSV files from the SKAB repository on first call and
    organises them by sub-folder name.

    Args:
        file_path: Root directory where SKAB data is stored or will be saved.

    Returns:
        Mapping from sub-folder name to a list of DataFrames, one per CSV file.
    """
    from urllib.parse import urlparse

    url = "https://api.github.com/repos/waico/SKAB/contents/data"

    if not Path(file_path).exists():

        def download_csv_from_git(
            url: str,
            save_path: str,
            *,
            add_base: bool = True,
        ) -> None:
            # Parse the URL to get the folder name
            parsed_url = urlparse(url)
            folder_name = Path(parsed_url.path).name

            # Create the folder if it doesn't exist
            if add_base:
                folder_path = str(Path(save_path) / folder_name)
            else:
                folder_path = save_path
            Path(folder_path).mkdir(parents=True, exist_ok=True)

            # Get the contents of the folder
            response = requests.get(url, timeout=_TIMEOUT)
            if response.status_code == _HTTP_OK:
                for item in response.json():
                    if item["type"] == "file" and item["name"].endswith(
                        ".csv",
                    ):
                        logger.info("Downloading %s", item["name"])
                        file_url = item["download_url"]
                        file_name = Path(file_url).name
                        file_path = str(Path(folder_path) / file_name)
                        with Path(file_path).open("wb") as f:
                            f.write(
                                requests.get(
                                    file_url,
                                    timeout=_TIMEOUT,
                                ).content,
                            )
                    elif item["type"] == "dir":
                        download_csv_from_git(item["url"], folder_path)

        download_csv_from_git(url, file_path, add_base=False)

    # Recursively go through directories in file_path
    data_dict: dict[str, list] = {}
    for root, _, files in os.walk(file_path):
        # Create a dictionary to store the data frames
        relative_path = os.path.relpath(root, file_path)
        if relative_path != ".":
            data_dict[relative_path] = []
            for file in files:
                if file.endswith(".csv"):
                    # Get the relative path of the file
                    # Create the corresponding directory structure in the dictionary
                    df = pd.read_csv(
                        Path(root) / file,
                        index_col=0,
                        sep=";",
                    )
                    # Store the data frame in the dictionary
                    data_dict[relative_path].append(df)
    # Return the data dictionary
    return data_dict


def load_usp(
    file_path: str = "data/usp-stream-data",
) -> dict[str, pd.DataFrame]:
    """Load the USP stream benchmark dataset from ARFF files.

    Reads all ``.arff`` files found under ``file_path`` and converts them to
    numeric DataFrames with a unified ``class`` column.

    Args:
        file_path: Root directory containing the extracted USP dataset.

    Returns:
        Mapping from dataset name to the corresponding numeric DataFrame.

    Raises:
        NotImplementedError: If the dataset directory does not exist (automatic
            download is not implemented; the user must download it manually).
    """
    from scipy.io.arff import loadarff
    from tqdm import tqdm

    url = (
        "http://sites.labic.icmc.usp.br/vsouza/repository/usp-stream-data.zip"
    )

    if not Path(file_path).exists():
        response = requests.get(url, timeout=_TIMEOUT)
        if response.status_code == _HTTP_OK:
            msg = (
                f"Please, download the data from the following URL: {url}.\n"
                "Feel free to contribute by implementing the download process."
            )
            raise NotImplementedError(
                msg,
            )

    def convert_dtypes_numeric(df: pd.DataFrame) -> pd.DataFrame:
        for col in df:
            df[col] = df[col].map(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x,
            )
            # We are only interested in ordinal numeric values
            df[col] = df[col].map(
                lambda x: (
                    float(x) if isinstance(x, str) and x.isnumeric() else x
                ),
            )
        return df

    # Recursively go through directories in file_path
    data_dict: dict[str, pd.DataFrame] = {}
    for root, _, files in os.walk(file_path):
        # Create a dictionary to store the data frames
        if root != ".":
            with tqdm(total=len(files), mininterval=1.0) as pbar:
                for file in files:
                    if file.endswith(".arff"):
                        pbar.set_description(f"Loading {file}")
                        # Get the relative path of the file
                        # Create the corresponding directory structure in the dictionary
                        raw_data, meta = loadarff(Path(root) / file)
                        df = pd.DataFrame(raw_data, columns=meta.names())
                        # Store the data frame in the dictionary
                        data_dict[file.split(".")[0]] = df
                    pbar.update(1)

    # Rename the class column to "class" for consistency
    data_dict["chess"] = data_dict["chess"].rename(
        columns={"outcome": "class"},
    )
    data_dict["airlines"] = data_dict["airlines"].rename(
        columns={"Delay": "class"},
    )
    data_dict["gassensor"] = data_dict["gassensor"].rename(
        columns={"Class": "class"},
    )
    data_dict["ozone"] = data_dict["ozone"].rename(columns={"Class": "class"})

    # Convert the data types to numeric
    with tqdm(total=len(files), mininterval=1.0) as pbar:
        for k, df in data_dict.items():
            pbar.set_description(f"Processing {k}")
            df_ = convert_dtypes_numeric(df)
            gt = df_["class"].copy(deep=True)
            df_ = df_.select_dtypes(include="number")
            if "class" not in df_.columns:
                if any(gt.apply(lambda x: isinstance(x, str))):
                    df_["class"] = gt.astype("category").cat.codes
                else:
                    df_["class"] = gt
            df_.index = pd.to_datetime(df_.index, unit="s")
            data_dict[k] = df_
            pbar.update(1)
    return data_dict


def load_bess() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the KOKAM BESS normalised dataset and ground-truth labels.

    Downloads the CSV files from GitHub on first call and caches them locally.

    Returns:
        Tuple of (features DataFrame, ground-truth DataFrame), both indexed by
        datetime.
    """
    folder_path: str = "data/kokam"
    X_path: str = f"{folder_path}/kokam_norm.csv"
    y_path: str = f"{folder_path}/kokam_ground_truth.csv"
    base_url: str = "https://raw.githubusercontent.com/MarekWadinger/adaptive-interpretable-ad/main/examples/"

    for path in [X_path, y_path]:
        url = base_url + path
        if not Path(path).exists():
            Path(folder_path).mkdir(parents=True, exist_ok=True)

            # Read the data from the file into a numpy array
            def download_csv_from_git(url: str, save_path: str) -> None:
                # Get the contents of the folder
                response = requests.get(url, timeout=_TIMEOUT)
                if response.status_code == _HTTP_OK:
                    with Path(save_path).open("wb") as f:
                        f.write(response.content)

            download_csv_from_git(url, path)

    X = pd.read_csv(X_path, index_col=0)
    X.index = pd.to_datetime(X.index)
    y = pd.read_csv(y_path, index_col=0)
    y.index = pd.to_datetime(y.index)
    return X, y
