import argparse
import json
import os
from pathlib import Path
import zipfile


def ensure_kaggle_config(project_root: Path) -> None:
    """
    Ensures Kaggle API can find credentials in this project.
    Expects kaggle.json at project root.
    """
    kaggle_json = project_root / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError(
            f"Missing credentials file: {kaggle_json}. Put your kaggle.json in project root."
        )

    # Kaggle API reads credentials from %KAGGLE_CONFIG_DIR%/kaggle.json
    os.environ["KAGGLE_CONFIG_DIR"] = str(project_root)

    # Optional quick validation for helpful error messages
    with kaggle_json.open("r", encoding="utf-8") as f:
        creds = json.load(f)
    if "username" not in creds or "key" not in creds:
        raise ValueError("kaggle.json must contain both 'username' and 'key'.")


def download_and_extract(dataset: str, output_dir: Path) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    output_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset: {dataset}")
    api.dataset_download_files(
        dataset=dataset,
        path=str(output_dir),
        unzip=False,
        quiet=False,
    )

    zip_path = output_dir / f"{dataset.split('/')[-1]}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Download completed but zip file was not found: {zip_path}"
        )

    print(f"Extracting: {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    print(f"Training data is ready in: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract Kaggle training data using local kaggle.json."
    )
    parser.add_argument(
        "--dataset",
        default="andy8744/playing-cards-object-detection-dataset",
        help="Kaggle dataset slug in the form owner/dataset-name.",
    )
    parser.add_argument(
        "--out",
        default="training_data",
        help="Output folder for downloaded and extracted data.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    ensure_kaggle_config(project_root)
    download_and_extract(args.dataset, project_root / args.out)

if __name__ == "__main__":
    main()
