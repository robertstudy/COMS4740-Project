import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

# python train_yolo.py --device 0 --batch 4 

def load_class_names(data_yaml_path: Path) -> list[str]:
    """Read class names from existing dataset YAML."""
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml_path}")

    with data_yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    if not names or not isinstance(names, list):
        raise ValueError("Could not read class names from dataset data.yaml")
    return names

def write_training_yaml(dataset_dir: Path, names: list[str]) -> Path:
    """Create a stable YAML with absolute paths for YOLO training."""
    train_images = (dataset_dir / "train" / "images").resolve()
    val_images = (dataset_dir / "valid" / "images").resolve()
    test_images = (dataset_dir / "test" / "images").resolve()

    for p in [train_images, val_images]:
        if not p.exists():
            raise FileNotFoundError(f"Required dataset path missing: {p}")

    training_yaml = {
        "train": str(train_images),
        "val": str(val_images),
        "test": str(test_images) if test_images.exists() else str(val_images),
        "nc": len(names),
        "names": names,
    }

    out_path = dataset_dir / "data_local.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(training_yaml, f, sort_keys=False)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO on playing card dataset")
    parser.add_argument("--dataset-dir", default="training_data", help="Dataset folder")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO checkpoint to start from (e.g., yolov8n.pt, yolov8s.pt)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default=None, help="Device: 0, cpu, etc.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    dataset_dir = (project_root / args.dataset_dir).resolve()

    source_yaml = dataset_dir / "data_local.yaml"
    names = load_class_names(source_yaml)
    local_yaml = write_training_yaml(dataset_dir, names)

    print(f"Using dataset yaml: {local_yaml}")
    print(f"Training model: {args.model}")

    model = YOLO(args.model)
    model.train(
        data=str(local_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(project_root / "runs"),
        name="playing_cards_yolo",
        exist_ok=True,
        device=args.device,
        workers=0
    )

    print("Training complete. Check runs/playing_cards_yolo for outputs.")

if __name__ == "__main__":
    main()