from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

RANK_ORDER = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]


def rank_id(rank: str) -> int:
    try:
        return RANK_ORDER.index(rank)
    except ValueError as e:
        raise ValueError(f"Unknown rank {rank!r}; expected one of {RANK_ORDER}") from e


def parse_rank(class_name: str) -> str:
    """Strip trailing suit letter (c,d,h,s). Ranks are A,2-9,10,J,Q,K."""
    if len(class_name) < 2:
        raise ValueError(f"Invalid class name: {class_name!r}")
    suit = class_name[-1].lower()
    if suit not in "cdhs":
        raise ValueError(f"Expected suit suffix c/d/h/s in {class_name!r}")
    return class_name[:-1]


def build_old_to_new(names: list[str]) -> list[int]:
    return [rank_id(parse_rank(n)) for n in names]


def remap_label_file(path: Path, old_to_new: list[int]) -> None:
    lines_out: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_id = int(parts[0])
            new_id = old_to_new[old_id]
            parts[0] = str(new_id)
            lines_out.append(" ".join(parts))
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines_out))
        if lines_out:
            f.write("\n")


def backup_labels_if_needed(labels_dir: Path, backup_dir: Path) -> None:
    if backup_dir.exists():
        return
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels dir: {labels_dir}")
    shutil.copytree(labels_dir, backup_dir)


def max_label_id_in_split(labels_dir: Path) -> int | None:
    files = list(labels_dir.glob("*.txt"))
    if not files:
        return None
    m = -1
    for txt in files:
        with txt.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = max(m, int(line.split()[0]))
    return m


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remap 52-class suit labels to 13 rank classes for blackjack."
    )
    parser.add_argument(
        "--dataset-dir",
        default="training_data",
        help="Dataset root (contains data.yaml, train/valid/test)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remap even if labels look already rank-only (max class id <= 12). Risky unless restored from labels_suit.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    dataset_dir = (project_root / args.dataset_dir).resolve()
    source_yaml = dataset_dir / "data.yaml"
    if not source_yaml.exists():
        raise FileNotFoundError(f"Not found: {source_yaml}")

    with source_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if not names or not isinstance(names, list):
        raise ValueError("data.yaml must contain a list 'names'")

    old_to_new = build_old_to_new(names)
    if len(set(old_to_new)) != 13:
        raise RuntimeError("Mapping did not produce 13 distinct rank classes; check data.yaml names.")

    for split in ("train", "valid", "test"):
        labels_dir = dataset_dir / split / "labels"
        if not labels_dir.is_dir():
            continue
        mx = max_label_id_in_split(labels_dir)
        if mx is not None and mx <= 12 and not args.force:
            print(
                f"{split}/labels: max class id is {mx} (rank-only datasets use 0–12). "
                "Skip remapping to avoid corrupting labels. Restore from */labels_suit/ if needed, or use --force.",
                file=sys.stderr,
            )
            sys.exit(1)

    for split in ("train", "valid", "test"):
        labels_dir = dataset_dir / split / "labels"
        if not labels_dir.is_dir():
            continue
        backup_dir = dataset_dir / split / "labels_suit"
        backup_labels_if_needed(labels_dir, backup_dir)
        for txt in sorted(labels_dir.glob("*.txt")):
            remap_label_file(txt, old_to_new)

    out_yaml = dataset_dir / "data_blackjack.yaml"
    blackjack = {
        "nc": len(RANK_ORDER),
        "names": RANK_ORDER,
    }
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(blackjack, f, sort_keys=False)

    print(f"Remapped labels under {dataset_dir}/train|valid|test/labels")
    print(f"Backups (first run only): */labels_suit/")
    print(f"Wrote {out_yaml}")
    print("Train with: python train_yolo.py --dataset-dir training_data --data-yaml data_blackjack.yaml ...")


if __name__ == "__main__":
    main()
