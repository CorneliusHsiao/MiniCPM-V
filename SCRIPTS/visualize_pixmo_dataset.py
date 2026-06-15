"""
Usage:
    python MiniCPM-V/SCRIPTS/visualize_pixmo_dataset.py --num-samples 9 --save-path /wekafs/ict/hanyuanx/Datasets/pixmo-count/VIS --split train
"""
import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import load_from_disk
from PIL import Image


DEFAULT_DATASET_DIRS = [
    Path("/wekafs/ict/hanyuanx/Datasets/pixmo-count/dataset_with_local_paths"),
    Path("/wekafs/ict/hanyuanx/datasets/pixmo-count/dataset_with_local_paths"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize the offline Pixmo-Count dataset with local images."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Path to dataset_with_local_paths directory (created by download script).",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument(
        "--indices",
        type=str,
        default="",
        help="Comma-separated sample indices to visualize, e.g. '0,12,53'.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6,
        help="How many random samples to visualize when --indices is not provided.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--point-size",
        type=float,
        default=40.0,
        help="Marker size for annotated points.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, opens an interactive window.",
    )
    parser.add_argument(
        "--show-label",
        action="store_true",
        help="Show class label in subplot title.",
    )
    return parser.parse_args()


def resolve_dataset_dir(user_dir):
    if user_dir is not None:
        if not user_dir.exists():
            raise FileNotFoundError(f"Dataset dir not found: {user_dir}")
        return user_dir

    for candidate in DEFAULT_DATASET_DIRS:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(p) for p in DEFAULT_DATASET_DIRS)
    raise FileNotFoundError(
        f"Could not find dataset_with_local_paths. Tried: {tried}. "
        "Please pass --dataset-dir explicitly."
    )


def resolve_local_image_path(example, dataset_dir):
    raw = Path(example["local_image_path"])
    image_name = f'{example["image_sha256"]}.jpg'

    candidates = [
        raw,
        dataset_dir.parent / "images" / image_name,
    ]

    raw_str = str(raw)
    if "/datasets/" in raw_str:
        candidates.append(Path(raw_str.replace("/datasets/", "/Datasets/")))
    if "/Datasets/" in raw_str:
        candidates.append(Path(raw_str.replace("/Datasets/", "/datasets/")))

    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def select_indices(dataset, dataset_dir, indices_arg, num_samples, seed):
    total_rows = len(dataset)
    if indices_arg.strip():
        indices = []
        for token in indices_arg.split(","):
            token = token.strip()
            if not token:
                continue
            idx = int(token)
            if idx < 0 or idx >= total_rows:
                raise IndexError(f"Index {idx} is out of range [0, {total_rows - 1}]")
            indices.append(idx)
        if not indices:
            raise ValueError("No valid indices found in --indices.")
        return indices, True

    sample_size = min(num_samples, total_rows)
    rng = random.Random(seed)

    shuffled = list(range(total_rows))
    rng.shuffle(shuffled)

    selected = []
    for idx in shuffled:
        example = dataset[idx]
        image_path = resolve_local_image_path(example, dataset_dir)
        if image_path.exists():
            selected.append(idx)
            if len(selected) >= sample_size:
                break

    if selected:
        return sorted(selected), True

    # Fallback: if no local image exists, still return random rows for debugging.
    return sorted(rng.sample(range(total_rows), sample_size)), False


def split_has_local_images(dataset, dataset_dir):
    for example in dataset:
        if resolve_local_image_path(example, dataset_dir).exists():
            return True
    return False


def normalize_points(x_coords, y_coords, width, height):
    if not x_coords or not y_coords:
        return x_coords, y_coords

    max_x = max(x_coords)
    max_y = max(y_coords)
    if max_x <= 1.5 and max_y <= 1.5:
        x_coords = [x * width for x in x_coords]
        y_coords = [y * height for y in y_coords]
    return x_coords, y_coords


def main():
    args = parse_args()
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    dataset_dict = load_from_disk(str(dataset_dir))
    active_split = args.split
    dataset = dataset_dict[active_split]
    indices, has_local = select_indices(
        dataset, dataset_dir, args.indices, args.num_samples, args.seed
    )

    if not has_local and not args.indices.strip():
        for candidate in ["validation", "train", "test"]:
            if candidate not in dataset_dict or candidate == active_split:
                continue
            candidate_ds = dataset_dict[candidate]
            if split_has_local_images(candidate_ds, dataset_dir):
                active_split = candidate
                dataset = candidate_ds
                indices, _ = select_indices(
                    dataset, dataset_dir, args.indices, args.num_samples, args.seed
                )
                print(
                    f"[INFO] Split '{args.split}' has no local images. "
                    f"Auto-switched to '{active_split}'."
                )
                break

    cols = min(3, len(indices))
    rows = math.ceil(len(indices) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]

    missing_images = []
    for ax, idx in zip(axes, indices):
        example = dataset[idx]
        image_path = resolve_local_image_path(example, dataset_dir)

        if not image_path.exists():
            missing_images.append((idx, str(image_path)))
            ax.text(0.5, 0.5, f"Missing image\nidx={idx}", ha="center", va="center")
            ax.axis("off")
            continue

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        x_coords = list(example["points"]["x"])
        y_coords = list(example["points"]["y"])
        x_coords, y_coords = normalize_points(x_coords, y_coords, width, height)

        ax.imshow(image)
        if x_coords and y_coords:
            ax.scatter(
                x_coords,
                y_coords,
                s=args.point_size,
                c="red",
                marker="o",
                edgecolors="white",
                linewidths=0.7,
                alpha=0.9,
            )
        title = f"idx={idx}  count={example['count']}  points={len(x_coords)}"
        if args.show_label:
            title += f"  label={example['label']}"
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    for ax in axes[len(indices):]:
        ax.axis("off")

    fig.suptitle(f"Pixmo-Count ({active_split})", fontsize=14)
    fig.tight_layout()

    if missing_images:
        print(f"[WARN] Missing {len(missing_images)} images.")
        for idx, path in missing_images[:10]:
            print(f"  idx={idx}: {path}")

    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_path, dpi=180, bbox_inches="tight")
        print(f"Saved visualization to: {args.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
