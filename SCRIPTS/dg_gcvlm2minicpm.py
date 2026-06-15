#!/usr/bin/env python3

"""
Usage:
    python SCRIPTS/dg_gcvlm2minicpm.py \
        --input-jsonl /wekafs/ict/hanyuanx/dg_vlm/qwen-30b/dg_pairs_noflip.jsonl \
        --output-json /wekafs/ict/hanyuanx/MiniCPM-V/finetune/dg_pairs_minicpm_train.json \
        --needs-human-review false \
        --fraction 1.0
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote


DEFAULT_PROMPT = (
    "<image_00>\n<image_01>\n"
    "Do these two images depict the same real-world place or object? "
    "Answer with exactly one word: yes or no."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert dg_pairs_noflip.jsonl into MiniCPM-V finetune JSON format "
            "for pairwise multi-image classification."
        )
    )
    parser.add_argument("--input-jsonl", required=True, help="Path to input JSONL file.")
    parser.add_argument("--output-json", required=True, help="Path to output train JSON file.")
    parser.add_argument(
        "--eval-output-json",
        default=None,
        help="Optional eval JSON output path (used with --eval-fraction > 0).",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.0,
        help="Fraction of processed samples written to eval split.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of filtered data to keep before optional train/eval split.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of processed samples after filtering/sampling.",
    )
    parser.add_argument(
        "--needs-human-review",
        choices=["all", "true", "false"],
        default="all",
        help="Filter by needs_human_review field.",
    )
    parser.add_argument(
        "--dataset-root",
        default="/wekafs/ict/ict_vgl/hanyuanx/mvdg-dataset/doppelgangers",
        help=(
            "Root of doppelgangers dataset. The converter resolves image paths "
            "under this root and only keeps samples with valid files."
        ),
    )
    parser.add_argument(
        "--dataset-split",
        choices=["auto", "train_set_noflip", "train_set_flip", "test_set"],
        default="auto",
        help="Dataset split under <dataset-root>/images used for relative original_image_* paths.",
    )
    parser.add_argument(
        "--image-source",
        choices=["original", "original_abspath", "auto"],
        default="original",
        help=(
            "How to read pair image paths from each JSONL row: "
            "original_image_* | original_image_*_abspath | fallback automatically."
        ),
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Optional root directory prepended to relative image paths.",
    )
    parser.add_argument(
        "--user-prompt",
        default=DEFAULT_PROMPT,
        help="User prompt template for each training sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling/splitting.",
    )
    parser.add_argument(
        "--id-prefix",
        default="dg_pair",
        help="Prefix for generated sample IDs.",
    )
    parser.add_argument("--strict-image-exists", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    return None


def keep_by_review_flag(record: Dict[str, Any], mode: str) -> bool:
    if mode == "all":
        return True
    flag = to_bool(record.get("needs_human_review"))
    if flag is None:
        return False
    target = mode == "true"
    return flag == target


def normalize_decision(value: Any) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()
    aliases = {
        "yes": "yes",
        "y": "yes",
        "true": "yes",
        "1": "yes",
        "same": "yes",
        "match": "yes",
        "no": "no",
        "n": "no",
        "false": "no",
        "0": "no",
        "different": "no",
        "mismatch": "no",
    }
    return aliases.get(v, v if v else None)


def infer_split(record: Dict[str, Any], override: str) -> str:
    if override != "auto":
        return override
    text = f"{record.get('metadata_file', '')} {record.get('metadata', '')}".lower()
    if "noflip" in text:
        return "train_set_noflip"
    if "flip" in text:
        return "train_set_flip"
    if "test" in text:
        return "test_set"
    return "train_set_noflip"


def normalize_path_text(path_text: str) -> List[str]:
    if not path_text:
        return []
    variants = {path_text, unquote(path_text)}
    return [v for v in variants if v]


def candidate_paths(
    path_text: str,
    dataset_root: Path,
    dataset_split: str,
    image_root: Optional[Path],
) -> List[Path]:
    candidates: List[Path] = []
    for variant in normalize_path_text(path_text):
        p = Path(variant)
        if p.is_absolute():
            candidates.append(p)

        # Common prefixes from source jsonl fields.
        for prefix in ("data/doppelgangers/", "doppelgangers/"):
            if variant.startswith(prefix):
                stripped = variant[len(prefix) :]
                candidates.append((dataset_root / stripped))

        if variant.startswith("images/"):
            candidates.append(dataset_root / variant)
        else:
            candidates.append(dataset_root / "images" / dataset_split / variant)
            candidates.append(dataset_root / "images" / variant)

        if image_root is not None:
            candidates.append(image_root / variant)

    # Deduplicate while preserving order.
    deduped: List[Path] = []
    seen = set()
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def resolve_existing_path(
    path_text: str,
    record: Dict[str, Any],
    dataset_root: Path,
    dataset_split: str,
    image_root: Optional[Path],
) -> Optional[str]:
    split = infer_split(record, dataset_split)
    for candidate in candidate_paths(path_text, dataset_root, split, image_root):
        if candidate.exists():
            return str(candidate.resolve())
    return None


def resolve_pair_images(
    record: Dict[str, Any],
    image_source: str,
    dataset_root: Path,
    dataset_split: str,
    image_root: Optional[Path],
) -> Optional[Tuple[str, str]]:
    def _pick(base_key: str, abspath_key: str) -> Optional[str]:
        if image_source == "original":
            value = record.get(base_key)
            if value is None:
                value = record.get(abspath_key)
            return value
        if image_source == "original_abspath":
            value = record.get(abspath_key)
            if value is None:
                value = record.get(base_key)
            return value
        # auto
        return (
            record.get(abspath_key)
            or record.get(base_key)
        )

    img0 = _pick("original_image_0", "original_image_0_abspath")
    img1 = _pick("original_image_1", "original_image_1_abspath")
    if not img0 or not img1:
        return None

    resolved0 = resolve_existing_path(
        path_text=img0,
        record=record,
        dataset_root=dataset_root,
        dataset_split=dataset_split,
        image_root=image_root,
    )
    resolved1 = resolve_existing_path(
        path_text=img1,
        record=record,
        dataset_root=dataset_root,
        dataset_split=dataset_split,
        image_root=image_root,
    )
    if not resolved0 or not resolved1:
        return None
    return resolved0, resolved1


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[warn] skip malformed json line {line_idx}")
    return rows


def sample_fraction(items: List[Any], fraction: float, rng: random.Random) -> List[Any]:
    if fraction >= 1.0:
        return list(items)
    if fraction <= 0.0 or not items:
        return []
    k = int(len(items) * fraction)
    if k == 0:
        k = 1
    idxs = sorted(rng.sample(range(len(items)), k))
    return [items[i] for i in idxs]


def split_train_eval(
    items: List[Any],
    eval_fraction: float,
    rng: random.Random,
) -> Tuple[List[Any], List[Any]]:
    if eval_fraction <= 0.0 or not items:
        return items, []
    if eval_fraction >= 1.0:
        return [], list(items)

    eval_size = int(len(items) * eval_fraction)
    if eval_size == 0:
        eval_size = 1
    idxs = list(range(len(items)))
    rng.shuffle(idxs)
    eval_idxs = set(idxs[:eval_size])
    train_items = [x for i, x in enumerate(items) if i not in eval_idxs]
    eval_items = [x for i, x in enumerate(items) if i in eval_idxs]
    return train_items, eval_items


def build_sample(
    record: Dict[str, Any],
    sample_id: str,
    image0: str,
    image1: str,
    decision: str,
    user_prompt: str,
) -> Dict[str, Any]:
    sample = dict(record)  # Keep all source attributes.
    if "label" in sample:
        sample["source_label"] = sample["label"]
    sample["label"] = decision  # training target from final_decision
    sample["target_label_field"] = "final_decision"
    sample["id"] = sample_id
    sample["image"] = {
        "<image_00>": image0,
        "<image_01>": image1,
    }
    sample["conversations"] = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": decision},
    ]
    return sample


def write_json(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_json)
    eval_output_path = Path(args.eval_output_json) if args.eval_output_json else None
    image_root = Path(args.image_root) if args.image_root else None
    dataset_root = Path(args.dataset_root)

    rows = read_jsonl(input_path)
    print(f"[info] loaded rows: {len(rows)}")

    converted: List[Dict[str, Any]] = []
    stats = {
        "skipped_review_filter": 0,
        "skipped_missing_images": 0,
        "skipped_missing_decision": 0,
    }

    for idx, row in enumerate(rows):
        if not keep_by_review_flag(row, args.needs_human_review):
            stats["skipped_review_filter"] += 1
            continue

        decision = normalize_decision(row.get("final_decision"))
        if not decision:
            stats["skipped_missing_decision"] += 1
            continue

        images = resolve_pair_images(
            record=row,
            image_source=args.image_source,
            dataset_root=dataset_root,
            dataset_split=args.dataset_split,
            image_root=image_root,
        )
        if images is None:
            stats["skipped_missing_images"] += 1
            continue
        image0, image1 = images

        pair_index = row.get("pair_index", idx)
        sample_id = f"{args.id_prefix}_{pair_index}"
        converted.append(
            build_sample(
                record=row,
                sample_id=sample_id,
                image0=image0,
                image1=image1,
                decision=decision,
                user_prompt=args.user_prompt,
            )
        )

    converted = sample_fraction(converted, args.fraction, rng)
    if args.max_samples is not None:
        converted = converted[: max(args.max_samples, 0)]

    train_data, eval_data = split_train_eval(converted, args.eval_fraction, rng)
    if args.eval_fraction > 0 and eval_output_path is None:
        eval_output_path = output_path.with_name(f"{output_path.stem}.eval.json")

    write_json(output_path, train_data)
    print(f"[info] wrote train samples: {len(train_data)} -> {output_path}")

    if eval_output_path is not None and eval_data:
        write_json(eval_output_path, eval_data)
        print(f"[info] wrote eval samples: {len(eval_data)} -> {eval_output_path}")

    print("[info] conversion stats:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
