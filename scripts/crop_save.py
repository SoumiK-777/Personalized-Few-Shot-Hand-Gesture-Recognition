"""
Crop hand gesture images using bounding boxes from HaGRID v2 annotations.
Saves cropped images to cropped_images/{race_name}/{train,test,val}/
and prints final race category counts.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
ANNOTATIONS_ROOT = Path("annotations/annotations")   # contains train/, test/, val/
IMAGES_ROOT      = Path("hagridv2_512/HaGRIDv2_dataset_512")
OUTPUT_ROOT      = Path("cropped_images")
SPLITS           = ["train", "test", "val"]
# ─────────────────────────────────────────────────────────────────────────────

race_counts: dict[str, int] = defaultdict(int)
stats = {s: {"found": 0, "missing": 0, "crops": 0} for s in SPLITS}


def xywh_to_pixel(bbox, img_w, img_h):
    """Convert normalised [x, y, w, h] → pixel [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    x1 = int(x * img_w)
    y1 = int(y * img_h)
    x2 = int((x + w) * img_w)
    y2 = int((y + h) * img_h)
    # clamp
    x1, x2 = max(0, x1), min(img_w, x2)
    y1, y2 = max(0, y1), min(img_h, y2)
    return x1, y1, x2, y2


def process_split(split: str):
    ann_dir = ANNOTATIONS_ROOT / split
    if not ann_dir.exists():
        print(f"[WARN] Annotation dir not found: {ann_dir}")
        return

    json_files = sorted(ann_dir.glob("*.json"))
    if not json_files:
        print(f"[WARN] No JSON files in {ann_dir}")
        return

    for json_path in json_files:
        gesture_name = json_path.stem          # e.g. "thumb_index2"
        img_dir = IMAGES_ROOT / gesture_name   # folder with source images

        with open(json_path, "r") as f:
            annotations: dict = json.load(f)

        for img_id, ann in annotations.items():
            # ── locate image ──────────────────────────────────────────────
            img_path = None
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = img_dir / f"{img_id}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                stats[split]["missing"] += 1
                continue

            stats[split]["found"] += 1

            # ── get race(s) for this sample ───────────────────────────────
            meta  = ann.get("meta", {})
            races = [r for r in meta.get("race", []) if r]
            for race in races:
                race_counts[race] += 1

            # ── crop with each bbox ───────────────────────────────────────
            bboxes = ann.get("bboxes", [])
            labels = ann.get("labels", [])

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] Cannot open {img_path}: {e}")
                continue

            img_w, img_h = img.size

            for idx, (bbox, label) in enumerate(zip(bboxes, labels)):
                x1, y1, x2, y2 = xywh_to_pixel(bbox, img_w, img_h)
                if x2 <= x1 or y2 <= y1:
                    continue  # degenerate box

                crop = img.crop((x1, y1, x2, y2))

                # Save as cropped_images/<race>/<split>/<gesture>/<img_id>_<idx>.jpg
                # One image may have multiple races; save under each.
                target_races = races if races else ["unknown"]
                for race in target_races:
                    race_dir = OUTPUT_ROOT / race / split / label
                    race_dir.mkdir(parents=True, exist_ok=True)
                    out_path = race_dir / f"{img_id}_{idx}.jpg"
                    crop.save(out_path, "JPEG", quality=95)

                stats[split]["crops"] += 1

    print(f"[{split:5s}]  found={stats[split]['found']}  "
          f"missing={stats[split]['missing']}  "
          f"crops={stats[split]['crops']}")


def main():
    for split in SPLITS:
        print(f"\nProcessing split: {split}")
        process_split(split)

    # ── Race category summary ─────────────────────────────────────────────────
    print("\n" + "=" * 45)
    print("  Race category counts (all splits combined)")
    print("=" * 45)
    total = sum(race_counts.values())
    for race, count in sorted(race_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total else 0
        print(f"  {race:<25s} {count:>7,}  ({pct:.1f}%)")
    print(f"  {'TOTAL':<25s} {total:>7,}")
    print("=" * 45)

    # ── Overall crop summary ──────────────────────────────────────────────────
    print("\nCrop summary by split:")
    for split in SPLITS:
        print(f"  {split}: {stats[split]['crops']:,} crops "
              f"({stats[split]['missing']} images skipped / not found)")


if __name__ == "__main__":
    main()