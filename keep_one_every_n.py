#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


def collect_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def main():
    parser = argparse.ArgumentParser(
        description="Copy 1 image every N images (by filename order) to another folder."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/nas1/hyh22/backup/PanoGS/data_big_room2_sparse/input_raw"),
        help="Input folder path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output folder path for kept images.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=8,
        help="Keep 1 file every N files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview copy actions without writing files.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files if destination already exists.",
    )
    args = parser.parse_args()

    if args.step <= 0:
        raise ValueError("--step must be a positive integer")

    folder = args.input_dir
    out_dir = args.output_dir
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Input directory not found: {folder}")

    files = collect_images(folder)
    if not files:
        print(f"No image files found in: {folder}")
        return

    keep = set(files[:: args.step])
    to_copy = sorted(keep)

    print(f"Input: {folder}")
    print(f"Output: {out_dir}")
    print(f"Total images: {len(files)}")
    print(f"Will copy: {len(to_copy)}")

    if args.dry_run:
        print("\n[Dry run] Files to copy:")
        for p in to_copy:
            print(p.name)
        return

    if to_copy and not args.yes:
        answer = input("Proceed with copy? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled.")
            return

    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for src in to_copy:
        dst = out_dir / src.name
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1

    print(f"Done. Copied: {copied}, Skipped(existing): {skipped}")


if __name__ == "__main__":
    main()
