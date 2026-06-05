#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from internal.utils.colmap import (
    Camera,
    Image,
    Point3D,
    qvec2rotmat,
    read_images_text,
    read_points3D_text,
    rotmat2qvec,
    write_cameras_text,
    write_images_text,
    write_points3D_text,
)


def _resolve_transforms_path(input_dir: Path, preferred: str) -> Path:
    candidates = []
    if preferred:
        candidates.append(input_dir / preferred)
        candidates.append(input_dir / "nerfstudio" / preferred)

    candidates.extend(
        [
            input_dir / "nerfstudio" / "transforms_undistorted.json",
            input_dir / "nerfstudio" / "transforms.json",
            input_dir / "transforms_undistorted.json",
            input_dir / "transforms.json",
        ]
    )

    for p in candidates:
        if p.exists():
            return p

    tried = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"No transforms json found. Tried:\n{tried}")


def _frame_to_colmap_pose(c2w_gl: np.ndarray):
    # Match existing 3DGS/2DGS blender loader conversion.
    c2w = c2w_gl.copy()
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    rot = w2c[:3, :3]
    tvec = w2c[:3, 3]
    qvec = rotmat2qvec(rot)
    return qvec, tvec


def _load_transforms(transforms_path: Path):
    with transforms_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    required = ["fl_x", "fl_y", "cx", "cy", "w", "h", "frames"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing required keys in {transforms_path}: {missing}")

    return data


def _camera_center_from_image(image: Image) -> np.ndarray:
    rot = qvec2rotmat(image.qvec)
    return -rot.T @ image.tvec


def _estimate_similarity(src_xyz: np.ndarray, dst_xyz: np.ndarray):
    if src_xyz.shape != dst_xyz.shape or src_xyz.ndim != 2 or src_xyz.shape[1] != 3:
        raise ValueError("Invalid camera center arrays for similarity estimation")
    if src_xyz.shape[0] < 3:
        raise ValueError("Need at least 3 camera correspondences to estimate similarity")

    src_mean = src_xyz.mean(axis=0)
    dst_mean = dst_xyz.mean(axis=0)

    src_centered = src_xyz - src_mean
    dst_centered = dst_xyz - dst_mean

    cov = (dst_centered.T @ src_centered) / src_xyz.shape[0]
    u, s, vt = np.linalg.svd(cov)
    rot = u @ vt

    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt

    src_var = np.sum(src_centered**2) / src_xyz.shape[0]
    if src_var <= 0:
        raise ValueError("Degenerate source camera centers; cannot estimate scale")

    scale = float(np.sum(s) / src_var)
    trans = dst_mean - scale * (rot @ src_mean)
    return scale, rot, trans


def _transform_points3d_to_converted_frame(input_dir: Path, converted_images: dict):
    points3d_src = input_dir / "colmap" / "points3D.txt"
    images_src = input_dir / "colmap" / "images.txt"

    if not points3d_src.exists():
        raise FileNotFoundError(f"points3D source not found: {points3d_src}")
    if not images_src.exists():
        raise FileNotFoundError(f"images source not found (needed for frame alignment): {images_src}")

    src_images = read_images_text(str(images_src))
    src_by_name = {img.name: img for img in src_images.values()}
    dst_by_name = {img.name: img for img in converted_images.values()}

    common_names = sorted(set(src_by_name.keys()) & set(dst_by_name.keys()))
    if len(common_names) < 3:
        raise ValueError(
            f"Not enough shared image names to align points3D: {len(common_names)} found"
        )

    src_centers = np.stack([_camera_center_from_image(src_by_name[name]) for name in common_names], axis=0)
    dst_centers = np.stack([_camera_center_from_image(dst_by_name[name]) for name in common_names], axis=0)
    scale, rot, trans = _estimate_similarity(src_centers, dst_centers)

    aligned_centers = (scale * (rot @ src_centers.T)).T + trans
    center_err = np.linalg.norm(aligned_centers - dst_centers, axis=1)

    src_points = read_points3D_text(str(points3d_src))
    transformed_points = {}
    for point_id, point in src_points.items():
        xyz_new = scale * (rot @ point.xyz) + trans
        transformed_points[point_id] = Point3D(
            id=point.id,
            xyz=xyz_new,
            rgb=point.rgb,
            error=point.error,
            image_ids=point.image_ids,
            point2D_idxs=point.point2D_idxs,
        )

    stats = {
        "src_points_path": str(points3d_src),
        "src_images_path": str(images_src),
        "shared_images": len(common_names),
        "scale": scale,
        "center_err_p50": float(np.percentile(center_err, 50)),
        "center_err_p95": float(np.percentile(center_err, 95)),
        "center_err_max": float(np.max(center_err)),
    }
    return transformed_points, stats


def convert(input_dir: Path, output_dir: Path, transforms_name: str, skip_bad: bool, copy_points3d: bool):
    transforms_path = _resolve_transforms_path(input_dir, transforms_name)
    data = _load_transforms(transforms_path)
    frames = data["frames"]

    camera_model = data.get("camera_model", "PINHOLE")
    if camera_model != "PINHOLE":
        raise ValueError(
            f"Expected PINHOLE transforms for 2DGS compatibility, got: {camera_model}. "
            "Please use an undistorted transforms json."
        )

    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    camera = Camera(
        id=1,
        model="PINHOLE",
        width=int(data["w"]),
        height=int(data["h"]),
        params=np.array(
            [
                float(data["fl_x"]),
                float(data["fl_y"]),
                float(data["cx"]),
                float(data["cy"]),
            ],
            dtype=float,
        ),
    )

    images = {}
    skipped_bad = 0
    for i, frame in enumerate(frames, start=1):
        if skip_bad and bool(frame.get("is_bad", False)):
            skipped_bad += 1
            continue

        c2w = np.asarray(frame["transform_matrix"], dtype=float)
        if c2w.shape != (4, 4):
            raise ValueError(f"Invalid transform_matrix shape for frame {frame.get('file_path')}: {c2w.shape}")

        qvec, tvec = _frame_to_colmap_pose(c2w)
        image_name = Path(frame["file_path"]).name
        images[len(images) + 1] = Image(
            id=len(images) + 1,
            qvec=qvec,
            tvec=tvec,
            camera_id=1,
            name=image_name,
            xys=np.zeros((0, 2), dtype=float),
            point3D_ids=np.zeros((0,), dtype=np.int64),
        )

    cameras = {1: camera}
    write_cameras_text(cameras, str(sparse_dir / "cameras.txt"))
    write_images_text(images, str(sparse_dir / "images.txt"))

    points3d_dst = sparse_dir / "points3D.txt"
    if copy_points3d:
        transformed_points, transform_stats = _transform_points3d_to_converted_frame(input_dir, images)
        write_points3D_text(transformed_points, str(points3d_dst))
        points3d_mode = (
            "transformed from source points3D with camera-aligned similarity: "
            f"shared_images={transform_stats['shared_images']}, "
            f"scale={transform_stats['scale']:.9f}, "
            f"center_err_p95={transform_stats['center_err_p95']:.6e}"
        )
    else:
        write_points3D_text({}, str(points3d_dst))
        points3d_mode = "created empty points3D.txt"

    print(f"transforms: {transforms_path}")
    print(f"output sparse: {sparse_dir}")
    print(f"camera_model: {camera_model}")
    print(f"frames total: {len(frames)}")
    print(f"frames exported: {len(images)}")
    print(f"frames skipped_bad: {skipped_bad}")
    print(f"points3D: {points3d_mode}")


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Convert nerfstudio transforms to COLMAP sparse text model (sparse/0)."
    )
    parser.add_argument("--input_dir", required=True, help="Input dataset directory")
    parser.add_argument("--output_dir", required=True, help="Output dataset directory")
    parser.add_argument(
        "--transforms_name",
        default="transforms_undistorted.json",
        help="Preferred transforms filename (searched in input_dir and input_dir/nerfstudio)",
    )
    parser.add_argument(
        "--skip_bad",
        action="store_true",
        help="Skip frames with is_bad=true in transforms json",
    )
    parser.add_argument(
        "--no_copy_points3d",
        action="store_true",
        help="Do not copy input_dir/colmap/points3D.txt; write an empty points3D.txt instead",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    convert(
        input_dir=Path(args.input_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        transforms_name=args.transforms_name,
        skip_bad=args.skip_bad,
        copy_points3d=not args.no_copy_points3d,
    )


if __name__ == "__main__":
    main()


"""
python convert_nerfstudio_to_colmap.py \
  --input_dir data_big_room_816e9 \
  --output_dir data_big_room_816e9_2dgs \
  --skip_bad

python convert_nerfstudio_to_colmap.py \
  --input_dir data_big_room_2a1b5 \
  --output_dir data_big_room_2a1b5_2dgs

  
python convert_nerfstudio_to_colmap.py --input_dir data_big_room_816e9 --output_dir data_big_room_816e9_2dgs_optimized --skip_bad
"""