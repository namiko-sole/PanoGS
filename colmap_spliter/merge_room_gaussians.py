#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from internal.utils.gaussian_utils import Gaussian


def _load_gaussians(ply_paths: List[Path]) -> List[Gaussian]:
    gaussians: List[Gaussian] = []
    for p in ply_paths:
        if not p.exists():
            raise FileNotFoundError(f"PLY not found: {p}")
        g = Gaussian.load_from_ply(str(p), sh_degrees=-1)
        gaussians.append(g)
        print(f"[INFO] loaded {p} | points={g.xyz.shape[0]} | sh_degree={g.sh_degrees}")
    return gaussians


def _check_compatible(gaussians: List[Gaussian]) -> None:
    if not gaussians:
        raise RuntimeError("No Gaussian inputs provided")

    base = gaussians[0]
    for idx, g in enumerate(gaussians[1:], start=1):
        if g.sh_degrees != base.sh_degrees:
            raise ValueError(
                f"Incompatible SH degree: input#{idx} has {g.sh_degrees}, expected {base.sh_degrees}"
            )
        if g.features_rest.shape[1:] != base.features_rest.shape[1:]:
            raise ValueError(
                f"Incompatible features_rest shape: input#{idx} has {g.features_rest.shape}, "
                f"expected (*, {base.features_rest.shape[1]}, {base.features_rest.shape[2]})"
            )
        if g.scales.shape[1] != base.scales.shape[1]:
            raise ValueError(
                f"Incompatible scales shape: input#{idx} has {g.scales.shape}, expected (*, {base.scales.shape[1]})"
            )
        if g.rotations.shape[1] != base.rotations.shape[1]:
            raise ValueError(
                f"Incompatible rotations shape: input#{idx} has {g.rotations.shape}, expected (*, {base.rotations.shape[1]})"
            )


def _merge_gaussians(gaussians: List[Gaussian]) -> Gaussian:
    merged = Gaussian(
        sh_degrees=gaussians[0].sh_degrees,
        xyz=np.concatenate([g.xyz for g in gaussians], axis=0),
        opacities=np.concatenate([g.opacities for g in gaussians], axis=0),
        features_dc=np.concatenate([g.features_dc for g in gaussians], axis=0),
        features_rest=np.concatenate([g.features_rest for g in gaussians], axis=0),
        scales=np.concatenate([g.scales for g in gaussians], axis=0),
        rotations=np.concatenate([g.rotations for g in gaussians], axis=0),
        real_features_extra=np.concatenate([g.real_features_extra for g in gaussians], axis=0),
    )
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge multiple room-stylized Gaussian PLY files into one scene PLY."
    )
    parser.add_argument(
        "--ply",
        nargs="+",
        required=True,
        help="Input room PLY paths, e.g. room_1_scene_styled.ply room_2_scene_styled.ply ...",
    )
    parser.add_argument("--output", required=True, help="Output merged .ply path")
    parser.add_argument(
        "--with-colors",
        action="store_true",
        help="Also write RGB color attributes for convenience when visualizing in generic PLY tools",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    ply_paths = [Path(p).resolve() for p in args.ply]
    output_path = Path(args.output).resolve()

    gaussians = _load_gaussians(ply_paths)
    _check_compatible(gaussians)
    merged = _merge_gaussians(gaussians)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save_to_ply(str(output_path), with_colors=bool(args.with_colors))

    print(f"[INFO] merged inputs: {len(ply_paths)}")
    print(f"[INFO] merged points: {merged.xyz.shape[0]}")
    print(f"[INFO] output: {output_path}")


if __name__ == "__main__":
    main()
