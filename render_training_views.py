"""Render all training views of a stylized scene to a folder.

Loads a stylized point cloud (the final ``scene_styled.ply`` produced by
``run_stylization.py`` / ``run_stylization_old.py``) together with the
colmap scene, then renders every training camera view with the 2D-GS
viewer rasterizer and writes the images to an output directory.

The stylized PLY can be supplied directly via ``--model_path``. If
``--prep_dir`` is given instead, the script reads ``preprocess.json`` to
find the last processed camera and loads
``{prep_dir}/cam_{last}/styled/scene_styled.ply`` automatically.

Usage:
    python render_training_views.py \
        --model_path /path/to/scene_styled.ply \
        -s /path/to/colmap/source \
        --output_dir renders/myrun

    # or auto-locate the final styled PLY from a preprocess dir:
    python render_training_views.py \
        --prep_dir preprocess/myrun \
        -s /path/to/colmap/source \
        --output_dir renders/myrun

    CUDA_VISIBLE_DEVICES=6 python render_training_views.py \
        /data1/hyh/github/PanoGS/preprocess/dl3dv_389a4_fast_train256_style256_script/cam_15/styled/scene_styled.ply \
        -s /data1/hyh/github/PanoGS/data_3dgs/DL3DV-10K-Benchmark/389a460ca1995e0658e85fe8e6b520b4e88b370cd6710dfe728b1564bba31aee/gaussian_splat \
        --output_dir /data1/hyh/github/PanoGS/output_trainingview/dl3dv_389a4_fast_train256_style256
"""

import os
import sys
import json
import re
import warnings
from argparse import ArgumentParser
from typing import Tuple, Optional, List

warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "2d_gaussian_splatting"))

import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm

from internal.viewer import ViewerRenderer
from internal.viewer import GaussianModelforViewer as GaussianModel
from internal.utils.sh_utils import RGB2SH, SH2RGB
from scene.cameras import Simple_Camera
from scene.dataset_readers import sceneLoadTypeCallbacks


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _ConfigSlider:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


def _parse_background_color(values):
    if len(values) == 1 and isinstance(values[0], str):
        if values[0] == "white":
            return (1.0, 1.0, 1.0)
        if values[0] == "black":
            return (0.0, 0.0, 0.0)
        return (0.5, 0.5, 0.5)
    return tuple(float(v) for v in values)


def _sanitize_filename(name: str) -> str:
    # keep the original stem but replace path separators
    name = name.replace(os.sep, "_").replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name


def _resolve_styled_ply(
    model_path: Optional[str], prep_dir: Optional[str]
) -> str:
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model_path not found: {model_path}")
        return model_path

    if not prep_dir:
        raise ValueError(
            "Either --model_path or --prep_dir must be provided."
        )

    preprocess_json_path = os.path.join(prep_dir, "preprocess.json")
    if not os.path.exists(preprocess_json_path):
        raise FileNotFoundError(
            f"preprocess.json not found in prep_dir: {prep_dir}"
        )
    with open(preprocess_json_path, "r") as f:
        preprocess_json = json.load(f)
    cam_id = preprocess_json.get("cam_id", [])
    if not cam_id:
        raise ValueError("preprocess.json contains no cam_id list.")

    last_cam = cam_id[-1]
    candidates = [
        os.path.join(prep_dir, f"cam_{last_cam}", "styled", "scene_styled.ply"),
        os.path.join(prep_dir, f"cam_{last_cam}", "scene_styled.ply"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            print(f"[INFO] auto-located styled PLY: {cand}")
            return cand
    raise FileNotFoundError(
        f"Could not find scene_styled.ply for cam_{last_cam} under {prep_dir}"
    )


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #


class TrainingViewRenderer:
    """Headless renderer that mirrors the viewer's render path for the
    colmap training cameras."""

    def __init__(
        self,
        model_path: str,
        source_path: str,
        output_dir: str,
        background_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        sh_degree: int = 0,
        active_sh_degree: int = 3,
        scale: float = 1.0,
        depth_ratio: float = 0.0,
        sparsity: int = 1,
        enable_ptc: bool = False,
        surfel_mode: str = "ptc",
        point_size: float = 0.01,
    ):
        self.model_path = model_path
        self.source_path = source_path
        self.output_dir = output_dir
        self.sh_degree = sh_degree
        self.active_sh_degree = active_sh_degree

        self.device = torch.device("cuda")
        self.background_color = torch.tensor(
            background_color, dtype=torch.float32, device="cuda"
        )

        # Slider wrappers so _render_params stays identical to the
        # stylization pipeline.
        self.scale_slider = _ConfigSlider(float(scale))
        self.depth_ratio_slider = _ConfigSlider(float(depth_ratio))
        self.sparsity_slider = _ConfigSlider(int(sparsity))
        self.enable_ptc = _ConfigSlider(bool(enable_ptc))
        self.surfel_mode = _ConfigSlider(str(surfel_mode))
        self.point_size = _ConfigSlider(float(point_size))
        self.active_sh_degree_slider = _ConfigSlider(int(active_sh_degree))

        self._init_model()
        self._init_colmap_cameras()

    def _init_model(self):
        self.gaussian_model = GaussianModel(sh_degree=self.sh_degree)
        ply_path = self.model_path
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"Point cloud not found: {ply_path}")
        print(f"[INFO] ply path loaded from: {ply_path}")
        self.gaussian_model.load_ply(ply_path)
        print(f"[INFO] number of points: {self.gaussian_model._xyz.shape[0]}")
        self.gaussian_model.active_sh_degree = int(self.active_sh_degree)
        self.viewer_renderer = ViewerRenderer(
            self.gaussian_model, self.background_color, True
        )

    def _init_colmap_cameras(self):
        col_scene = sceneLoadTypeCallbacks["Colmap"](self.source_path, None, False)
        # train_cameras here are CameraInfo namedtuples (uid, R, T, FovX,
        # FovY, image, image_path, image_name, width, height).
        self.train_cameras = col_scene.train_cameras
        print(f"[INFO] number of training cameras: {len(self.train_cameras)}")

    def _render_params(self, override_color=None):
        params = {
            "active_sh_degree": self.active_sh_degree_slider.value,
            "scaling_modifier": self.scale_slider.value,
            "depth_ratio": self.depth_ratio_slider.value,
            "bg_color": self.viewer_renderer.background_color,
            "sparsity": self.sparsity_slider.value,
            "valid_range": None,
            "show_ptc": self.enable_ptc.value
            and (self.surfel_mode.value == "ptc"),
            "show_disk": self.enable_ptc.value
            and (self.surfel_mode.value == "disk"),
            "point_size": self.point_size.value,
        }
        if override_color is not None:
            params["override_color"] = override_color
        return params

    def render_camera(self, cam_info):
        cam = Simple_Camera(
            cam_info.uid,
            np.array(cam_info.R),
            np.array(cam_info.T),
            float(cam_info.FovX),
            float(cam_info.FovY),
            int(cam_info.height),
            int(cam_info.width),
            cam_info.image_name,
            cam_info.uid,
        )

        render_params = self._render_params(
            override_color=SH2RGB(self.gaussian_model._features_dc.squeeze())
        )

        with torch.no_grad():
            results = self.viewer_renderer.render_viewer(cam, **render_params)
        rendered = results["render"].clamp(0, 1).permute(1, 2, 0)
        rendered = (rendered.detach().cpu().numpy() * 255).astype(np.uint8)
        return rendered

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[INFO] rendering {len(self.train_cameras)} views into {self.output_dir}")

        with torch.no_grad():
            for cam_info in tqdm(self.train_cameras):
                img = self.render_camera(cam_info)
                name = _sanitize_filename(cam_info.image_name)
                if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                    name = name + ".png"
                Image.fromarray(img).save(os.path.join(self.output_dir, name))
                torch.cuda.empty_cache()

        print(f"[INFO] done. outputs in {self.output_dir}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main():
    parser = ArgumentParser(
        description="Render all training views of a stylized PanoGS scene."
    )
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default=None,
        help="Stylized PLY file. Optional if --prep_dir is given.",
    )
    parser.add_argument("--source_path", "-s", type=str, required=True)
    parser.add_argument(
        "--prep_dir",
        type=str,
        default=None,
        help="Preprocess dir; used to auto-locate the final scene_styled.ply "
        "when model_path is omitted.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cameras_json", type=str, default=None)
    parser.add_argument(
        "--background_color",
        "-b",
        type=str,
        nargs="+",
        default=["gray"],
        help="e.g. white, gray, black, '0.5 0.5 0.5'",
    )
    parser.add_argument("--sh_degree", type=int, default=0)
    parser.add_argument(
        "--active_sh_degree",
        type=int,
        default=3,
        help="Active SH degree for rendering (matches stylization default).",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--depth_ratio", type=float, default=0.0)
    parser.add_argument("--sparsity", type=int, default=1)
    parser.add_argument(
        "--float32_matmul_precision", "--fp", type=str, default=None
    )
    args = parser.parse_args()

    if args.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(args.float32_matmul_precision)

    styled_ply = _resolve_styled_ply(args.model_path, args.prep_dir)

    renderer = TrainingViewRenderer(
        model_path=styled_ply,
        source_path=args.source_path,
        output_dir=args.output_dir,
        background_color=_parse_background_color(args.background_color),
        sh_degree=args.sh_degree,
        active_sh_degree=args.active_sh_degree,
        scale=args.scale,
        depth_ratio=args.depth_ratio,
        sparsity=args.sparsity,
    )
    renderer.run()


if __name__ == "__main__":
    main()
