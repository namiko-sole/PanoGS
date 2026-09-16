"""Standalone preprocess + stylization pipeline.

Usage:
    python run_stylization.py \
        /path/to/point_cloud.ply \
        -s /path/to/colmap/source \
        --cameras_json /path/to/cameras.json \
        --style_img /path/to/style.png \
        --prep_dir preprocess/myrun \
        --prompt "a cartoon-style room"

example:
    CUDA_VISIBLE_DEVICES=0 python run_stylization.py \
        /nas1/nas1/data/hyh22/backup/PanoGS/data_2dgs/output/playroom/point_cloud/iteration_30000/point_cloud.ply \
        -s /nas1/nas1/data/hyh22/backup/PanoGS/data_3dgs/db/playroom \
        --cameras_json /nas1/nas1/data/hyh22/backup/PanoGS/data_2dgs/output/playroom/cameras.json \
        --style_img /nas1/nas1/data/hyh22/backup/PanoGS/style_data/indoor/room3.jpg \
        --prep_dir preprocess/my_scene_test \
        --prompt ""
"""

import os
import sys
import math
import json
import random
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, List, Optional

warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "2d_gaussian_splatting"))

import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

import viser.transforms as vtf

from internal.viewer import ViewerRenderer
from internal.viewer import GaussianModelforViewer as GaussianModel
from internal.utils.sh_utils import RGB2SH, SH2RGB
from internal.utils.graphics_utils import fov2focal
from internal.utils.pano_utils import (
    get_depth_distort,
    direction_to_pano_coord,
    pano_to_img_coord,
)
import internal.utils.equirec.Equirec2Perspec as E2P
import internal.utils.equirec.multi_Perspec2Equirec as m_P2E
from internal.utils.point_cloud import get_hidden_point_mask
from internal.utils.knn import K_nearest_neighbors
from internal.utils.loss_utils import l1_loss, ssim

from scene.cameras import Simple_Camera
from scene.dataset_readers import sceneLoadTypeCallbacks

from arguments import OptimizationParams

from diffusers_inference import generate_image
from internal.utils.adain_utils.adain_api import generate_adain


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #


class _ConfigSlider:
    """Minimal stand-in for viser GUI sliders.

    The viewer code accesses ``foo.value`` everywhere; wrapping a config value
    in this object lets the copied logic stay untouched.
    """

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


# Default configuration. Mirrors the GUI defaults defined in viewer.py
# (see `_setup_general_features_folder`). These are merged with any YAML
# overrides loaded from --config.
DEFAULT_CONFIG = {
    "render": {
        "scale": 1.0,
        "depth_ratio": 0.0,
        "sparsity": 1,
        "enable_ptc": False,
        "surfel_mode": "ptc",
        "point_size": 0.01,
        "active_sh_degree": 0,
    },
    "preprocess": {
        "camera_gap": 3.0,
        "min_camera_object_dist": 0.3,
        "enable_camera_center": False,
        "camera_focus": True,
        "sort_camera_center": True,
        "start_from_center": True,
        "mask_threshold": 0.9,
        "pano_res": 1024,
        # Cap on number of candidate cameras. If selection exceeds this,
        # camera_gap is increased by ``max_candidates_gap_step`` and the
        # selection retried, up to ``max_candidates_max_iters`` times.
        # Set to 0 or a negative number to disable the cap.
        "max_candidates": 10,
        "max_candidates_gap_step": 1.0,
        "max_candidates_max_iters": 50,
    },
    "stylization": {
        "active_sh_degree": 3,
        "train_steps": 100,
        "color_lr_scaler": 3.0,
        "knn_number": 3,
        "train_res": 1024,
        "num_sample_views": 6,
        "step_per_cam": 25,
        "step_base": 200,
        "color_upscale": 4,
        "missing_rate_threshold": 0.7,
        "first_cam_strength": 1.0,
        "subsequent_cam_strength": 0.3,
    },
}


def load_config(path: Optional[str]):
    cfg = OmegaConf.create(DEFAULT_CONFIG)
    if path is not None:
        override = OmegaConf.load(path)
        cfg = OmegaConf.merge(cfg, override)
    return cfg


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


PERS_PARAMS = [
    (90, 0, 0),
    (90, 90, 0),
    (90, 180, 0),
    (90, 270, 0),
    (90, 0, 90),
    (90, 0, -90),
]


class StylizationPipeline:
    """Headless port of the viewer's preprocess/stylization callbacks.

    The structure intentionally mirrors :class:`viewer.Viewer` so the copied
    methods need only minimal edits (sliders -> ``_ConfigSlider``, GUI side
    effects -> no-ops).
    """

    def __init__(
        self,
        cfg,
        model_path: str,
        source_path: str,
        cameras_json: str,
        style_img: str,
        prep_dir: str,
        prompt: str = "",
        background_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        sh_degree: int = 0,
        iterations: int = 30000,
    ):
        self.cfg = cfg
        self.model_path = model_path
        self.source_path = source_path
        self.cameras_json = cameras_json
        self.style_img = style_img
        self.prep_dir = prep_dir
        self.sh_degree = sh_degree
        self.iterations = iterations

        self.device = torch.device("cuda")
        self.total_device_memory = (
            torch.cuda.get_device_properties(self.device).total_memory / 1024 ** 2
        )
        self.background_color = torch.tensor(
            background_color, dtype=torch.float32, device="cuda"
        )

        # Build slider-compatible wrappers around config values so the copied
        # `render_params` dictionaries keep working unchanged.
        r = cfg.render
        self.scale_slider = _ConfigSlider(float(r.scale))
        self.depth_ratio_slider = _ConfigSlider(float(r.depth_ratio))
        self.sparsity_slider = _ConfigSlider(int(r.sparsity))
        self.enable_ptc = _ConfigSlider(bool(r.enable_ptc))
        self.surfel_mode = _ConfigSlider(str(r.surfel_mode))
        self.point_size = _ConfigSlider(float(r.point_size))
        self.active_sh_degree_slider = _ConfigSlider(int(r.active_sh_degree))

        p = cfg.preprocess
        self.camera_gap_slider = _ConfigSlider(float(p.camera_gap))
        self.min_camera_object_dist_slider = _ConfigSlider(
            float(p.min_camera_object_dist)
        )
        self.enable_camera_center = _ConfigSlider(bool(p.enable_camera_center))
        self.camera_focus = _ConfigSlider(bool(p.camera_focus))
        self.sort_camera_center = _ConfigSlider(bool(p.sort_camera_center))
        self.start_from_center = _ConfigSlider(bool(p.start_from_center))

        s = cfg.stylization
        self.train_steps = _ConfigSlider(int(s.train_steps))
        self.color_lr_scaler = _ConfigSlider(float(s.color_lr_scaler))
        self.knn_number = _ConfigSlider(int(s.knn_number))
        self.prompt_text = _ConfigSlider(str(prompt))

        self._init_models(iterations)
        self._init_camera_poses(self.cameras_json)
        self._init_colmap_cameras()

    # ------------------------------------------------------------------ #
    # Initialisation                                                     #
    # ------------------------------------------------------------------ #

    def _init_models(self, iterations):
        self.gaussian_model = GaussianModel(sh_degree=self.sh_degree)
        if self.model_path.lower().endswith(".ply"):
            self.ply_path = self.model_path
        else:
            self.ply_path = os.path.join(
                self.model_path, "point_cloud", f"iteration_{iterations}", "point_cloud.ply"
            )
        if not os.path.exists(self.ply_path):
            raise FileNotFoundError(f"Point cloud not found: {self.ply_path}")
        print(f"[INFO] ply path loaded from: {self.ply_path}")
        self.gaussian_model.load_ply(self.ply_path)
        print(f"[INFO] number of points: {self.gaussian_model._xyz.shape[0]}")
        # Mirror viewer.py: read-only renderer (third arg = True).
        self.viewer_renderer = ViewerRenderer(
            self.gaussian_model, self.background_color, True
        )

    def _init_camera_poses(self, cameras_json_path):
        if not os.path.exists(cameras_json_path):
            self.camera_poses = []
            self.camera_center = np.zeros(3)
            return
        with open(cameras_json_path, "r") as f:
            camera_poses = json.load(f)
        if camera_poses:
            self.camera_center = np.mean(
                np.asarray([i["position"] for i in camera_poses]), axis=0
            )
        else:
            self.camera_center = np.zeros(3)
        self.camera_poses = camera_poses

    def _init_colmap_cameras(self):
        col_scene = sceneLoadTypeCallbacks["Colmap"](self.source_path, None, False)
        self.scene = col_scene
        self.colmap_cameras = col_scene.train_cameras

    # ------------------------------------------------------------------ #
    # GUI shims                                                          #
    # ------------------------------------------------------------------ #

    def update_client(self):
        """No-op replacement for viser's client refresh.

        Intentionally retained at every original call site so future hooks
        (logging, checkpoint saving, etc.) can plug in without diffing.
        """
        pass

    def get_gpu_memory_usage(self):
        total_memory = (
            torch.cuda.memory_allocated() + torch.cuda.memory_reserved()
        )
        return f"{total_memory / 1024 ** 2:.1f} / {self.total_device_memory:.1f} MB"

    # ------------------------------------------------------------------ #
    # Panorama rendering                                                 #
    # ------------------------------------------------------------------ #

    def render_panorama(
        self,
        camera_center,
        res=1024,
        camera_rotation=np.diag(np.ones((3))),
        save_dir=None,
        mask=None,
    ):
        # print(f"generate panorama center in {camera_center}...")
        Rs = [
            torch.from_numpy(
                camera_rotation
                @ vtf.SO3.from_rpy_radians(
                    math.radians(p[2]), math.radians(p[1]), math.radians(0)
                ).as_matrix()
            )
            for p in PERS_PARAMS
        ]
        T = torch.tensor([0, 0, 0])
        Trans = torch.tensor(camera_center)

        cams = [
            Simple_Camera(
                0,
                R.numpy(),
                T.numpy(),
                math.radians(90),
                math.radians(90),
                res,
                res,
                "",
                0,
                trans=Trans.numpy(),
            )
            for R in Rs
        ]

        override_color = (
            SH2RGB(self.gaussian_model._features_dc.squeeze())
            if mask is None
            else torch.from_numpy(mask)[..., None]
            .float()
            .repeat(1, 3)
            .to(self.device)
        )
        render_params = self._render_params(override_color=override_color)

        with torch.no_grad():
            results = [
                self.viewer_renderer.render_viewer(cam, **render_params)
                for cam in cams
            ]
        torch.cuda.synchronize()

        depth_distort = get_depth_distort(res=res).to(
            results[0]["surf_depth"].device
        ) * math.radians(90)

        pano_images = []
        pano_depthes = []
        for result in results:
            pano_image_face = (
                result["render"].clip(0, 1).permute(1, 2, 0).detach().cpu().numpy()
                * 255
            ).astype(np.uint8)
            pano_depth_face = (
                (result["surf_depth"][0, :, :, 0] + depth_distort)[..., None]
                .detach()
                .cpu()
                .numpy()
            )
            pano_images.append(pano_image_face)
            pano_depthes.append(np.repeat(pano_depth_face, 3, axis=-1))

        pano_image_depthes = [
            np.concatenate((image.astype(np.float32), depth[..., :1]), axis=-1)
            for image, depth in zip(pano_images, pano_depthes)
        ]
        ee_image_depth = m_P2E.Perspective(pano_image_depthes, PERS_PARAMS)

        # print("generating panorama image/depth...")
        pano_image_depth = ee_image_depth.GetEquirec(res, res * 2)
        pano_image = pano_image_depth[..., :3]

        if save_dir:
            # print(f"saving into {save_dir}")
            os.makedirs(save_dir, exist_ok=True)
            Image.fromarray(np.clip(pano_image, 0, 255).astype(np.uint8)).save(
                os.path.join(save_dir, "pano_img.png")
            )
            np.save(
                os.path.join(save_dir, "camera_center.npy"),
                Trans.detach().cpu().numpy(),
            )
            np.save(os.path.join(save_dir, "camera_rotation.npy"), camera_rotation)
            np.save(os.path.join(save_dir, "pano_pimages.npy"), pano_images)

        return pano_image

    def check_and_render_panorama(
        self, camera_center, camera_rotation, save_dir, mask=None, res=1024
    ):
        if os.path.exists(os.path.join(save_dir, "pano_img.png")):
            # print(f"Loading panorama in {save_dir}...")
            image = np.array(Image.open(os.path.join(save_dir, "pano_img.png")))
        else:
            # print(f"Generating panorama to {save_dir}...")
            image = self.render_panorama(
                camera_center,
                camera_rotation=camera_rotation,
                save_dir=save_dir,
                res=res,
                mask=mask,
            )
        return image

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
            "sphere_mode": True,
        }
        if override_color is not None:
            params["override_color"] = override_color
        return params

    # ------------------------------------------------------------------ #
    # Camera selection                                                   #
    # ------------------------------------------------------------------ #

    def is_camera_center_too_close(self, camera_center):
        min_dist = float(self.min_camera_object_dist_slider.value)
        if min_dist <= 0:
            return False
        point_xyz = self.gaussian_model.get_xyz
        if point_xyz.shape[0] == 0:
            return False
        with torch.no_grad():
            center = torch.tensor(
                camera_center, dtype=point_xyz.dtype, device=point_xyz.device
            )
            nearest_dist = torch.linalg.norm(point_xyz - center, dim=1).min().item()
        return nearest_dist < min_dist

    def _get_preprocess_candidate_cam_ids(self):
        def compute_mean_nn_distance(ids):
            if len(ids) < 2:
                return 0.0, np.zeros(len(ids), dtype=np.float32)
            centers = np.asarray(
                [self.camera_poses[idx]["position"] for idx in ids],
                dtype=np.float32,
            )
            pairwise_dist = np.linalg.norm(
                centers[:, None, :] - centers[None, :, :], axis=-1
            )
            np.fill_diagonal(pairwise_dist, np.inf)
            nn_dist = pairwise_dist.min(axis=1)
            return float(nn_dist.mean()), nn_dist

        def get_gap_threshold():
            all_cam_id = list(range(len(self.colmap_cameras)))
            filter_cam_id = all_cam_id.copy()

            avg_all_dist, all_nn_dist = compute_mean_nn_distance(all_cam_id)
            if len(all_cam_id) > 1 and avg_all_dist > 0:
                filter_cam_id = [
                    idx
                    for idx, nn_dist in zip(all_cam_id, all_nn_dist)
                    if nn_dist >= avg_all_dist
                ]
                if len(filter_cam_id) == 0:
                    filter_cam_id = all_cam_id.copy()

            avg_filtered_dist, _ = compute_mean_nn_distance(filter_cam_id)
            if avg_filtered_dist > 0:
                gap_threshold = (
                    float(self.camera_gap_slider.value) * avg_filtered_dist
                )
            else:
                gap_threshold = float(self.camera_gap_slider.value)

            debug_stats = {
                "avg_all": float(avg_all_dist),
                "avg_filtered": float(avg_filtered_dist),
                "slider": float(self.camera_gap_slider.value),
                "gap_threshold": float(gap_threshold),
                "stage1_kept": int(len(filter_cam_id)),
                "total_cameras": int(len(all_cam_id)),
            }

            print(
                f"[DEBUG] camera distance stats | avg_all={avg_all_dist:.4f}, "
                f"avg_filtered={avg_filtered_dist:.4f}, "
                f"slider={float(self.camera_gap_slider.value):.4f}, "
                f"gap_threshold={gap_threshold:.4f}, "
                f"stage1_kept={len(filter_cam_id)}/{len(all_cam_id)}"
            )
            return gap_threshold, debug_stats

        gap_threshold, camera_distance_stats = get_gap_threshold()
        self.last_preprocess_camera_distance_stats = camera_distance_stats
        cam_id = list(range(len(self.colmap_cameras)))

        if self.sort_camera_center.value:
            cam_id = sorted(
                cam_id,
                key=lambda idx: np.linalg.norm(self.camera_poses[idx]["position"]),
            )

        if len(self.camera_poses) > 0:
            all_colmap_centers = np.asarray(
                [pose["position"] for pose in self.camera_poses], dtype=np.float32
            )
            center_cam = all_colmap_centers.mean(axis=0, keepdims=True)
        else:
            center_cam = np.array([0, 0, 0], dtype=np.float32)[None]

        if self.start_from_center.value and len(cam_id) > 0:
            nearest_id = cam_id[0]
            for candidate_id in cam_id:
                center = np.array(self.camera_poses[candidate_id]["position"])
                if np.all(
                    np.linalg.norm(center_cam - center, axis=1) > gap_threshold
                ):
                    nearest_id = candidate_id
                    break
            nearest_idx = cam_id.index(nearest_id)
            cam_id = cam_id[nearest_idx:] + cam_id[:nearest_idx]

        filtered_cam_id = []
        for candidate_id in cam_id:
            center = np.array(self.camera_poses[candidate_id]["position"])
            too_near_center_cam = not np.all(
                np.linalg.norm(center_cam - center, axis=1) > gap_threshold
            )
            too_close_to_object = self.is_camera_center_too_close(center)
            if too_near_center_cam or too_close_to_object:
                continue
            filtered_cam_id.append(candidate_id)
            center_cam = np.concatenate((center_cam, center[None]), axis=0)

        if self.enable_camera_center.value:
            filtered_cam_id.insert(0, -1)
        return filtered_cam_id

    # ------------------------------------------------------------------ #
    # Color projection / propagation                                     #
    # ------------------------------------------------------------------ #

    def color_update_proj(
        self,
        camera_center_path,
        camera_rotation,
        styled_img_path,
        point_mask,
        upscale=1,
        color_gaussian=True,
    ):
        camera_center = np.load(camera_center_path)

        pano_styled = cv2.imread(styled_img_path)
        pano_styled = cv2.resize(
            pano_styled,
            (pano_styled.shape[1] * upscale, pano_styled.shape[0] * upscale),
        )
        height, width = pano_styled.shape[:2]
        pano_styled = cv2.cvtColor(pano_styled, cv2.COLOR_BGR2RGB)

        point_centers = self.gaussian_model.get_xyz
        point_centers = point_centers - torch.from_numpy(camera_center).to(
            point_centers
        )

        # Offset record (kept identical to viewer.py).
        ### Playroom: -90 0 90 ###
        ### Drjohnson: -90 0 180 ###
        theta_offset, phi_offset, gamma_offset = -90, 0, 90
        R = camera_rotation @ vtf.SO3.from_rpy_radians(
            math.radians(phi_offset),
            math.radians(theta_offset),
            math.radians(gamma_offset),
        ).as_matrix()
        rotated_point_centers = torch.matmul(
            point_centers.cpu(), torch.from_numpy(R).float()
        ).to(point_centers)

        pano_coords = direction_to_pano_coord(rotated_point_centers)

        img_coords = (
            torch.round(
                pano_to_img_coord(pano_coords, width=width, height=height)
            )
            .int()
            .detach()
            .cpu()
        )
        point_rgbs = pano_styled[img_coords[..., 0], img_coords[..., 1]]

        pano_vis = point_mask

        fused_color = RGB2SH((torch.tensor(point_rgbs).float() / 255.0).cuda())
        features = (
            torch.zeros(
                (
                    fused_color.shape[0],
                    3,
                    (self.gaussian_model.max_sh_degree + 1) ** 2,
                )
            )
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        _features_dc = torch.nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        _features_rest = torch.nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )

        if color_gaussian:
            with torch.no_grad():
                self.gaussian_model._features_dc[pano_vis] = _features_dc[pano_vis]
                self.gaussian_model._features_rest[pano_vis] = _features_rest[
                    pano_vis
                ]
        return (
            features[:, :, 0:1].transpose(1, 2),
            features[:, :, 1:].transpose(1, 2),
        )

    def hidden_color_propogation(self, point_mask_path):
        # Typo preserved on purpose to mirror viewer.py (see plan handover).
        # print("Propogating Color...")
        point_mask = np.load(point_mask_path).astype(np.bool_)

        point_centers = self.gaussian_model.get_xyz
        _, nn_idx = K_nearest_neighbors(
            point_centers[point_mask],
            self.knn_number.value,
            point_centers[~point_mask],
        )
        nn_feat_dc = self.gaussian_model._features_dc[point_mask][nn_idx]
        nn_feat_rest = self.gaussian_model._features_rest[point_mask][nn_idx]

        mean_feat_dc = nn_feat_dc.mean(axis=1)
        mean_feat_rest = nn_feat_rest.mean(axis=1)
        if self.knn_number.value == 1:
            mean_feat_dc = mean_feat_dc.unsqueeze(1)
            mean_feat_rest = mean_feat_rest.unsqueeze(1)
        with torch.no_grad():
            self.gaussian_model._features_dc[~point_mask] = mean_feat_dc
            self.gaussian_model._features_rest[~point_mask] = mean_feat_rest

        self.update_client()
        print("Propogating Color Success...")

    # ------------------------------------------------------------------ #
    # Preprocess: render panoramas + visibility masks                    #
    # ------------------------------------------------------------------ #

    def _select_cam_ids_with_cap(self) -> List[int]:
        """Run candidate selection, optionally retrying with a larger
        ``camera_gap`` until the result fits ``max_candidates``.

        ``camera_gap`` is mutated on ``self.camera_gap_slider`` so the
        value persists for any later step that re-reads it.
        """
        max_n = int(self.cfg.preprocess.max_candidates)
        cam_id = self._get_preprocess_candidate_cam_ids()
        print(
            f"Candidate Length: {len(cam_id)} | "
            f"camera_gap={self.camera_gap_slider.value:.4f} | List: {cam_id}"
        )
        if max_n <= 0 or len(cam_id) <= max_n:
            return cam_id

        gap_step = float(self.cfg.preprocess.max_candidates_gap_step)
        max_iters = int(self.cfg.preprocess.max_candidates_max_iters)
        if gap_step <= 0:
            print(
                "[WARN] max_candidates_gap_step <= 0; cannot enforce "
                "max_candidates cap. Returning current selection."
            )
            return cam_id

        for it in range(max_iters):
            self.camera_gap_slider.value = (
                float(self.camera_gap_slider.value) + gap_step
            )
            cam_id = self._get_preprocess_candidate_cam_ids()
            print(
                f"[INFO] retry {it + 1}/{max_iters}: "
                f"camera_gap={self.camera_gap_slider.value:.4f}, "
                f"Candidate Length: {len(cam_id)} | List: {cam_id}"
            )
            if len(cam_id) <= max_n:
                print(
                    f"[INFO] candidate cap satisfied "
                    f"({len(cam_id)} <= {max_n}) at "
                    f"camera_gap={self.camera_gap_slider.value:.4f}"
                )
                return cam_id

        print(
            f"[WARN] max_candidates_max_iters={max_iters} reached, "
            f"final length {len(cam_id)} still > {max_n}"
        )
        return cam_id

    def run_preprocess(self) -> List[int]:
        cam_id = self._select_cam_ids_with_cap()

        save_dir = self.prep_dir
        os.makedirs(save_dir, exist_ok=True)
        preprocess_json = {
            "cam_id": cam_id,
            "save_dir": save_dir,
            "camera_distance_stats": getattr(
                self, "last_preprocess_camera_distance_stats", None
            ),
        }
        with open(
            os.path.join(save_dir, "preprocess.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(preprocess_json, f, ensure_ascii=False, indent=4)

        mask_thres = float(self.cfg.preprocess.mask_threshold)
        pano_res = int(self.cfg.preprocess.pano_res)

        all_point_mask = np.zeros(
            self.gaussian_model.get_xyz.shape[0], dtype=np.uint8
        )
        with torch.no_grad():
            for idx, cam in enumerate(tqdm(cam_id)):
                if os.path.exists(
                    os.path.join(save_dir, f"cam_{cam}", "pano_mask_all.npy")
                ):
                    continue
                if cam == -1:
                    camera_center = np.array([0, 0, 0])
                    camera_rotation = np.diag(np.ones((3)))
                else:
                    pose = self.camera_poses[cam]
                    camera_center = np.array(pose["position"])
                    if self.camera_focus.value:
                        camera_rotation = np.array(pose["rotation"])
                    else:
                        camera_rotation = np.diag(np.ones((3)))

                image = self.check_and_render_panorama(
                    camera_center,
                    camera_rotation,
                    save_dir=os.path.join(save_dir, f"cam_{cam}"),
                    res=pano_res,
                )
                if idx != 0:
                    image_mask = self.check_and_render_panorama(
                        camera_center,
                        camera_rotation,
                        mask=all_point_mask,
                        save_dir=os.path.join(save_dir, f"cam_{cam}", "mask"),
                        res=pano_res,
                    )
                    mask = (np.array(image_mask).sum(axis=2) / 3 / 255) > mask_thres
                    Image.fromarray((mask.astype(np.uint8) * 255)).save(
                        os.path.join(save_dir, f"cam_{cam}", "pano_mask.png")
                    )
                    masked_image = np.array(image.copy())
                    masked_image[~mask] = 0
                    Image.fromarray(masked_image.astype(np.uint8)).save(
                        os.path.join(save_dir, f"cam_{cam}", "pano_img_masked.png")
                    )
                    mask = cv2.erode(
                        mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3
                    )
                    Image.fromarray((mask.astype(np.uint8) * 255)).save(
                        os.path.join(save_dir, f"cam_{cam}", "pano_mask_eroded.png")
                    )
                else:
                    os.makedirs(
                        os.path.join(save_dir, f"cam_{cam}", "mask"), exist_ok=True
                    )
                    Image.fromarray(
                        (np.zeros_like(image).astype(np.uint8) * 255)
                    ).save(os.path.join(save_dir, f"cam_{cam}", "pano_mask.png"))
                    Image.fromarray(
                        (np.zeros_like(image).astype(np.uint8) * 255)
                    ).save(
                        os.path.join(save_dir, f"cam_{cam}", "pano_mask_eroded.png")
                    )
                    Image.fromarray(
                        (np.zeros_like(image).astype(np.uint8) * 255)
                    ).save(
                        os.path.join(save_dir, f"cam_{cam}", "mask", "pano_img.png")
                    )

                cur_point_map, cur_point_mask = get_hidden_point_mask(
                    self.gaussian_model.get_xyz, camera_center
                )
                np.save(
                    os.path.join(save_dir, f"cam_{cam}", "pano_mask.npy"),
                    cur_point_mask,
                )
                all_point_mask = all_point_mask | cur_point_mask
                np.save(
                    os.path.join(save_dir, f"cam_{cam}", "pano_mask_all.npy"),
                    all_point_mask,
                )

        print("Preprocess success...")
        return cam_id

    # ------------------------------------------------------------------ #
    # Stylization core                                                   #
    # ------------------------------------------------------------------ #

    def color_update(self, cam_id, save_dir):
        # print("Reading and processing images...")

        res = int(self.cfg.preprocess.pano_res)
        train_res_default = int(self.cfg.stylization.train_res)
        num_sample_views = int(self.cfg.stylization.num_sample_views)
        upscale = int(self.cfg.stylization.color_upscale)
        step_base = int(self.cfg.stylization.step_base)
        step_per_cam = int(self.cfg.stylization.step_per_cam)
        missing_rate_threshold = float(
            self.cfg.stylization.missing_rate_threshold
        )
        first_cam_strength = float(self.cfg.stylization.first_cam_strength)
        subsequent_cam_strength = float(
            self.cfg.stylization.subsequent_cam_strength
        )

        def get_pano_imgs_tensor(equ):
            images = []
            for p in PERS_PARAMS:
                image = equ.GetPerspective(p[0], p[1], p[2], res, res)
                image = np.clip(image, 0, 255)
                images.append(image[..., ::-1])
            image_tensor = torch.tensor(images).permute(0, 3, 1, 2).float() / 255.0
            return image_tensor

        def train_pano(
            styled_imgs,
            feat_dc,
            feat_rest,
            steps,
            random_flag=False,
            train_res=None,
            num_sample_views=6,
        ):
            if train_res is None:
                train_res = res

            def calculate_total_variation_loss_spherical(x, p=1, reduction="mean"):
                B, C, H, W = x.shape
                device = x.device
                dtype = x.dtype

                theta_w = (
                    torch.arange(H, dtype=dtype, device=device) + 0.5
                ) * (math.pi / H)
                weight_w = torch.sin(theta_w).view(1, 1, H, 1)
                diff_w = torch.roll(x, shifts=-1, dims=-1) - x

                theta_h = (
                    torch.arange(H - 1, dtype=dtype, device=device) + 1.0
                ) * (math.pi / H)
                weight_h = torch.sin(theta_h).view(1, 1, H - 1, 1)
                diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]

                if p == 1:
                    loss_w = torch.abs(diff_w) * weight_w
                    loss_h = torch.abs(diff_h) * weight_h
                elif p == 2:
                    loss_w = torch.pow(diff_w, 2) * weight_w
                    loss_h = torch.pow(diff_h, 2) * weight_h
                else:
                    raise ValueError("Parameter 'p' must be 1 or 2.")

                total_loss = torch.sum(loss_w) + torch.sum(loss_h)
                if reduction == "mean":
                    total_loss = total_loss / (B * C * H * W)
                return total_loss

            train_steps = steps

            point_centers = self.gaussian_model.get_xyz
            _, nn_idx = K_nearest_neighbors(
                point_centers[all_point_mask],
                self.knn_number.value,
                point_centers[~all_point_mask],
            )

            for step in tqdm(range(train_steps)):
                self.gaussian_model.update_learning_rate(step + steps)
                # ViewerRenderer caches means2D/shs/opacity/etc. as tensor
                # references on first construction. The cached `means2D` in
                # particular is a `requires_grad=True` zero tensor whose
                # autograd graph would otherwise persist across iterations
                # and trigger "backward through the graph a second time"
                # on the next step. Rebuild the cache every step so every
                # render starts from a fresh autograd graph that lives only
                # for the current backward.
                self.viewer_renderer.update_pc_features()

                if random_flag and step >= 100:
                    random_img = random.sample(styled_imgs, 1)[0]
                else:
                    random_img = styled_imgs[-1]

                camera_center, camera_rotation, equ_styled, equ_mask = (
                    random_img["camera_center"],
                    random_img["camera_rotation"],
                    random_img["styled_img_tensor"],
                    random_img["pano_mask_tensor"],
                )

                if train_res != res:
                    equ_styled = torch.nn.functional.interpolate(
                        equ_styled,
                        size=(train_res, train_res),
                        mode="bilinear",
                        align_corners=False,
                    )
                    equ_mask = torch.nn.functional.interpolate(
                        equ_mask, size=(train_res, train_res), mode="nearest"
                    )

                color_loss = 0
                ssim_loss = 0
                tv_loss = 0
                sampled_indices = random.sample(
                    range(len(PERS_PARAMS)), min(num_sample_views, len(PERS_PARAMS))
                )
                for idx in sampled_indices:
                    fov, theta, phi = PERS_PARAMS[idx]
                    R = camera_rotation @ vtf.SO3.from_rpy_radians(
                        math.radians(phi),
                        math.radians(theta),
                        math.radians(0),
                    ).as_matrix()
                    T = [0, 0, 0]
                    trans = camera_center

                    cam = Simple_Camera(
                        0,
                        R,
                        T,
                        math.radians(90),
                        math.radians(90),
                        train_res,
                        train_res,
                        "",
                        0,
                        trans=trans,
                    )

                    render_params = self._render_params(
                        override_color=SH2RGB(
                            self.gaussian_model._features_dc.squeeze()
                        )
                    )

                    results = self.viewer_renderer.render_viewer(
                        cam, **render_params
                    )
                    rendered = results["render"].permute(1, 2, 0)

                    color_loss += l1_loss(
                        equ_mask[idx].to(self.device)
                        * rendered.permute(2, 0, 1)[None],
                        equ_mask[idx].to(self.device)
                        * equ_styled[idx][None].to(self.device),
                    )
                    ssim_loss += 1.0 - ssim(
                        equ_mask[idx].to(self.device)
                        * rendered.permute(2, 0, 1)[None],
                        equ_mask[idx].to(self.device)
                        * equ_styled[idx][None].to(self.device),
                    )
                    tv_loss += calculate_total_variation_loss_spherical(
                        rendered.permute(2, 0, 1)[None]
                    )
                    torch.cuda.empty_cache()

                project_loss = l1_loss(
                    self.gaussian_model._features_dc[all_point_mask],
                    feat_dc[all_point_mask],
                ) + l1_loss(
                    self.gaussian_model._features_rest[all_point_mask],
                    feat_rest[all_point_mask],
                )

                nn_feat_dc = self.gaussian_model._features_dc[all_point_mask][
                    nn_idx
                ]
                nn_feat_rest = self.gaussian_model._features_rest[all_point_mask][
                    nn_idx
                ]
                mean_feat_dc = nn_feat_dc.mean(axis=1)
                mean_feat_rest = nn_feat_rest.mean(axis=1)
                if self.knn_number.value == 1:
                    mean_feat_dc = mean_feat_dc.unsqueeze(1)
                    mean_feat_rest = mean_feat_rest.unsqueeze(1)
                propagate_loss = l1_loss(
                    self.gaussian_model._features_dc[~all_point_mask], mean_feat_dc
                ) + l1_loss(
                    self.gaussian_model._features_rest[~all_point_mask],
                    mean_feat_rest,
                )

                color_term = 0.8 * color_loss / num_sample_views
                ssim_term = 0.2 * ssim_loss / num_sample_views
                tv_term = 1e-3 * tv_loss / num_sample_views
                propagate_term = 1.0 * propagate_loss
                project_term = 1.0 * project_loss

                loss = (
                    color_term
                    + ssim_term
                    + propagate_term
                    + project_term
                    + tv_term
                )

                if step % 20 == 0:
                    print(
                        "total_loss:", loss.data,
                        "\t color_loss:", color_term.data,
                        "\t ssim_loss:", ssim_term.data,
                        "\t tv_loss:", tv_term.data,
                        "\t propagate_loss:", propagate_term.data,
                        "\t project_loss:", project_term.data,
                    )
                    self.update_client()

                torch.cuda.empty_cache()
                loss.backward()
                self.gaussian_model.optimizer.step()
                self.gaussian_model.optimizer.zero_grad(set_to_none=True)

        opt = OptimizationParams(
            parser=ArgumentParser(description="Training script parameters"),
            max_steps=self.train_steps.value,
            color_lr_scaler=self.color_lr_scaler.value,
        )
        opt = OmegaConf.create(vars(opt))
        self.gaussian_model.max_radii2D = torch.zeros(
            (self.gaussian_model.get_xyz.shape[0]), device=self.device
        )

        styled_imgs = []
        feat_dc_visible, feat_rest_visible = None, None
        total_cams = len(cam_id)
        for i in range(total_cams):
            idx = cam_id[i]
            camera_center = np.load(
                os.path.join(save_dir, f"cam_{idx}", "camera_center.npy")
            )
            camera_rotation = np.load(
                os.path.join(save_dir, f"cam_{idx}", "camera_rotation.npy")
            )
            pano_mask = cv2.imread(
                os.path.join(save_dir, f"cam_{idx}", "pano_mask.png")
            )
            prev_all_point_mask = np.load(
                os.path.join(
                    save_dir, f"cam_{cam_id[i - 1]}", "pano_mask_all.npy"
                )
            ).astype(np.bool_)
            cur_point_mask = np.load(
                os.path.join(save_dir, f"cam_{idx}", "pano_mask.npy")
            ).astype(np.bool_)
            all_point_mask = np.load(
                os.path.join(save_dir, f"cam_{idx}", "pano_mask_all.npy")
            ).astype(np.bool_)

            r_image = self.check_and_render_panorama(
                camera_center,
                camera_rotation,
                save_dir=os.path.join(save_dir, f"cam_{idx}", "styled"),
            )

            refined_path = os.path.join(
                save_dir, f"cam_{idx}", "styled", "pano_styled_refined.png"
            )
            alt_refined_path = os.path.join(
                save_dir, f"cam_{idx}", "pano_styled_refined.png"
            )
            if os.path.exists(refined_path):
                styled_img = cv2.imread(refined_path)
            elif os.path.exists(alt_refined_path):
                styled_img = cv2.imread(alt_refined_path)
            else:
                missing_rate = 1 - (
                    pano_mask[..., 0].astype(np.bool_).sum()
                    / (pano_mask.shape[0] * pano_mask.shape[1])
                )
                print(f"Pixel Missing Rate: {missing_rate}")
                if i == 0 or missing_rate > missing_rate_threshold:
                    print("Too many pixel missing, regenerating in all....")
                    adain_img = generate_adain(
                        content_path=os.path.join(
                            save_dir, f"cam_{idx}", "pano_img.png"
                        ),
                        style_path=self.style_img,
                        save_path=os.path.join(
                            save_dir, f"cam_{idx}", "styled", "pano_adain.png"
                        ),
                    )
                    styled_img = generate_image(
                        prompt=self.prompt_text.value,
                        style_img_path=self.style_img,
                        input_img_path=os.path.join(
                            save_dir, f"cam_{idx}", "styled", "pano_adain.png"
                        ),
                        ref_img_path=os.path.join(
                            save_dir, f"cam_{idx}", "pano_img.png"
                        ),
                        strength=first_cam_strength,
                    )
                    styled_img.save(refined_path)
                else:
                    adain_img = generate_adain(
                        content_path=os.path.join(
                            save_dir, f"cam_{idx}", "pano_img.png"
                        ),
                        style_path=self.style_img,
                        save_path=os.path.join(
                            save_dir, f"cam_{idx}", "styled", "pano_adain.png"
                        ),
                    )

                    adain_img_np = np.array(adain_img)
                    pano_img_np = cv2.imread(
                        os.path.join(
                            save_dir, f"cam_{idx}", "styled", "pano_img.png"
                        )
                    )
                    pano_img_np = cv2.cvtColor(pano_img_np, cv2.COLOR_BGR2RGB)
                    pano_mask_np = cv2.imread(
                        os.path.join(save_dir, f"cam_{idx}", "pano_mask.png"),
                        cv2.IMREAD_GRAYSCALE,
                    ).astype(np.bool_)
                    fused_img_np = pano_img_np.copy()
                    fused_img_np[~pano_mask_np] = adain_img_np[~pano_mask_np]
                    fused_img = Image.fromarray(fused_img_np)
                    fused_img.save(
                        os.path.join(
                            save_dir, f"cam_{idx}", "styled", "pano_fused.png"
                        )
                    )

                    styled_img, inpainted_img = generate_image(
                        prompt=self.prompt_text.value,
                        style_img_path=self.style_img,
                        input_img_path=os.path.join(
                            save_dir, f"cam_{idx}", "styled", "pano_img.png"
                        ),
                        mask_img_path=os.path.join(
                            save_dir, f"cam_{idx}", "pano_mask.png"
                        ),
                        ref_img_path=os.path.join(
                            save_dir, f"cam_{idx}", "pano_img.png"
                        ),
                        strength=subsequent_cam_strength,
                    )
                    styled_img.save(refined_path)
                    inpainted_img.save(
                        os.path.join(
                            save_dir,
                            f"cam_{idx}",
                            "styled",
                            "pano_styled_inpainted.png",
                        )
                    )

                styled_img = cv2.imread(refined_path)

            # print("Init Styled Gaussian...")
            # print(f"Before save: {self.get_gpu_memory_usage()}")
            cam_mask = (
                (all_point_mask != prev_all_point_mask) if i != 0 else all_point_mask
            )
            _feat_dc_visible, _feat_rest_visible = self.color_update_proj(
                camera_center_path=os.path.join(
                    save_dir, f"cam_{idx}", "camera_center.npy"
                ),
                camera_rotation=camera_rotation,
                styled_img_path=os.path.join(
                    save_dir, f"cam_{idx}", "styled", "pano_styled_refined.png"
                ),
                point_mask=cam_mask,
                upscale=upscale,
                color_gaussian=True,
            )
            # Detach the projection-derived features: they are reused as a
            # fixed L1 target inside train_pano's project_loss across many
            # backward passes, so they must not carry their own autograd
            # graph (otherwise the second backward fails).
            _feat_dc_visible = _feat_dc_visible.detach()
            _feat_rest_visible = _feat_rest_visible.detach()
            if feat_dc_visible is None:
                feat_dc_visible = _feat_dc_visible
                feat_rest_visible = _feat_rest_visible
            else:
                feat_dc_visible[cam_mask] = _feat_dc_visible[cam_mask]
                feat_rest_visible[cam_mask] = _feat_rest_visible[cam_mask]

            self.viewer_renderer.update_pc_features()

            self.hidden_color_propogation(
                os.path.join(save_dir, f"cam_{idx}", "pano_mask_all.npy")
            )
            self.viewer_renderer.update_pc_features()

            styled_imgs.append(
                {
                    "cam_id": idx,
                    "camera_center": camera_center,
                    "camera_rotation": camera_rotation,
                    "point_mask": cur_point_mask,
                    "all_point_mask": all_point_mask,
                    "cam_mask": cam_mask,
                    "styled_img_tensor": get_pano_imgs_tensor(
                        E2P.Equirectangular(np.array(styled_img))
                    ),
                    "pano_mask_tensor": get_pano_imgs_tensor(
                        E2P.Equirectangular(255 - pano_mask)
                    ),
                }
            )

            # print(f"After save: {self.get_gpu_memory_usage()}")
            torch.cuda.empty_cache()

            scene_styled_path = os.path.join(
                save_dir, f"cam_{idx}", "styled", "scene_styled.ply"
            )
            if not os.path.exists(scene_styled_path):
                print(f"Training scene in cam_{idx}...")
                self.gaussian_model.training_setup(opt)
                train_pano(
                    styled_imgs,
                    feat_dc_visible,
                    feat_rest_visible,
                    step_base + step_per_cam * i,
                    random_flag=True,
                    train_res=train_res_default,
                    num_sample_views=num_sample_views,
                )
                self.gaussian_model.save_ply(scene_styled_path)
            else:
                print(f"Loading {scene_styled_path}")
                self.gaussian_model.load_ply(scene_styled_path)

            r_image = self.check_and_render_panorama(
                camera_center,
                camera_rotation,
                save_dir=os.path.join(save_dir, f"cam_{idx}", "after_styled"),
            )

            self.update_client()

    # ------------------------------------------------------------------ #
    # Top-level driver                                                   #
    # ------------------------------------------------------------------ #

    def run(self):
        cam_id = self.run_preprocess()
        self.gaussian_model.active_sh_degree = int(
            self.cfg.stylization.active_sh_degree
        )
        self.color_update(cam_id, self.prep_dir)
        self.update_client()
        print("update success.")


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def _parse_background_color(values):
    if len(values) == 1 and isinstance(values[0], str):
        if values[0] == "white":
            return (1.0, 1.0, 1.0)
        if values[0] == "black":
            return (0.0, 0.0, 0.0)
        return (0.5, 0.5, 0.5)
    return tuple(float(v) for v in values)


def main():
    parser = ArgumentParser(description="Headless preprocess + stylization runner")
    parser.add_argument("model_path", type=str, help="PLY file or model dir")
    parser.add_argument("--source_path", "-s", type=str, required=True)
    parser.add_argument("--cameras_json", "--cameras-json", type=str, required=True)
    parser.add_argument("--style_img", type=str, required=True)
    parser.add_argument("--prep_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument(
        "--background_color",
        "-b",
        type=str,
        nargs="+",
        default=["gray"],
        help="e.g. white, gray, black, '0.5 0.5 0.5'",
    )
    parser.add_argument("--sh_degree", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional OmegaConf YAML to override DEFAULT_CONFIG",
    )
    parser.add_argument(
        "--float32_matmul_precision",
        "--fp",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for the panoramic training loop",
    )
    args = parser.parse_args()

    if args.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(args.float32_matmul_precision)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    print("[INFO] effective config:")
    print(OmegaConf.to_yaml(cfg))

    pipeline = StylizationPipeline(
        cfg=cfg,
        model_path=args.model_path,
        source_path=args.source_path,
        cameras_json=args.cameras_json,
        style_img=args.style_img,
        prep_dir=args.prep_dir,
        prompt=args.prompt,
        background_color=_parse_background_color(args.background_color),
        sh_degree=args.sh_degree,
        iterations=args.iterations,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
