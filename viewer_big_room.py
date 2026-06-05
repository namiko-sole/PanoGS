import os
import sys 
import math
import glob
import shutil
import time
import json
import torch
import numpy as np
import trimesh
import viser
import viser.transforms as vtf
import threading
import warnings 
warnings.filterwarnings('ignore')
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '2d_gaussian_splatting'))

from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple, Literal, List, Dict, Set, Optional
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

from arguments import ModelParams, PipelineParams, get_combined_args
from internal.viewer import ViewerRenderer, ClientThread
from internal.viewer import GaussianModelforViewer as GaussianModel
from internal.viewer.ui import RenderPanel, TransformPanel, EditPanel

from PIL import Image, ImageChops
import cv2
# from utils.sh_utils import RGB2SH, SH2RGB, eval_sh
# from scene.cameras import Simple_Camera as Camera
from internal.utils.sh_utils import RGB2SH, SH2RGB, eval_sh
from internal.cameras.cameras import Cameras

from internal.utils.graphics_utils import fov2focal
from internal.utils.pano_utils import get_depth_distort, get_camera_center, camera_position, SobelOperator, RGB2Gray

import internal.utils.equirec.Equirec2Perspec as E2P
import internal.utils.equirec.multi_Perspec2Equirec as m_P2E
from scene.dataset_readers import sceneLoadTypeCallbacks

from internal.utils.depth_proj import DepthSplatting
import random
from tqdm import tqdm

from arguments import (
    PipelineParams,
    OptimizationParams,
)
from omegaconf import OmegaConf
from scene.cameras import Simple_Camera
from scene import Scene
from sdwebui_api import inpaint
from internal.utils.perceptual import PerceptualLoss
from diffusers_inference import generate_image

import torchvision
import open3d as o3d
from internal.utils.point_cloud import get_hidden_point_mask
from internal.utils.knn import K_nearest_neighbors
from internal.utils import nnfm_utils
from internal.utils.loss_utils import *
import torchvision.transforms as transforms
from internal.utils.adain_utils.adain_api import generate_adain

DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE = "@Direct"

ROOM_PALETTE = [
    (231, 76, 60),
    (52, 152, 219),
    (46, 204, 113),
    (241, 196, 15),
    (155, 89, 182),
    (230, 126, 34),
    (26, 188, 156),
    (149, 165, 166),
    (192, 57, 43),
    (41, 128, 185),
    (39, 174, 96),
    (243, 156, 18),
]
UNASSIGNED_COLOR = (170, 170, 170)
SELECTED_COLOR = (255, 255, 0)

class Viewer:
    def __init__(
            self,
            args,
            model_path: str,
            source_path: str = '',
            host: str = "0.0.0.0",
            port: int = 8080,
            background_color: Tuple = (0.5, 0.5, 0.5),
            image_format: Literal["jpeg", "png"] = "jpeg",
            reorient: Literal["auto", "enable", "disable"] = "auto",
            sh_degree: int = 0,
            enable_transform: bool = False,
            show_cameras: bool = False,
            cameras_json: str = None,
            up: list = None,
            default_camera_position: List = None,
            default_camera_look_at: List = None,
            no_edit_panel: bool = False,
            no_render_panel: bool = False,
            iterations: int=30000,
            crop_box_size: float=16.0,
            from_direct_path: str = None, 
            is_training: bool = False,
            style_img:str = None,
            prep_dir:str = None,
    ):
        self.render_type_name = {
            "RGB": 'render', 
            "Edge": 'edge',
            "Alpha": 'rend_alpha', 
            "Normal": 'rend_normal', 
            "View-Normal": 'view_normal',
            "Depth": 'depth',
            "Depth-Distort": 'rend_dist',
            "Depth-to-Normal": 'surf_normal',
            "Depth-to-Curvature": 'curvature',
            "None": 'render',
        }
        self.args = args
        self.model_path = model_path
        self.source_path = source_path
        self.cameras_json = os.path.join(self.model_path, "cameras.json") if cameras_json is None else cameras_json

        self.host = host
        self.port = port
        self.background_color = torch.tensor(background_color, dtype=torch.float32, device="cuda")
        self.image_format = image_format
        self.sh_degree = sh_degree
        self.enable_transform = enable_transform
        self.show_cameras = show_cameras
        self.crop_box_size = crop_box_size

        self.device = torch.device("cuda")
        self.total_device_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024 ** 2

        self.up_direction = np.asarray([0., 0., 1.])
        self.camera_center = np.asarray([0., 0., 0.])
        self.default_camera_position = default_camera_position
        self.default_camera_look_at = default_camera_look_at
        self.is_training = is_training
        self.show_edit_panel = ~no_edit_panel
        self.show_render_panel = ~no_render_panel

        self.style_img = style_img
        self.prep_dir = prep_dir

        self.manual_split_selected_ids: Set[int] = set()
        self.manual_split_assignments: Dict[int, int] = {}
        self.manual_split_last_assignments: Optional[Dict[int, int]] = None
        self.manual_split_room_point_masks: Dict[int, np.ndarray] = {}
        self.manual_split_point_room_ids: Optional[np.ndarray] = None
        self.manual_split_selected_point_mask: Optional[np.ndarray] = None
        self.manual_split_point_xyz_cache: Optional[np.ndarray] = None
        self.manual_split_selected_point_preview_backup_dc: Optional[torch.Tensor] = None
        self.manual_split_selected_point_preview_color: Tuple[int, int, int] = (255, 0, 255)
        self.camera_handles_by_idx: Dict[int, viser.CameraFrustumHandle] = {}
        self.camera_handle_names: Dict[int, str] = {}
        self.manual_split_rect_select_clients: Set[int] = set()
        self.json_camera_poses: List[dict] = []
        self.colmap_camera_poses: List[dict] = []
        self.camera_pose_source_active: str = "colmap"
        self.manual_split_color_backup_dc: Optional[torch.Tensor] = None
        self.manual_split_color_backup_rest: Optional[torch.Tensor] = None
        self.manual_split_is_applying: bool = False
        self.preprocess_test_room_groups: Dict[int, List[int]] = {}
        self.preprocess_test_selected_cam_ids: List[int] = []

        # Tunable weights for point-room assignment after manual camera split.
        self.room_assign_k_nearest: int = 2
        self.room_assign_support_expected: float = 2.0
        self.room_assign_support_penalty_lambda: float = 0.8
        self.room_assign_dist_weight: float = 1.0
        self.room_assign_center_weight: float = 0.15
        self.room_assign_color_weight: float = 0.35
        self.room_assign_support_reward_weight: float = 0.12
        self.room_assign_conf_margin_threshold: float = 0.08
        self.room_assign_graph_refine_enable: bool = True
        self.room_assign_graph_knn: int = 12
        self.room_assign_graph_lambda: float = 0.35
        self.room_assign_graph_sigma_x: float = 0.0
        self.room_assign_graph_sigma_c: float = 0.15
        self.room_assign_graph_iters: int = 6
        self.room_assign_graph_only_low_conf: bool = True
        self.room_assign_graph_low_conf_margin: float = 0.12
        self.room_assign_graph_use_normal_consistency: bool = True
        self.room_assign_graph_normal_weight: float = 0.8

        # init model & scene 
        self._init_models(iterations)
        self._init_scene_camera_transform(self.cameras_json, reorient, up)
        self._init_camera_poses(self.cameras_json)
        self.clients = {}
        
    def _init_models(self, iterations):
        # init gaussian model & renderer
        self.gaussian_model = GaussianModel(sh_degree=self.sh_degree)
        if not self.is_training:
            self.iteration = iterations
            self.ply_path = os.path.join(self.model_path, "point_cloud", f"iteration_{iterations}", "point_cloud.ply") if not self.model_path.lower().endswith('.ply') else self.model_path
            if not os.path.exists(self.ply_path):
                print(f'[Alert] there is no pointcloud in: {self.ply_path}')
                raise FileNotFoundError
            print(f'[INFO] ply path loaded from: {self.ply_path}')
            self.gaussian_model.load_ply(self.ply_path)
            print(f'[INFO] number of points: {self.gaussian_model._xyz.shape[0]}')
        self.viewer_renderer = ViewerRenderer(self.gaussian_model, self.background_color, not self.is_training)

    def _init_scene_camera_transform(self, cameras_json_path, mode, up):
        transform = torch.eye(4, dtype=torch.float)
        self.camera_transform = transform
        if mode == "disable" or not os.path.exists(cameras_json_path): return
        
        print(f"[Info] Load cameras from: {cameras_json_path}")
        with open(cameras_json_path, "r") as f:
            cameras = json.load(f)
        up_vector = torch.zeros(3)
        for cam in cameras:
            up_vector += torch.tensor(cam["rotation"])[:3, 1]
        up_vector = -up_vector / torch.linalg.norm(up_vector)
        print(f"[INFO] up vector = {up_vector}")
        self.up_direction = up_vector.numpy()

        if up is not None:
            transform = torch.eye(4, dtype=torch.float)
            up_vector = torch.tensor(up)
            up_vector = -up_vector / torch.linalg.norm(up_vector)
            self.up_direction = up_vector.numpy()

        self.camera_transform = transform

    def _build_colmap_fallback_camera_poses(self) -> List[dict]:
        camera_poses: List[dict] = []
        for idx, cam in enumerate(self.colmap_cameras):
            r_wc = np.asarray(cam.R, dtype=np.float32)
            t = np.asarray(cam.T, dtype=np.float32)
            cam_center = (-r_wc @ t).astype(np.float32)
            width = int(getattr(cam, "width", 1024))
            height = int(getattr(cam, "height", 1024))
            fx = float(fov2focal(float(cam.FovX), width)) if hasattr(cam, "FovX") else float(max(width, 1))
            camera_poses.append({
                "img_name": str(getattr(cam, "image_name", f"cam_{idx:04d}.png")),
                "rotation": r_wc.tolist(),
                "position": cam_center.tolist(),
                "width": width,
                "height": height,
                "fx": fx,
                "pose_source": "colmap_fallback",
            })
        return camera_poses

    def _set_active_camera_pose_source(self, source: str) -> str:
        if source == "json" and len(self.json_camera_poses) > 0:
            self.camera_poses = list(self.json_camera_poses)
            self.camera_pose_source_active = "json"
        else:
            self.camera_poses = list(self.colmap_camera_poses)
            self.camera_pose_source_active = "colmap"

        if len(self.camera_poses) > 0:
            self.camera_center = np.mean(np.asarray([i["position"] for i in self.camera_poses]), axis=0)
        return self.camera_pose_source_active

    def _init_camera_poses(self, cameras_json_path):
        col_scene = sceneLoadTypeCallbacks["Colmap"](self.source_path, None, False)
        self.scene = col_scene
        self.colmap_cameras = col_scene.train_cameras

        self.colmap_camera_poses = self._build_colmap_fallback_camera_poses()
        self.json_camera_poses = []

        if os.path.exists(cameras_json_path):
            with open(cameras_json_path, "r") as f:
                self.json_camera_poses = json.load(f)
            print(f"[INFO] load camera poses from {cameras_json_path}, count={len(self.json_camera_poses)}")
        else:
            print(f"[WARN] camera json not found: {cameras_json_path}, fallback to COLMAP train cameras")

        initial_source = "json" if len(self.json_camera_poses) > 0 else "colmap"
        self._set_active_camera_pose_source(initial_source)

    def _get_training_gaussians(self, new_gaussians):
        # slow and large gpu consumption
        self.gaussian_model._xyz = new_gaussians._xyz.clone().detach()
        self.gaussian_model._scaling = new_gaussians._scaling.clone().detach()
        self.gaussian_model._opacity = new_gaussians._opacity.clone().detach()
        self.gaussian_model._rotation = new_gaussians._rotation.clone().detach()
        self.gaussian_model._features_dc = new_gaussians._features_dc.clone().detach()
        self.gaussian_model._features_rest = new_gaussians._features_rest.clone().detach()
        self.viewer_renderer = ViewerRenderer(self.gaussian_model, self.background_color, self.is_training)

    def get_gpu_memory_usage(self):
        total_memory = torch.cuda.memory_allocated() + torch.cuda.memory_reserved() 
        return f"{total_memory / 1024 ** 2:.1f} / {self.total_device_memory:.1f} MB"

    @staticmethod
    def _room_color(room_id: int) -> Tuple[int, int, int]:
        return ROOM_PALETTE[room_id % len(ROOM_PALETTE)]

    def _manual_split_status_text(self) -> str:
        room_counts: Dict[int, int] = {}
        for room_id in self.manual_split_assignments.values():
            room_counts[room_id] = room_counts.get(room_id, 0) + 1
        if len(room_counts) == 0:
            room_msg = "none"
        else:
            room_msg = ", ".join([f"room_{rid}:{room_counts[rid]}" for rid in sorted(room_counts.keys())])

        selected_points = 0
        if self.manual_split_selected_point_mask is not None:
            selected_points = int(np.count_nonzero(self.manual_split_selected_point_mask))

        assigned_points = 0
        total_points = int(self.gaussian_model.get_xyz.shape[0]) if hasattr(self, "gaussian_model") else 0
        if self.manual_split_point_room_ids is not None:
            assigned_points = int(np.count_nonzero(self.manual_split_point_room_ids >= 0))

        return (
            f"Selected: {len(self.manual_split_selected_ids)} | Assigned: {len(self.manual_split_assignments)} / {len(self.camera_poses)} | "
            f"Rooms: {room_msg} | PointSel: {selected_points} | PointAssigned: {assigned_points}/{total_points}"
        )

    def _manual_split_visibility_dir(self) -> Optional[str]:
        if not self.prep_dir:
            return None
        split_dir = os.path.join(self.prep_dir, "split")
        visibility_dir = os.path.join(split_dir, "visibility")
        os.makedirs(visibility_dir, exist_ok=True)
        return visibility_dir

    def _get_camera_point_visibility_mask(self, cam_idx: int, point_xyz: torch.Tensor) -> np.ndarray:
        vis_mask = None
        visibility_dir = self._manual_split_visibility_dir()
        vis_path = None
        if visibility_dir is not None:
            vis_path = os.path.join(visibility_dir, f"cam_{cam_idx}_point_visible.npy")
            if os.path.exists(vis_path):
                vis_mask = np.load(vis_path).astype(np.bool_)

        if vis_mask is None:
            center = np.array(self.camera_poses[cam_idx]["position"], dtype=np.float32)
            vis_mask = self._compute_visibility_mask_spherical_zbuffer(
                point_xyz=point_xyz,
                camera_center=center,
            )
            if vis_path is not None:
                np.save(vis_path, vis_mask.astype(np.uint8))

        return vis_mask

    def _manual_split_select_points_from_selected_cameras(self, mode: str) -> int:
        if len(self.manual_split_selected_ids) == 0:
            raise ValueError("no selected cameras")

        point_xyz = self.gaussian_model.get_xyz.detach()
        hit_mask = np.zeros((point_xyz.shape[0],), dtype=np.bool_)
        for cam_idx in sorted(self.manual_split_selected_ids):
            if cam_idx < 0 or cam_idx >= len(self.camera_poses):
                continue
            hit_mask |= self._get_camera_point_visibility_mask(cam_idx, point_xyz)

        if self.manual_split_selected_point_mask is None or mode == "replace":
            self.manual_split_selected_point_mask = hit_mask
        elif mode == "append":
            self.manual_split_selected_point_mask |= hit_mask
        elif mode == "subtract":
            self.manual_split_selected_point_mask &= (~hit_mask)
        else:
            raise ValueError(f"unsupported point select mode: {mode}")

        self._refresh_manual_split_point_selection_preview()

        return int(np.count_nonzero(hit_mask))

    def _manual_split_assign_selected_points_to_room(self, room_id: int) -> int:
        if self.manual_split_selected_point_mask is None:
            raise ValueError("no selected points")

        point_count = int(self.gaussian_model.get_xyz.shape[0])
        if self.manual_split_point_room_ids is None:
            self.manual_split_point_room_ids = np.full((point_count,), -1, dtype=np.int32)

        if self.manual_split_point_room_ids.shape[0] != point_count:
            raise ValueError("point_room_ids size mismatch")

        selected = self.manual_split_selected_point_mask.astype(np.bool_)
        changed = int(np.count_nonzero(selected))
        self.manual_split_point_room_ids[selected] = int(room_id)
        self.manual_split_room_point_masks = self._build_room_point_masks(self.manual_split_point_room_ids)
        self._refresh_manual_split_point_selection_preview()
        return changed

    def _manual_split_unassign_selected_points(self) -> int:
        if self.manual_split_selected_point_mask is None:
            raise ValueError("no selected points")
        if self.manual_split_point_room_ids is None:
            return 0

        selected = self.manual_split_selected_point_mask.astype(np.bool_)
        changed = int(np.count_nonzero(selected & (self.manual_split_point_room_ids >= 0)))
        self.manual_split_point_room_ids[selected] = -1
        self.manual_split_room_point_masks = self._build_room_point_masks(self.manual_split_point_room_ids)
        self._refresh_manual_split_point_selection_preview()
        return changed

    def _clear_manual_split_point_selection_preview(self):
        if self.manual_split_selected_point_preview_backup_dc is None:
            return

        with torch.no_grad():
            self.gaussian_model._features_dc.copy_(self.manual_split_selected_point_preview_backup_dc)
        self.manual_split_selected_point_preview_backup_dc = None
        self.update_client()

    def _refresh_manual_split_point_selection_preview(self):
        if not hasattr(self, "manual_split_preview_selected_points"):
            return

        preview_enabled = bool(self.manual_split_preview_selected_points.value)
        has_selection = (
            self.manual_split_selected_point_mask is not None
            and bool(np.any(self.manual_split_selected_point_mask))
        )

        if (not preview_enabled) or (not has_selection):
            self._clear_manual_split_point_selection_preview()
            return

        # Always restore before re-applying highlight to avoid stacking edits.
        if self.manual_split_selected_point_preview_backup_dc is not None:
            with torch.no_grad():
                self.gaussian_model._features_dc.copy_(self.manual_split_selected_point_preview_backup_dc)

        with torch.no_grad():
            base_dc = self.gaussian_model._features_dc.detach().clone()
            current_rgb = SH2RGB(base_dc.squeeze()).detach().clone()
            mask_t = torch.from_numpy(self.manual_split_selected_point_mask.astype(np.bool_)).to(current_rgb.device)
            color_rgb = torch.tensor(self.manual_split_selected_point_preview_color, dtype=current_rgb.dtype, device=current_rgb.device) / 255.0
            current_rgb[mask_t] = color_rgb
            recolor_dc = RGB2SH(current_rgb).unsqueeze(1)

            self.manual_split_selected_point_preview_backup_dc = base_dc
            self.gaussian_model._features_dc.copy_(recolor_dc)

        self.update_client()

    def _save_manual_split_point_assignments(self):
        if self.prep_dir is None or self.prep_dir == "":
            raise ValueError("--prep_dir is empty, cannot save split data")
        if self.manual_split_point_room_ids is None:
            raise ValueError("No point-room assignment found")

        split_dir = os.path.join(self.prep_dir, "split")
        os.makedirs(split_dir, exist_ok=True)

        point_room_ids = self.manual_split_point_room_ids.astype(np.int32)
        self.manual_split_room_point_masks = self._build_room_point_masks(point_room_ids)
        np.save(os.path.join(split_dir, "point_room_ids.npy"), point_room_ids)

        room_point_counts = {
            str(room_id): int(mask.sum()) for room_id, mask in sorted(self.manual_split_room_point_masks.items(), key=lambda x: x[0])
        }

        summary = {
            "num_cameras_total": int(len(self.camera_poses)),
            "num_cameras_assigned": int(len(self.manual_split_assignments)),
            "num_points_total": int(point_room_ids.shape[0]),
            "num_points_assigned": int(np.count_nonzero(point_room_ids >= 0)),
            "room_point_counts": room_point_counts,
        }

        summary_path = os.path.join(split_dir, "manual_split_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                prev_summary = json.load(f)
            if isinstance(prev_summary, dict):
                prev_summary.update(summary)
                summary = prev_summary

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        for room_id, mask in self.manual_split_room_point_masks.items():
            np.save(os.path.join(split_dir, f"room_{room_id}_point_mask.npy"), mask.astype(np.uint8))

        print(f"[INFO] point-room assignments saved in {split_dir}")

    def _refresh_manual_split_camera_colors(self):
        if not hasattr(self, "manual_split_enable"):
            return

        for idx, handle in self.camera_handles_by_idx.items():
            if idx in self.manual_split_selected_ids:
                handle.color = SELECTED_COLOR
                continue
            room_id = self.manual_split_assignments.get(idx)
            if room_id is None:
                handle.color = UNASSIGNED_COLOR if self.manual_split_enable.value else (255, 255, 0)
            else:
                handle.color = self._room_color(room_id)

    @staticmethod
    def _project_world_to_screen01(
        point_world: np.ndarray,
        cam_wxyz: np.ndarray,
        cam_pos: np.ndarray,
        cam_fov: float,
        cam_aspect: float,
    ) -> Optional[np.ndarray]:
        t_camera_world = vtf.SE3.from_rotation_and_translation(
            vtf.SO3(cam_wxyz), cam_pos
        ).inverse()
        p_cam_h = t_camera_world.as_matrix() @ np.array([
            float(point_world[0]),
            float(point_world[1]),
            float(point_world[2]),
            1.0,
        ])
        p_cam = p_cam_h[:3]

        z = float(p_cam[2])
        if z <= 1e-6:
            return None

        tan_half = math.tan(float(cam_fov) * 0.5)
        if tan_half <= 1e-8:
            return None

        xy = p_cam[:2] / z
        xy /= tan_half
        xy[0] /= float(cam_aspect)
        xy01 = (1.0 + xy) * 0.5
        return np.array([xy01[0], xy01[1], z], dtype=np.float64)

    def _apply_manual_split_rect_select(
        self,
        client: viser.ClientHandle,
        screen_pos: Tuple[Tuple[float, float], Tuple[float, float]],
        mode: str,
    ) -> int:
        if len(self.camera_poses) == 0:
            return 0

        (x0, y0), (x1, y1) = screen_pos
        x_min, x_max = float(min(x0, x1)), float(max(x0, x1))
        y_min, y_max = float(min(y0, y1)), float(max(y0, y1))

        cam = client.camera
        cam_pos = np.asarray(cam.position, dtype=np.float64)
        cam_wxyz = np.asarray(cam.wxyz, dtype=np.float64)
        cam_fov = float(cam.fov)
        cam_aspect = float(cam.aspect)

        hits: Set[int] = set()
        camera_pose_transform = np.linalg.inv(self.camera_transform.cpu().numpy())
        for idx, camera in enumerate(self.camera_poses):
            c2w = np.eye(4)
            c2w[:3, :3] = np.asarray(camera["rotation"])
            c2w[:3, 3] = np.asarray(camera["position"])
            c2w[:3, 1:3] *= -1
            c2w = np.matmul(camera_pose_transform, c2w)
            center = c2w[:3, 3]

            proj = self._project_world_to_screen01(
                point_world=center,
                cam_wxyz=cam_wxyz,
                cam_pos=cam_pos,
                cam_fov=cam_fov,
                cam_aspect=cam_aspect,
            )
            if proj is None:
                continue
            if x_min <= proj[0] <= x_max and y_min <= proj[1] <= y_max:
                hits.add(idx)

        if mode == "replace":
            self.manual_split_selected_ids = hits
        else:
            self.manual_split_selected_ids |= hits

        self._refresh_manual_split_camera_colors()
        return len(hits)

    def _apply_manual_split_point_rect_select(
        self,
        client: viser.ClientHandle,
        screen_pos: Tuple[Tuple[float, float], Tuple[float, float]],
        mode: str,
    ) -> int:
        point_xyz = self.gaussian_model.get_xyz.detach()
        if point_xyz.shape[0] == 0:
            return 0

        # Cache CPU xyz to reduce repeated GPU->CPU transfer during drag-select.
        if (
            self.manual_split_point_xyz_cache is None
            or self.manual_split_point_xyz_cache.shape[0] != int(point_xyz.shape[0])
        ):
            self.manual_split_point_xyz_cache = point_xyz.detach().cpu().numpy().astype(np.float32)
        points_np = self.manual_split_point_xyz_cache

        (x0, y0), (x1, y1) = screen_pos
        x_min, x_max = float(min(x0, x1)), float(max(x0, x1))
        y_min, y_max = float(min(y0, y1)), float(max(y0, y1))

        cam = client.camera
        cam_pos = np.asarray(cam.position, dtype=np.float64)
        cam_wxyz = np.asarray(cam.wxyz, dtype=np.float64)
        cam_fov = float(cam.fov)
        cam_aspect = float(cam.aspect)

        tan_half = math.tan(cam_fov * 0.5)
        if tan_half <= 1e-8:
            return 0

        t_camera_world = vtf.SE3.from_rotation_and_translation(
            vtf.SO3(cam_wxyz), cam_pos
        ).inverse()
        mat = t_camera_world.as_matrix().astype(np.float64)

        ones = np.ones((points_np.shape[0], 1), dtype=np.float64)
        points_h = np.concatenate([points_np.astype(np.float64), ones], axis=1)
        p_cam = (mat @ points_h.T).T[:, :3]

        z = p_cam[:, 2]
        valid = z > 1e-6
        if not np.any(valid):
            return 0

        xy = np.zeros((p_cam.shape[0], 2), dtype=np.float64)
        xy[valid] = p_cam[valid, :2] / z[valid, None]
        xy /= tan_half
        xy[:, 0] /= max(cam_aspect, 1e-8)
        xy01 = (1.0 + xy) * 0.5

        in_rect = (
            valid
            & (xy01[:, 0] >= x_min)
            & (xy01[:, 0] <= x_max)
            & (xy01[:, 1] >= y_min)
            & (xy01[:, 1] <= y_max)
        )

        if self.manual_split_selected_point_mask is None or mode == "replace":
            self.manual_split_selected_point_mask = in_rect.copy()
        elif mode == "append":
            self.manual_split_selected_point_mask |= in_rect
        else:
            # For point selection target, allow subtract behavior even if UI mode is extended later.
            self.manual_split_selected_point_mask &= (~in_rect)

        self._refresh_manual_split_point_selection_preview()

        return int(np.count_nonzero(in_rect))

    def _bind_rect_select_for_client(self, client: viser.ClientHandle):
        if client.client_id in self.manual_split_rect_select_clients:
            return

        @client.scene.on_pointer_event(event_type="rect-select")
        def _rect_select(event: viser.ScenePointerEvent) -> None:
            if not self.manual_split_enable.value:
                return
            if not self.manual_split_drag_select_enabled.value:
                return
            if len(event.screen_pos) < 2:
                return

            target = self.manual_split_drag_select_target.value if hasattr(self, "manual_split_drag_select_target") else "camera"
            if target == "point":
                hit_count = self._apply_manual_split_point_rect_select(
                    client=event.client,
                    screen_pos=(event.screen_pos[0], event.screen_pos[1]),
                    mode=self.manual_split_select_mode.value,
                )
                print(f"[INFO] drag-select hit points: {hit_count}")
            else:
                hit_count = self._apply_manual_split_rect_select(
                    client=event.client,
                    screen_pos=(event.screen_pos[0], event.screen_pos[1]),
                    mode=self.manual_split_select_mode.value,
                )
                print(f"[INFO] drag-select hit cameras: {hit_count}")

            self.manual_split_status.value = self._manual_split_status_text()

        self.manual_split_rect_select_clients.add(client.client_id)

    def _update_rect_select_bindings(self, server: viser.ViserServer):
        enable_rect_select = bool(self.manual_split_enable.value and self.manual_split_drag_select_enabled.value)
        for client in server.get_clients().values():
            if enable_rect_select:
                self._bind_rect_select_for_client(client)
            else:
                if client.client_id in self.manual_split_rect_select_clients:
                    client.scene.remove_pointer_callback()
                    self.manual_split_rect_select_clients.remove(client.client_id)

    def _compute_visibility_mask_spherical_zbuffer(
        self,
        point_xyz: torch.Tensor,
        camera_center: np.ndarray,
        width: int = 512,
        height: int = 256,
        depth_eps: float = 1e-3,
    ) -> np.ndarray:
        center = torch.tensor(camera_center, dtype=point_xyz.dtype, device=point_xyz.device)
        rel = point_xyz - center[None, :]
        dist = torch.linalg.norm(rel, dim=1)
        valid = dist > 1e-8

        # Orientation-free spherical projection for fast per-viewpoint visibility.
        azimuth = torch.atan2(rel[:, 1], rel[:, 0])
        elev = torch.asin(torch.clamp(rel[:, 2] / torch.clamp(dist, min=1e-8), min=-1.0, max=1.0))

        u = ((azimuth + torch.pi) / (2.0 * torch.pi) * (width - 1)).long().clamp(0, width - 1)
        v = ((elev + torch.pi * 0.5) / torch.pi * (height - 1)).long().clamp(0, height - 1)
        linear_idx = v * width + u

        zbuf = torch.full((height * width,), float("inf"), dtype=dist.dtype, device=dist.device)
        valid_idx = linear_idx[valid]
        valid_dist = dist[valid]
        zbuf.scatter_reduce_(0, valid_idx, valid_dist, reduce="amin", include_self=True)

        min_depth = zbuf[linear_idx]
        visible = valid & (dist <= (min_depth + depth_eps))
        return visible.detach().cpu().numpy().astype(np.bool_)

    def _compute_point_room_assignments(self, assigned_camera_ids: List[int]) -> np.ndarray:
        if len(assigned_camera_ids) == 0:
            return np.full(self.gaussian_model.get_xyz.shape[0], -1, dtype=np.int32)

        split_dir = os.path.join(self.prep_dir, "split") if self.prep_dir else None
        visibility_dir = os.path.join(split_dir, "visibility") if split_dir is not None else None
        if visibility_dir is not None:
            os.makedirs(visibility_dir, exist_ok=True)

        point_xyz = self.gaussian_model.get_xyz.detach()
        cam_centers = []
        visibility_masks = []
        visibility_start = time.time()
        for cam_idx in assigned_camera_ids:
            center = np.array(self.camera_poses[cam_idx]["position"], dtype=np.float32)
            cam_centers.append(center)

            vis_mask = None
            if visibility_dir is not None:
                vis_path = os.path.join(visibility_dir, f"cam_{cam_idx}_point_visible.npy")
                if os.path.exists(vis_path):
                    vis_mask = np.load(vis_path).astype(np.bool_)

            if vis_mask is None:
                vis_mask = self._compute_visibility_mask_spherical_zbuffer(
                    point_xyz=point_xyz,
                    camera_center=center,
                )
                if visibility_dir is not None:
                    np.save(vis_path, vis_mask.astype(np.uint8))

            visibility_masks.append(vis_mask)
        print(
            f"[INFO] visibility by spherical z-buffer done: "
            f"cams={len(assigned_camera_ids)}, points={point_xyz.shape[0]}, "
            f"time={time.time() - visibility_start:.3f}s"
        )

        cam_centers_t = torch.tensor(
            np.stack(cam_centers, axis=0),
            dtype=point_xyz.dtype,
            device=point_xyz.device,
        )
        dist_to_camera = torch.cdist(point_xyz, cam_centers_t)
        nearest_any_idx = torch.argmin(dist_to_camera, dim=1)

        visible_mat = torch.from_numpy(np.stack(visibility_masks, axis=1)).to(point_xyz.device)
        room_ids_per_cam = np.asarray(
            [int(self.manual_split_assignments[cam_idx]) for cam_idx in assigned_camera_ids],
            dtype=np.int32,
        )
        unique_room_ids = sorted(set(room_ids_per_cam.tolist()))

        # Combined score terms; values can be tuned from GUI.
        room_k_nearest = max(1, int(getattr(self, "room_assign_k_nearest_slider", None).value)) \
            if hasattr(self, "room_assign_k_nearest_slider") else max(1, int(self.room_assign_k_nearest))
        support_expected = max(1e-3, float(getattr(self, "room_assign_support_expected_slider", None).value)) \
            if hasattr(self, "room_assign_support_expected_slider") else max(1e-3, float(self.room_assign_support_expected))
        support_penalty_lambda = float(getattr(self, "room_assign_support_penalty_slider", None).value) \
            if hasattr(self, "room_assign_support_penalty_slider") else float(self.room_assign_support_penalty_lambda)
        dist_weight = float(getattr(self, "room_assign_dist_weight_slider", None).value) \
            if hasattr(self, "room_assign_dist_weight_slider") else float(self.room_assign_dist_weight)
        center_weight = float(getattr(self, "room_assign_center_weight_slider", None).value) \
            if hasattr(self, "room_assign_center_weight_slider") else float(self.room_assign_center_weight)
        color_weight = float(getattr(self, "room_assign_color_weight_slider", None).value) \
            if hasattr(self, "room_assign_color_weight_slider") else float(self.room_assign_color_weight)
        support_reward_weight = float(getattr(self, "room_assign_support_reward_slider", None).value) \
            if hasattr(self, "room_assign_support_reward_slider") else float(self.room_assign_support_reward_weight)
        confidence_margin_threshold = float(getattr(self, "room_assign_confidence_margin_slider", None).value) \
            if hasattr(self, "room_assign_confidence_margin_slider") else float(self.room_assign_conf_margin_threshold)

        num_points = point_xyz.shape[0]
        num_rooms = len(unique_room_ids)
        room_scores = torch.full(
            (num_points, num_rooms),
            float("inf"),
            dtype=dist_to_camera.dtype,
            device=dist_to_camera.device,
        )

        # Use current SH->RGB as a weak appearance cue for room consistency.
        point_rgb = SH2RGB(self.gaussian_model._features_dc.squeeze()).detach()
        color_scale = torch.clamp(torch.std(point_rgb), min=1e-3)
        flat_dist = dist_to_camera.reshape(-1)
        max_scale_samples = 2_000_000
        if flat_dist.numel() > max_scale_samples:
            # Avoid quantile on huge tensors; sampled median is stable enough for normalization.
            stride = max(1, flat_dist.numel() // max_scale_samples)
            dist_sample = flat_dist[::stride]
        else:
            dist_sample = flat_dist
        dist_scale = torch.clamp(torch.median(dist_sample), min=1e-3)

        inf_dist = torch.full_like(dist_to_camera, float("inf"))
        for room_local_idx, room_id in enumerate(unique_room_ids):
            room_cam_mask_np = room_ids_per_cam == int(room_id)
            if not np.any(room_cam_mask_np):
                continue

            room_cam_mask = torch.from_numpy(room_cam_mask_np).to(dist_to_camera.device)
            room_dist = dist_to_camera[:, room_cam_mask]
            room_visible = visible_mat[:, room_cam_mask]

            room_visible_dist = torch.where(room_visible, room_dist, inf_dist[:, room_cam_mask])
            k_eff = min(room_k_nearest, int(room_visible_dist.shape[1]))
            topk_dist = torch.topk(room_visible_dist, k=k_eff, dim=1, largest=False).values
            topk_finite = torch.isfinite(topk_dist)
            topk_count = topk_finite.sum(dim=1)

            topk_sum = torch.where(topk_finite, topk_dist, torch.zeros_like(topk_dist)).sum(dim=1)
            mean_topk_dist = torch.where(
                topk_count > 0,
                topk_sum / topk_count.clamp(min=1),
                torch.full((num_points,), float("inf"), dtype=dist_to_camera.dtype, device=dist_to_camera.device),
            )

            visible_count = room_visible.sum(dim=1).to(dist_to_camera.dtype)
            support_gap = torch.clamp((support_expected - visible_count) / support_expected, min=0.0)
            support_penalty = 1.0 + support_penalty_lambda * support_gap
            support_reward = support_reward_weight * torch.clamp(visible_count / support_expected, max=1.0)

            room_cam_centers = cam_centers_t[room_cam_mask]
            room_center = room_cam_centers.mean(dim=0, keepdim=True)
            center_dist = torch.linalg.norm(point_xyz - room_center, dim=1) / dist_scale

            room_visible_any = room_visible.any(dim=1)
            if bool(torch.any(room_visible_any)):
                room_proto_rgb = point_rgb[room_visible_any].mean(dim=0, keepdim=True)
            else:
                room_proto_rgb = point_rgb.mean(dim=0, keepdim=True)
            color_dist = torch.mean(torch.abs(point_rgb - room_proto_rgb), dim=1) / color_scale

            dist_term = (mean_topk_dist / dist_scale) * support_penalty
            room_scores[:, room_local_idx] = (
                dist_weight * dist_term
                + center_weight * center_dist
                + color_weight * color_dist
                - support_reward
            )

        best_room_local_idx = torch.argmin(room_scores, dim=1)
        best_room_score = torch.min(room_scores, dim=1).values
        has_visible_room_support = torch.isfinite(best_room_score)

        room_ids_per_cam_t = torch.from_numpy(room_ids_per_cam).to(dist_to_camera.device)
        fallback_room_ids = room_ids_per_cam_t[nearest_any_idx]
        chosen_room_ids = torch.where(
            has_visible_room_support,
            torch.tensor(unique_room_ids, dtype=room_ids_per_cam_t.dtype, device=room_ids_per_cam_t.device)[best_room_local_idx],
            fallback_room_ids,
        )

        low_confidence: Optional[torch.Tensor] = None
        confidence_margin: Optional[torch.Tensor] = None
        if num_rooms > 1:
            two_best = torch.topk(room_scores, k=2, dim=1, largest=False).values
            confidence_margin = two_best[:, 1] - two_best[:, 0]
            low_confidence = has_visible_room_support & (confidence_margin < confidence_margin_threshold)
            chosen_room_ids = torch.where(low_confidence, fallback_room_ids, chosen_room_ids)
            print(
                f"[INFO] room assignment confidence fallback: low_conf={int(low_confidence.sum().item())}/{num_points}, "
                f"margin_th={confidence_margin_threshold:.3f}"
            )

        print(
            f"[INFO] split score params | k={room_k_nearest}, expected={support_expected:.2f}, "
            f"penalty={support_penalty_lambda:.2f}, w_dist={dist_weight:.2f}, w_center={center_weight:.2f}, "
            f"w_color={color_weight:.2f}, w_support={support_reward_weight:.2f}"
        )

        chosen_room_ids = self._refine_point_room_assignments_graph(
            point_xyz=point_xyz,
            room_scores=room_scores,
            unique_room_ids=unique_room_ids,
            current_assignments=chosen_room_ids,
            low_confidence_mask=low_confidence,
            confidence_margin=confidence_margin,
        )

        point_room_ids = chosen_room_ids.detach().cpu().numpy().astype(np.int32)

        return point_room_ids

    def _refine_point_room_assignments_graph(
        self,
        point_xyz: torch.Tensor,
        room_scores: torch.Tensor,
        unique_room_ids: List[int],
        current_assignments: torch.Tensor,
        low_confidence_mask: Optional[torch.Tensor] = None,
        confidence_margin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enable_graph_refine = bool(getattr(self, "room_assign_graph_refine_enable_checkbox", None).value) \
            if hasattr(self, "room_assign_graph_refine_enable_checkbox") else bool(self.room_assign_graph_refine_enable)
        if not enable_graph_refine:
            return current_assignments

        num_points = int(point_xyz.shape[0])
        num_rooms = len(unique_room_ids)
        if num_points == 0 or num_rooms <= 1:
            return current_assignments

        graph_k = int(getattr(self, "room_assign_graph_knn_slider", None).value) \
            if hasattr(self, "room_assign_graph_knn_slider") else int(self.room_assign_graph_knn)
        smooth_lambda = float(getattr(self, "room_assign_graph_lambda_slider", None).value) \
            if hasattr(self, "room_assign_graph_lambda_slider") else float(self.room_assign_graph_lambda)
        sigma_x_manual = float(getattr(self, "room_assign_graph_sigma_x_slider", None).value) \
            if hasattr(self, "room_assign_graph_sigma_x_slider") else float(self.room_assign_graph_sigma_x)
        sigma_c = float(getattr(self, "room_assign_graph_sigma_c_slider", None).value) \
            if hasattr(self, "room_assign_graph_sigma_c_slider") else float(self.room_assign_graph_sigma_c)
        num_iters = int(getattr(self, "room_assign_graph_iters_slider", None).value) \
            if hasattr(self, "room_assign_graph_iters_slider") else int(self.room_assign_graph_iters)
        only_low_conf = bool(getattr(self, "room_assign_graph_only_low_conf_checkbox", None).value) \
            if hasattr(self, "room_assign_graph_only_low_conf_checkbox") else bool(self.room_assign_graph_only_low_conf)
        low_conf_margin = float(getattr(self, "room_assign_graph_low_conf_margin_slider", None).value) \
            if hasattr(self, "room_assign_graph_low_conf_margin_slider") else float(self.room_assign_graph_low_conf_margin)
        use_normal_consistency = bool(getattr(self, "room_assign_graph_use_normal_consistency_checkbox", None).value) \
            if hasattr(self, "room_assign_graph_use_normal_consistency_checkbox") else bool(self.room_assign_graph_use_normal_consistency)
        normal_weight = float(getattr(self, "room_assign_graph_normal_weight_slider", None).value) \
            if hasattr(self, "room_assign_graph_normal_weight_slider") else float(self.room_assign_graph_normal_weight)

        if smooth_lambda <= 0.0 or num_iters <= 0:
            return current_assignments

        k_eff = max(1, min(graph_k, num_points - 1))
        if k_eff <= 0:
            return current_assignments

        _, nn_idx, nn_dist = K_nearest_neighbors(point_xyz, k_eff + 1, point_xyz, return_dist=True)
        if nn_idx.ndim == 1:
            nn_idx = nn_idx.unsqueeze(1)
            nn_dist = nn_dist.unsqueeze(1)

        nn_idx = nn_idx[:, 1:]
        nn_dist = nn_dist[:, 1:]

        if nn_idx.shape[1] == 0:
            return current_assignments

        if sigma_x_manual > 0:
            sigma_x = torch.tensor(sigma_x_manual, dtype=point_xyz.dtype, device=point_xyz.device)
        else:
            sigma_x = torch.clamp(torch.median(nn_dist), min=1e-4)
        sigma_c_t = torch.tensor(max(sigma_c, 1e-4), dtype=point_xyz.dtype, device=point_xyz.device)

        point_rgb = SH2RGB(self.gaussian_model._features_dc.squeeze()).detach()
        nn_rgb = point_rgb[nn_idx]
        color_diff = torch.mean(torch.abs(point_rgb[:, None, :] - nn_rgb), dim=2)

        w_x = torch.exp(-0.5 * (nn_dist / sigma_x) ** 2)
        w_c = torch.exp(-(color_diff / sigma_c_t))
        pair_w = w_x * w_c

        if use_normal_consistency and nn_idx.shape[1] >= 2 and normal_weight > 0.0:
            # Fast local normal approximation from two nearest neighbor directions.
            p0 = point_xyz
            p1 = point_xyz[nn_idx[:, 0]]
            p2 = point_xyz[nn_idx[:, 1]]
            v1 = p1 - p0
            v2 = p2 - p0
            n = torch.cross(v1, v2, dim=1)
            n = torch.nn.functional.normalize(n, dim=1, eps=1e-6)

            nn_n = n[nn_idx]
            cos_sim = torch.sum(n[:, None, :] * nn_n, dim=2).abs().clamp(0.0, 1.0)
            w_n = torch.exp(-normal_weight * (1.0 - cos_sim))
            pair_w = pair_w * w_n

        unary = torch.nan_to_num(room_scores.detach(), nan=1e6, posinf=1e6, neginf=0.0)
        room_ids_t = torch.tensor(unique_room_ids, dtype=current_assignments.dtype, device=current_assignments.device)

        labels = current_assignments.clone()
        target_mask = torch.ones((num_points,), dtype=torch.bool, device=labels.device)
        if only_low_conf:
            if low_confidence_mask is not None:
                target_mask = low_confidence_mask.to(labels.device)
            elif confidence_margin is not None:
                target_mask = confidence_margin.to(labels.device) < low_conf_margin
            else:
                two_best = torch.topk(unary, k=min(2, unary.shape[1]), dim=1, largest=False).values
                if two_best.shape[1] == 2:
                    target_mask = (two_best[:, 1] - two_best[:, 0]) < low_conf_margin
                else:
                    target_mask = torch.zeros((num_points,), dtype=torch.bool, device=labels.device)

            if not bool(torch.any(target_mask)):
                print("[INFO] graph refine skipped: no low-confidence points")
                return labels

        for _ in range(num_iters):
            smooth_cost = torch.zeros_like(unary)
            for ridx in range(num_rooms):
                room_id = room_ids_t[ridx]
                same_label = (labels[nn_idx] == room_id).to(pair_w.dtype)
                smooth_cost[:, ridx] = torch.sum(pair_w * (1.0 - same_label), dim=1)

            total_cost = unary + smooth_lambda * smooth_cost
            new_local = torch.argmin(total_cost, dim=1)
            new_labels = room_ids_t[new_local]
            if only_low_conf:
                new_labels = torch.where(target_mask, new_labels, labels)

            changed = int((new_labels != labels).sum().item())
            labels = new_labels
            if changed == 0:
                break

        print(
            f"[INFO] graph refine | k={k_eff}, lambda={smooth_lambda:.3f}, "
            f"sigma_x={float(sigma_x.item()):.4f}, sigma_c={float(sigma_c_t.item()):.4f}, iters={num_iters}, "
            f"only_low_conf={only_low_conf}, low_conf_points={int(target_mask.sum().item())}/{num_points}, "
            f"normal_consistency={use_normal_consistency}, normal_w={normal_weight:.3f}"
        )
        return labels

    def _build_room_point_masks(self, point_room_ids: np.ndarray) -> Dict[int, np.ndarray]:
        room_masks: Dict[int, np.ndarray] = {}
        unique_room_ids = sorted(set([int(v) for v in point_room_ids.tolist() if int(v) >= 0]))
        if len(unique_room_ids) == 0:
            unique_room_ids = sorted(set([int(v) for v in self.manual_split_assignments.values()]))
        for room_id in unique_room_ids:
            room_masks[room_id] = (point_room_ids == room_id)
        return room_masks

    def _colorize_points_by_room(self):
        if self.manual_split_point_room_ids is None:
            print("[WARN] no room assignment for points, please apply split first")
            return

        if self.manual_split_color_backup_dc is None:
            self.manual_split_color_backup_dc = self.gaussian_model._features_dc.detach().clone()
            self.manual_split_color_backup_rest = self.gaussian_model._features_rest.detach().clone()

        with torch.no_grad():
            current_rgb = SH2RGB(self.gaussian_model._features_dc.squeeze()).detach().clone()
            room_ids = torch.from_numpy(self.manual_split_point_room_ids).to(current_rgb.device)
            point_rooms = sorted(set([int(v) for v in self.manual_split_point_room_ids.tolist() if int(v) >= 0]))
            if len(point_rooms) == 0:
                point_rooms = sorted(set([int(v) for v in self.manual_split_assignments.values()]))
            for room_id in point_rooms:
                mask = room_ids == int(room_id)
                color = torch.tensor(self._room_color(int(room_id)), dtype=current_rgb.dtype, device=current_rgb.device) / 255.0
                current_rgb[mask] = color

            recolor_dc = RGB2SH(current_rgb).unsqueeze(1)
            self.gaussian_model._features_dc.copy_(recolor_dc)

        self.update_client()
        print("[INFO] colorized points by room assignment")

    def _restore_point_colors_after_room_visualization(self):
        self._clear_manual_split_point_selection_preview()

        if self.manual_split_color_backup_dc is None:
            print("[INFO] no backup color to restore")
            return

        with torch.no_grad():
            self.gaussian_model._features_dc.copy_(self.manual_split_color_backup_dc)
            self.gaussian_model._features_rest.copy_(self.manual_split_color_backup_rest)

        self.manual_split_color_backup_dc = None
        self.manual_split_color_backup_rest = None
        self.update_client()
        self._refresh_manual_split_point_selection_preview()
        print("[INFO] restored original point colors")

    def _save_manual_split_data(self):
        if self.prep_dir is None or self.prep_dir == "":
            raise ValueError("--prep_dir is empty, cannot save split data")
        if len(self.manual_split_assignments) == 0:
            raise ValueError("No camera assignment found")

        split_dir = os.path.join(self.prep_dir, "split")
        os.makedirs(split_dir, exist_ok=True)

        assigned_camera_ids = sorted(self.manual_split_assignments.keys())
        point_room_ids = self._compute_point_room_assignments(assigned_camera_ids)
        room_masks = self._build_room_point_masks(point_room_ids)

        self.manual_split_point_room_ids = point_room_ids
        self.manual_split_room_point_masks = room_masks

        np.save(os.path.join(split_dir, "point_room_ids.npy"), point_room_ids.astype(np.int32))

        room_point_counts = {
            str(room_id): int(mask.sum()) for room_id, mask in sorted(room_masks.items(), key=lambda x: x[0])
        }
        camera_assignments = {
            str(cam_idx): {
                "room_id": int(room_id),
                "img_name": str(self.camera_poses[cam_idx].get("img_name", f"cam_{cam_idx}")),
            }
            for cam_idx, room_id in sorted(self.manual_split_assignments.items(), key=lambda x: x[0])
        }

        summary = {
            "num_cameras_total": int(len(self.camera_poses)),
            "num_cameras_assigned": int(len(self.manual_split_assignments)),
            "num_points_total": int(point_room_ids.shape[0]),
            "camera_assignments": camera_assignments,
            "room_point_counts": room_point_counts,
        }
        with open(os.path.join(split_dir, "manual_split_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        with open(os.path.join(split_dir, "camera_room_assignments.json"), "w", encoding="utf-8") as f:
            json.dump({str(k): int(v) for k, v in sorted(self.manual_split_assignments.items())}, f, ensure_ascii=False, indent=2)

        for room_id, mask in room_masks.items():
            np.save(os.path.join(split_dir, f"room_{room_id}_point_mask.npy"), mask.astype(np.uint8))

        print(f"[INFO] split data saved in {split_dir}")

    def _load_manual_split_data(self):
        if self.prep_dir is None or self.prep_dir == "":
            return
        split_dir = os.path.join(self.prep_dir, "split")
        assignments_path = os.path.join(split_dir, "camera_room_assignments.json")
        if not os.path.exists(assignments_path):
            return

        with open(assignments_path, "r", encoding="utf-8") as f:
            assignments_raw = json.load(f)
        self.manual_split_assignments = {
            int(k): int(v)
            for k, v in assignments_raw.items()
            if 0 <= int(k) < len(self.camera_poses)
        }
        self.manual_split_selected_ids = set()
        self.manual_split_selected_point_mask = None

        point_room_path = os.path.join(split_dir, "point_room_ids.npy")
        if os.path.exists(point_room_path):
            self.manual_split_point_room_ids = np.load(point_room_path).astype(np.int32)
            self.manual_split_room_point_masks = self._build_room_point_masks(self.manual_split_point_room_ids)

    def add_cameras_to_scene(self, viser_server):
        if len(self.camera_poses) == 0:
            print("[WARN] no camera poses, cannot add camera frustums")
            return

        self.camera_handles = []
        self.camera_handles_by_idx = {}
        self.camera_handle_names = {}
        camera_pose_transform = np.linalg.inv(self.camera_transform.cpu().numpy())
        for idx, camera in enumerate(self.camera_poses):
            name = camera["img_name"]
            if camera.get("pose_source", "") == "colmap_fallback":
                # Keep exactly the same convention as manual_split_viewer for COLMAP fallback.
                r_wc = np.asarray(camera["rotation"], dtype=np.float32)
                center = np.asarray(camera["position"], dtype=np.float32)
                R = vtf.SO3.from_matrix(r_wc) @ vtf.SO3.from_x_radians(np.pi)
                position = center
            else:
                c2w = np.eye(4)
                c2w[:3, :3] = np.asarray(camera["rotation"])
                c2w[:3, 3] = np.asarray(camera["position"])
                c2w[:3, 1:3] *= -1
                c2w = np.matmul(camera_pose_transform, c2w)

                R = vtf.SO3.from_matrix(c2w[:3, :3])
                R = R @ vtf.SO3.from_x_radians(np.pi)
                position = c2w[:3, 3]

            cx = camera["width"] // 2
            cy = camera["height"] // 2
            fx = camera["fx"]

            camera_handle = viser_server.add_camera_frustum(
                name="cameras/{}".format(name),
                fov=float(2 * np.arctan(cx / fx)),
                scale=0.05,
                aspect=float(cx / cy),
                wxyz=R.wxyz,
                position=position,
                color=(255, 255, 0),
            )

            @camera_handle.on_click
            def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle], camera_idx=idx) -> None:
                if hasattr(self, "manual_split_enable") and self.manual_split_enable.value:
                    print(f"Clicked camera idx: {camera_idx}")
                    if camera_idx in self.manual_split_selected_ids:
                        self.manual_split_selected_ids.remove(camera_idx)
                    else:
                        self.manual_split_selected_ids.add(camera_idx)
                    self._refresh_manual_split_camera_colors()
                    if hasattr(self, "manual_split_status"):
                        self.manual_split_status.value = self._manual_split_status_text()
                    return

                if hasattr(self, "training_view_slider"):
                    target_idx = int(max(0, min(camera_idx, self.training_view_slider.max)))
                    self.training_view_slider.value = target_idx

                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz

            self.camera_handles.append(camera_handle)
            self.camera_handles_by_idx[idx] = camera_handle
            self.camera_handle_names[idx] = name

        if not hasattr(self, "camera_visible"):
            self.camera_visible = bool(self.show_cameras)
        for handle in self.camera_handles:
            handle.visible = self.camera_visible

        if not hasattr(self, "show_cameras_frustrum"):
            self.show_cameras_frustrum = viser_server.add_gui_button("Toggle Train Cameras")

            @self.show_cameras_frustrum.on_click
            def toggle_camera_visibility(_):
                with viser_server.atomic():
                    self.camera_visible = not self.camera_visible
                    for i in self.camera_handles:
                        i.visible = self.camera_visible

        self._refresh_manual_split_camera_colors()

    def start(self, block: bool = True, server_config_fun=None, tab_config_fun=None):
        # create viser server
        server = viser.ViserServer(host=self.host, port=self.port)
        self._setup_titles(server)
        if server_config_fun is not None:
            server_config_fun(self, server)

        tabs = server.add_gui_tab_group()
        if tab_config_fun is not None:
            tab_config_fun(self, server, tabs)

        # setup panels 
        self._setup_general_features_folder(server, tabs)

        if self.show_edit_panel:
            with tabs.add_tab("Edit") as edit_tab:
                self.edit_panel = EditPanel(server, self, edit_tab)
                @self.edit_panel.show_point_cloud_checkbox.on_update
                @self.edit_panel.show_mesh_button.on_click 
                @self.edit_panel.unshow_mesh_button.on_click
                def _(event): 
                    with server.atomic(): self._handle_option_updated(_)

        self.transform_panel: TransformPanel = None
        if self.enable_transform:
            with tabs.add_tab("Transform"):
                self.transform_panel = TransformPanel(server, self)

        if self.show_render_panel:
            with tabs.add_tab("Render"):
                self.render_panel = RenderPanel(server, 
                                                self, 
                                                self.model_path,
                                                Path('./renders'),
                                                orientation_transform=torch.linalg.inv(self.camera_transform).cpu().numpy(),
                                                enable_transform=self.enable_transform,
                                                background_color=self.background_color.detach().cpu().numpy().tolist(),
                                                sh_degree=self.sh_degree,)
                
        # register hooks
        server.on_client_connect(self._handle_new_client)
        server.on_client_disconnect(self._handle_client_disconnect)
        if block is True:
            while True:
                time.sleep(999)
    
    def _setup_titles(self, server):
        buttons = (
            # TitlebarButton(
            #     text="Simple Viser Viewer for 2D Gaussian Splatting",
            #     icon="GitHub",
            #     href="https://github.com/hwanhuh/2D-GS-Viser-Viewer/tree/main",
            # ),
            # TitlebarButton(
            #     text="Hwan Heo",
            #     icon="GitHub",
            #     href="https://github.com/hwanhuh",
            # ),
        )
        image = TitlebarImage(
            image_url_light="https://viser.studio/latest/_static/logo.svg",
            image_alt="Logo",
            href="https://github.com/nerfstudio-project/viser"
        )
        titlebar_theme = TitlebarConfig(buttons=buttons, image=image)
        brand_color = server.add_gui_rgb("Brand color", (10, 10, 10), visible=False)
        server.configure_theme(
            titlebar_content=titlebar_theme,
            show_logo=True,
            brand_color=brand_color.value,
        )

    def update_client(self):
        # self.viewer_renderer = ViewerRenderer(self.gaussian_model, self.background_color, not self.is_training)
        # for client_id in self.clients:
        #     self.clients[client_id].renderer = self.viewer_renderer
        self.viewer_renderer.update_pc_features()
    
    def render(self, R, T, fx, fy, w, h):
        pano_camera_params = {
            'fx': torch.tensor([fov2focal(fx, w)], dtype=torch.float),
            'fy': torch.tensor([fov2focal(fy, h)], dtype=torch.float), 
            'cx': torch.tensor([(w // 2)], dtype=torch.int),
            'cy': torch.tensor([(h // 2)], dtype=torch.int),
            'width': torch.tensor([w], dtype=torch.int),
            'height': torch.tensor([h], dtype=torch.int),
            'appearance_id': torch.tensor([0], dtype=torch.int),
            'normalized_appearance_id': torch.tensor([0.], dtype=torch.float),
            'time': torch.tensor([0], dtype=torch.float),
            'distortion_params': None,
            'camera_type': torch.tensor([0], dtype=torch.int),
        }

        R = torch.from_numpy(np.array(R))
        T = torch.from_numpy(np.array(T))

        cam = Cameras(
            R=R.transpose(0,1).unsqueeze(0),
            T=T.unsqueeze(0),
            # **camera_position(R.unsqueeze(0), T.unsqueeze(0), Trans.unsqueeze(0)),
            **pano_camera_params,
        )[0].to_device(self.device)

        # cam = Simple_Camera(0, np.array(rotation), np.array([0,0,0]), fx, fy,
        #                     h, w, "", 0, trans=np.array(position))

        render_params = {
            'active_sh_degree': self.active_sh_degree_slider.value if hasattr(self, 'active_sh_degree_slider') else 0, 
            'scaling_modifier': self.scale_slider.value, 
            'depth_ratio': self.depth_ratio_slider.value,
            'bg_color': self.viewer_renderer.background_color,
            'sparsity': self.sparsity_slider.value, 
            'valid_range': None,
            'show_ptc': self.enable_ptc.value and (self.surfel_mode.value == 'ptc'),
            'show_disk': self.enable_ptc.value and (self.surfel_mode.value == 'disk'),
            'point_size': self.point_size.value,
            'override_color': SH2RGB(self.gaussian_model._features_dc.squeeze()),
        }

        with torch.no_grad():
            results = self.viewer_renderer.render_viewer(cam,
                **render_params,
            )
        return results
    
    def render_mask(self, R, T, fx, fy, w, h, mask):
        pano_camera_params = {
            'fx': torch.tensor([fov2focal(fx, w)], dtype=torch.float),
            'fy': torch.tensor([fov2focal(fy, h)], dtype=torch.float), 
            'cx': torch.tensor([(w // 2)], dtype=torch.int),
            'cy': torch.tensor([(h // 2)], dtype=torch.int),
            'width': torch.tensor([w], dtype=torch.int),
            'height': torch.tensor([h], dtype=torch.int),
            'appearance_id': torch.tensor([0], dtype=torch.int),
            'normalized_appearance_id': torch.tensor([0.], dtype=torch.float),
            'time': torch.tensor([0], dtype=torch.float),
            'distortion_params': None,
            'camera_type': torch.tensor([0], dtype=torch.int),
        }

        R = torch.from_numpy(np.array(R))
        T = torch.from_numpy(np.array(T))

        cam = Cameras(
            R=R.transpose(0,1).unsqueeze(0),
            T=T.unsqueeze(0),
            # **camera_position(R.unsqueeze(0), T.unsqueeze(0), Trans.unsqueeze(0)),
            **pano_camera_params,
        )[0].to_device(self.device)

        # cam = Simple_Camera(0, np.array(rotation), np.array([0,0,0]), fx, fy,
        #                     h, w, "", 0, trans=np.array(position))

        render_params = {
            'active_sh_degree': self.active_sh_degree_slider.value if hasattr(self, 'active_sh_degree_slider') else 0, 
            'scaling_modifier': self.scale_slider.value, 
            'depth_ratio': self.depth_ratio_slider.value,
            'bg_color': self.viewer_renderer.background_color,
            'sparsity': self.sparsity_slider.value, 
            'valid_range': None,
            'show_ptc': self.enable_ptc.value and (self.surfel_mode.value == 'ptc'),
            'show_disk': self.enable_ptc.value and (self.surfel_mode.value == 'disk'),
            'point_size': self.point_size.value,
            'override_color': torch.from_numpy(mask)[..., None].float().repeat(1, 3).to(self.device),
        }

        with torch.no_grad():
            results = self.viewer_renderer.render_viewer(cam,
                **render_params,
            )
        return results
    
    def render_panorama(self, camera_center, res=1024, camera_rotation=np.diag(np.ones((3))), save_dir=None, render_perspetive=False, verbose=True, mask=None):
        if verbose: print(f"generate panorama center in {camera_center}...")
        # image_height = 1024
        # image_width = 1024
        # res = res

        pers_params = [
            (90, 0,   0),
            (90, 90,  0),
            (90, 180, 0),
            (90, 270, 0),
        
            (90, 0, 90),
            (90, 0, -90),
        ]

        # Rs = [torch.from_numpy(vtf.SO3.from_rpy_radians(math.radians(p[2]), math.radians(p[1]), math.radians(0)).as_matrix()) for p in pers_params]
        Rs = [torch.from_numpy(camera_rotation @ vtf.SO3.from_rpy_radians(math.radians(p[2]), math.radians(p[1]), math.radians(0)).as_matrix()) for p in pers_params]
        T = torch.tensor([0,0,0])
        Trans = torch.tensor(camera_center)
        
        pano_camera_params = {
            'fx': torch.tensor([fov2focal(math.radians(90), res)], dtype=torch.float),
            'fy': torch.tensor([fov2focal(math.radians(90), res)], dtype=torch.float), 
            'cx': torch.tensor([(res // 2)], dtype=torch.int),
            'cy': torch.tensor([(res // 2)], dtype=torch.int),
            'width': torch.tensor([res], dtype=torch.int),
            'height': torch.tensor([res], dtype=torch.int),
            'appearance_id': torch.tensor([0], dtype=torch.int),
            'normalized_appearance_id': torch.tensor([0.], dtype=torch.float),
            'time': torch.tensor([0], dtype=torch.float),
            'distortion_params': None,
            'camera_type': torch.tensor([0], dtype=torch.int),
        }

        # cams = [Cameras(
        #     # R=R.unsqueeze(0),
        #     # T=T.unsqueeze(0),
        #     **camera_position(R.unsqueeze(0), T.unsqueeze(0), Trans.unsqueeze(0)),
        #     **pano_camera_params,
        # )[0].to_device(self.device) for R in Rs]


        cams = [Simple_Camera(0, R.numpy(), T.numpy(), math.radians(90), math.radians(90),
                            res, res, "", 0, trans=Trans.numpy()) for R in Rs]

        override_color = SH2RGB(self.gaussian_model._features_dc.squeeze()) if mask is None else torch.from_numpy(mask)[..., None].float().repeat(1, 3).to(self.device)
        # override_color = None if mask is None else torch.from_numpy(mask)[..., None].float().repeat(1, 3).to(self.device)
        render_params = {
            'active_sh_degree': self.active_sh_degree_slider.value if hasattr(self, 'active_sh_degree_slider') else 0, 
            'scaling_modifier': self.scale_slider.value, 
            'depth_ratio': self.depth_ratio_slider.value,
            'bg_color': self.viewer_renderer.background_color,
            'sparsity': self.sparsity_slider.value, 
            'valid_range': None,
            'show_ptc': self.enable_ptc.value and (self.surfel_mode.value == 'ptc'),
            'show_disk': self.enable_ptc.value and (self.surfel_mode.value == 'disk'),
            'point_size': self.point_size.value,
            # 'override_color': SH2RGB(self.gaussian_model._features_dc.squeeze()),
            'override_color': override_color,
        }

        with torch.no_grad():
            results = [self.viewer_renderer.render_viewer(cam,
                **render_params,
            ) for cam in cams]
        
        pano_images = []
        pano_depthes = []
        # pano_depthes_distorted = []

        depth_distort = get_depth_distort(res=res).to(results[0]['surf_depth'].device) * math.radians(90) #* 2
        # depth_distort = torch.zeros((1024,1024))

        if render_perspetive:
            # for idx, result in enumerate(results):
                # pano_images.append(result['render'])
            pano_images = torch.cat([r['render'].unsqueeze(0) for r in results], dim=0)
            if save_dir:
                for ridx, r in enumerate(results):
                    pimg = Image.fromarray((r['render'].clip(0,1).permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8))
                    pimg.save(os.path.join(save_dir, f'perspective_img{ridx}.png'))
            return pano_images
        else:
            for idx, result in enumerate(results):
                # result['render'] = self.

                # Image.fromarray((result['render'].clip(0,1).permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8)).save(f"temp{idx}.png")
                pano_images.append((result['render'].clip(0,1).permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8))
                # pano_depthes.append(np.repeat(result['surf_depth'][0].detach().cpu().numpy(), 3, axis=-1))
                # pano_depthes.append((result['surf_depth'][0,:,:,0])[...,None].repeat(1,1,3).detach().cpu().numpy())
                # pano_alphas.append(result['surf_mask'][0].detach().cpu().numpy())
                # pano_depthes.append((result['surf_depth']+result['rend_dist'])[0,:,:,0][...,None].repeat(1,1,3).detach().cpu().numpy())
                pano_depthes.append((result['surf_depth'][0,:,:,0]+depth_distort)[...,None].repeat(1,1,3).detach().cpu().numpy())
                # pano_depthes_distorted.append((result['surf_depth'][0,:,:,0]+depth_distort)[...,None].repeat(1,1,3).detach().cpu().numpy())
        
            ee_image = m_P2E.Perspective(pano_images, pers_params)
            ee_depth = m_P2E.Perspective(pano_depthes, pers_params)
            # ee_depth_distorted = m_P2E.Perspective(pano_depthes_distorted, pers_params)
            # ee_distort = m_P2E.Perspective(pano_distortes, pers_params)

            print("generating panorama image...")
            pano_image = ee_image.GetEquirec(res, res*2)
            print("generating panorama depth...")
            pano_depth = ee_depth.GetEquirec(res, res*2)
            # pano_depth_distorted = ee_depth_distorted.GetEquirec(1024, 2048)
            # print("generating panorama distort...")
            # pano_distort = ee_distort.GetEquirec(1024, 2048)

            # pano_depth = pano_depth + pano_distort

            if save_dir:
                print(f"saving into {save_dir}")
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(pano_image.astype(np.uint8)).save(os.path.join(save_dir, 'pano_img.png'))
                # Image.fromarray(((pano_depth-pano_depth.min())/(pano_depth.max()-pano_depth.min()+1e-9)*255).astype(np.uint8)).save(os.path.join(save_dir, 'pano_depth.png'))
                Image.fromarray((((pano_depth-pano_depth.min())/(pano_depth.max()-pano_depth.min())*0.6+0.2)*255).astype(np.uint8)).save(os.path.join(save_dir, 'pano_depth.png')) # depth 50-200            
                # Image.fromarray(((pano_depth_distorted-pano_depth_distorted.min())/(pano_depth_distorted.max()-pano_depth_distorted.min()+1e-9)*255).astype(np.uint8)).save(os.path.join(save_dir, 'pano_depth_distorted.png'))
                np.save(os.path.join(save_dir, 'pano_depth.npy'), pano_depth)
                # np.save(os.path.join(save_dir, 'pano_depth_distorted.npy'), pano_depth_distorted)
                np.save(os.path.join(save_dir, 'camera_center.npy'), Trans.detach().cpu().numpy())
                np.save(os.path.join(save_dir, 'camera_rotation.npy'), camera_rotation)

                np.save(os.path.join(save_dir, 'pano_pimages.npy'), pano_images)
                np.save(os.path.join(save_dir, 'pano_pdepthes.npy'), pano_depthes)
                # np.save(os.path.join(save_dir, 'pano_palpha.npy'), pano_alphas)
            return pano_image, pano_depth
    
    def get_pano_depth_distort(self):
        pers_params = [
            (90, 0,   0),
            (90, 90,  0),
            (90, 180, 0),
            (90, 270, 0),
        
            (90, 0, 90),
            (90, 0, -90),
        ]

        depth_distort = get_depth_distort() * math.radians(90) * 2
        pano_depthes_distorted = [(depth_distort)[...,None].repeat(1,1,3).detach().cpu().numpy()]*6

        ee_distort = m_P2E.Perspective(pano_depthes_distorted, pers_params)

        print("generating panorama distort...")
        pano_distort = ee_distort.GetEquirec(1024, 2048)
        Image.fromarray(((pano_distort-pano_distort.min())/(pano_distort.max()-pano_distort.min()+1e-9)*255).astype(np.uint8)).save('pano_distort.png')
        return pano_distort
    
    def check_and_render_panorama(self, camera_center, camera_rotation, save_dir, mask=None, res=1024):
        if os.path.exists(os.path.join(save_dir, "pano_img.png")) and os.path.exists(os.path.join(save_dir, "pano_depth.npy")):
            print(f"Loading panorama in {save_dir}...")
            image = np.array(Image.open(os.path.join(save_dir, "pano_img.png")))
            depth = np.load(os.path.join(save_dir, "pano_depth.npy"))
        else:
            print(f"Generating panorama to {save_dir}...")
            image, depth = self.render_panorama(camera_center, camera_rotation=camera_rotation, save_dir=save_dir, res=res, mask=mask)
        return image, depth

    def clear_prep_cam_folders(self, prep_dir):
        target_folders = ("after_styled", "styled")#, "mask")
        prep_path = os.path.abspath(prep_dir)
        if not os.path.isdir(prep_path):
            print(f"[WARN] prep_dir not found: {prep_path}")
            return 0, 0

        cam_dirs = sorted(
            path for path in glob.glob(os.path.join(prep_path, "**", "cam_*"), recursive=True) if os.path.isdir(path)
        )
        removed_count = 0
        for cam_dir in cam_dirs:
            for folder_name in target_folders:
                target_dir = os.path.join(cam_dir, folder_name)
                if not os.path.isdir(target_dir):
                    continue
                try:
                    shutil.rmtree(target_dir)
                    removed_count += 1
                    print(f"[INFO] removed {target_dir}")
                except Exception as err:
                    print(f"[WARN] failed to remove {target_dir}: {err}")

        return len(cam_dirs), removed_count

    def is_camera_center_too_close(self, camera_center):
        if not hasattr(self, "min_camera_object_dist_slider"):
            return False
        min_dist = float(self.min_camera_object_dist_slider.value)
        if min_dist <= 0:
            return False

        point_xyz = self.gaussian_model.get_xyz
        if point_xyz.shape[0] == 0:
            return False

        with torch.no_grad():
            center = torch.tensor(camera_center, dtype=point_xyz.dtype, device=point_xyz.device)
            nearest_dist = torch.linalg.norm(point_xyz - center, dim=1).min().item()

        return nearest_dist < min_dist

    def _get_preprocess_candidate_cam_ids(self):
        def compute_mean_nn_distance(ids):
            if len(ids) < 2:
                return 0.0, np.zeros(len(ids), dtype=np.float32)

            centers = np.asarray([self.camera_poses[idx]["position"] for idx in ids], dtype=np.float32)
            pairwise_dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
            np.fill_diagonal(pairwise_dist, np.inf)
            nn_dist = pairwise_dist.min(axis=1)
            return float(nn_dist.mean()), nn_dist
    
        def get_gap_threshold():
            all_cam_id = list(range(len(self.colmap_cameras)))
            filter_cam_id = all_cam_id.copy()

            # Stage 1: remove cameras with too-small local movement using global mean NN distance.
            avg_all_dist, all_nn_dist = compute_mean_nn_distance(all_cam_id)
            if len(all_cam_id) > 1 and avg_all_dist > 0:
                filter_cam_id = [idx for idx, nn_dist in zip(all_cam_id, all_nn_dist) if nn_dist >= avg_all_dist]
                if len(filter_cam_id) == 0:
                    filter_cam_id = all_cam_id.copy()

            # Stage 2: build adaptive gap threshold from remaining cameras.
            avg_filtered_dist, _ = compute_mean_nn_distance(filter_cam_id)
            if avg_filtered_dist > 0:
                gap_threshold = float(self.camera_gap_slider.value) * avg_filtered_dist
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
                f"avg_filtered={avg_filtered_dist:.4f}, slider={float(self.camera_gap_slider.value):.4f}, "
                f"gap_threshold={gap_threshold:.4f}, stage1_kept={len(filter_cam_id)}/{len(all_cam_id)}"
            )
            return gap_threshold, debug_stats

        gap_threshold, camera_distance_stats = get_gap_threshold()
        self.last_preprocess_camera_distance_stats = camera_distance_stats
        cam_id = list(range(len(self.colmap_cameras)))

        # center_cam = np.array([0, 0, 0])[None]
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
                if np.all(np.linalg.norm(center_cam - center, axis=1) > gap_threshold):
                    nearest_id = candidate_id
                    break
            nearest_idx = cam_id.index(nearest_id)
            cam_id = cam_id[nearest_idx:] + cam_id[:nearest_idx]

        filtered_cam_id = []
        for candidate_id in cam_id:
            center = np.array(self.camera_poses[candidate_id]["position"])
            too_near_center_cam = not np.all(np.linalg.norm(center_cam - center, axis=1) > gap_threshold)
            too_close_to_object = self.is_camera_center_too_close(center)
            if too_near_center_cam or too_close_to_object:
                continue
            filtered_cam_id.append(candidate_id)
            center_cam = np.concatenate((center_cam, center[None]), axis=0)

        if self.enable_camera_center.value:
            filtered_cam_id.insert(0, -1)
        return filtered_cam_id

    def _resolve_preprocess_room_groups(self, prefer_manual_split: bool = True) -> Tuple[Dict[int, List[int]], str]:
        candidate_cam_ids = [int(cam_idx) for cam_idx in self._get_preprocess_candidate_cam_ids()]

        if prefer_manual_split and len(self.manual_split_assignments) > 0:
            room_groups: Dict[int, List[int]] = {}
            skipped_special_ids: List[int] = []
            unassigned_candidate_ids: List[int] = []

            for cam_idx in candidate_cam_ids:
                if cam_idx < 0:
                    skipped_special_ids.append(cam_idx)
                    continue

                room_id = self.manual_split_assignments.get(cam_idx)
                if room_id is None:
                    unassigned_candidate_ids.append(cam_idx)
                    continue

                room_groups.setdefault(int(room_id), []).append(cam_idx)

            if len(room_groups) > 0:
                if len(skipped_special_ids) > 0:
                    print(f"[INFO] skip special candidate IDs in manual grouping: {skipped_special_ids}")
                if len(unassigned_candidate_ids) > 0:
                    print(
                        f"[INFO] candidate cameras without manual room assignment are skipped: "
                        f"count={len(unassigned_candidate_ids)}"
                    )
                
                if getattr(self.sort_camera_center, "value", False):
                    for room_id, cam_ids in room_groups.items():
                        centers = np.array([self.camera_poses[idx]["position"] for idx in cam_ids], dtype=np.float32)
                        mean_center = centers.mean(axis=0)
                        room_groups[room_id] = sorted(cam_ids, key=lambda idx: float(np.linalg.norm(np.array(self.camera_poses[idx]["position"], dtype=np.float32) - mean_center)))

                return room_groups, "auto_candidate_manual_split_grouped"

            print("[WARN] no candidate cameras matched manual split assignments, fallback to auto candidates")

        room_groups = {-1: candidate_cam_ids}
        if getattr(self.sort_camera_center, "value", False):
            for room_id, cam_ids in room_groups.items():
                valid_cam_ids = [idx for idx in cam_ids if idx >= 0]
                if len(valid_cam_ids) > 0:
                    centers = np.array([self.camera_poses[idx]["position"] for idx in valid_cam_ids], dtype=np.float32)
                    mean_center = centers.mean(axis=0)
                    sorted_valid = sorted(valid_cam_ids, key=lambda idx: float(np.linalg.norm(np.array(self.camera_poses[idx]["position"], dtype=np.float32) - mean_center)))
                    invalid_cam_ids = [idx for idx in cam_ids if idx < 0]
                    room_groups[room_id] = invalid_cam_ids + sorted_valid

        return room_groups, "auto_candidate"

    def color_update_style_bak(self):
        raw_rgbs = (SH2RGB(self.gaussian_model._features_dc.squeeze()).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        # print("lock success!")

        from skimage.color import deltaE_ciede2000
        from skimage import io, color
        from internal.utils.pano_utils import direction_to_pano_coord, pano_to_img_coord

        camera_center = np.array([0,0,0])
        # # camera_center = self.colmap_cameras[3].camera_center.detach().cpu().numpy()

        # rendering = self.render_pano(camera_center)
        # pano_image = rendering["comp_rgb"]
        # pano_depth = rendering["opacity"]
        # pano_vis = rendering["visibility_filter"]
        # # Image.fromarray((pano_image.cpu().detach().numpy()[0].clip(0,1)*255).astype(np.uint8)).save("pano_image_t1.png")
        # # Image.fromarray((pano_depth.cpu().detach().numpy()[0].clip(0,1)*255).astype(np.uint8)).save("pano_depth_t1.png")
        # os.makedirs("output/temp", exist_ok=True)
        # Image.fromarray(pano_image).save("output/temp/pano.png")
        # Image.fromarray(pano_depth).save("output/temp/pano_depth.png")

        pano_origin = cv2.imread("/data/hyh/github/GaussianEditor/output/playroom/pano.png")
        # pano_origin = cv2.imread("/home/hyh/github/GaussianEditor/output/drjohnson/pano.png")
        # pano_origin = cv2.resize(pano_origin, (pano_origin.shape[1]//2, pano_origin.shape[0]//2))
        pano_origin = cv2.resize(pano_origin, (pano_origin.shape[1]*4, pano_origin.shape[0]*4))
        pano_height, pano_width  = pano_origin.shape[:2]
        pano_origin = cv2.cvtColor(pano_origin, cv2.COLOR_BGR2RGB)

        pano_styled = cv2.imread("/data/hyh/github/InstantStyle/output/PanoGaussian/result_resize_cartoon_full_both_all.png")
        # pano_styled = cv2.imread("/home/hyh/github/InstantStyle/output/PanoGaussian/result_playroom_room6_both.png")
        pano_styled = cv2.resize(pano_styled, (pano_styled.shape[1]*4, pano_styled.shape[0]*4))
        style_height, style_width  = pano_styled.shape[:2]
        pano_styled = cv2.cvtColor(pano_styled, cv2.COLOR_BGR2RGB)

        point_centers = self.gaussian_model.get_xyz
        point_centers = point_centers - torch.from_numpy(camera_center).to(point_centers)

        ### Offset Record ###
        ### Playroom: -90 0 90 ###
        ### Drjohnson: -90 0 180 ###
        
        theta_offset, phi_offset, gamma_offset = -90, 0, 90
        R = vtf.SO3.from_rpy_radians(math.radians(phi_offset), math.radians(theta_offset), math.radians(gamma_offset))
        rotated_point_centers = torch.matmul(point_centers.cpu(), torch.from_numpy(R.as_matrix()).float()).to(point_centers)

        
        # shs_view = self.gaussian.get_features.transpose(1, 2).view(
        #     -1, 3, (self.gaussian.max_sh_degree + 1) ** 2
        # )
        # dir_pp = self.gaussian.get_xyz
        # dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        # sh2rgb = eval_sh(self.gaussian.active_sh_degree, shs_view, dir_pp_normalized)
        # raw_rgbs = torch.clamp_min(sh2rgb + 0.5, 0.0)
        # raw_rgbs = (raw_rgbs*255.).detach().cpu().numpy().clip(0,255).astype(np.uint8)


        pano_coords = direction_to_pano_coord(rotated_point_centers)

        # 双线性插值
        # img_coords = pano_to_img_coord(pano_coords)
        # img_coords_floor = torch.floor(img_coords).int().detach().cpu().numpy()
        # img_coords_ceil = torch.ceil(img_coords).int().detach().cpu().numpy()
        # img_coords = img_coords.detach().cpu().numpy()
        # point_rgbs_x1y1 = pano_styled[img_coords_floor[...,0], img_coords_floor[...,1]]*(img_coords_ceil[...,0]-img_coords[...,0])[...,None]*(img_coords_ceil[...,1]-img_coords[...,1])[...,None]
        # point_rgbs_x2y1 = pano_styled[img_coords_ceil[...,0], img_coords_floor[...,1]]*(img_coords[...,0]-img_coords_floor[...,0])[...,None]*(img_coords_ceil[...,1]-img_coords[...,1])[...,None]
        # point_rgbs_x1y2 = pano_styled[img_coords_floor[...,0], img_coords_ceil[...,1]]*(img_coords_ceil[...,0]-img_coords[...,0])[...,None]*(img_coords[...,1]-img_coords_floor[...,1])[...,None]
        # point_rgbs_x2y2 = pano_styled[img_coords_ceil[...,0], img_coords_ceil[...,1]]*(img_coords[...,0]-img_coords_floor[...,0])[...,None]*(img_coords[...,1]-img_coords_floor[...,1])[...,None]

        # gt_rgbs = point_rgbs_x1y1 + point_rgbs_x2y1 + point_rgbs_x1y2 + point_rgbs_x2y2
        # gt_rgbs = gt_rgbs.clip(0, 255)
        

        # 最近邻插值
        img_coords = torch.round(pano_to_img_coord(pano_coords, width=style_width, height=style_height)).int().detach().cpu()
        point_rgbs = pano_styled[img_coords[...,0], img_coords[...,1]]

        # 双线性插值
        # img_coords = pano_to_img_coord(pano_coords)
        # img_coords_floor = torch.floor(img_coords).int().detach().cpu().numpy()
        # img_coords_ceil = torch.ceil(img_coords).int().detach().cpu().numpy()
        # img_coords = img_coords.detach().cpu().numpy()
        # point_rgbs_x1y1 = pano_styled[img_coords_floor[...,0], img_coords_floor[...,1]]*(img_coords_ceil[...,0]-img_coords[...,0])[...,None]*(img_coords_ceil[...,1]-img_coords[...,1])[...,None]
        # point_rgbs_x2y1 = pano_styled[img_coords_ceil[...,0], img_coords_floor[...,1]]*(img_coords[...,0]-img_coords_floor[...,0])[...,None]*(img_coords_ceil[...,1]-img_coords[...,1])[...,None]
        # point_rgbs_x1y2 = pano_styled[img_coords_floor[...,0], img_coords_ceil[...,1]]*(img_coords_ceil[...,0]-img_coords[...,0])[...,None]*(img_coords[...,1]-img_coords_floor[...,1])[...,None]
        # point_rgbs_x2y2 = pano_styled[img_coords_ceil[...,0], img_coords_ceil[...,1]]*(img_coords[...,0]-img_coords_floor[...,0])[...,None]*(img_coords[...,1]-img_coords_floor[...,1])[...,None]

        # point_rgbs = point_rgbs_x1y1 + point_rgbs_x2y1 + point_rgbs_x1y2 + point_rgbs_x2y2
        # point_rgbs = point_rgbs.clip(0, 255)

        fused_color = RGB2SH((torch.tensor(point_rgbs).float()/255.).cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.gaussian_model.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        # opacities = inverse_sigmoid(
        #     1.0
        #     * torch.ones(
        #         (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
        #     )
        # )

        img_coords = torch.round(pano_to_img_coord(pano_coords, width=pano_width, height=pano_height)).int().detach().cpu()
        gt_rgbs = pano_origin[img_coords[...,0], img_coords[...,1]]
        
        # color1 = color.rgb2lab(gt_rgbs)
        # color2 = color.rgb2lab(self.raw_rgbs)
        # color1 = color.rgb2yuv(gt_rgbs)
        # color2 = color.rgb2yuv(self.raw_rgbs)
        color1 = gt_rgbs
        color2 = raw_rgbs
        distance = deltaE_ciede2000(color1, color2)
        # pano_vis = distance < self.diff_slider.value

        _features_dc = torch.nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        _features_rest = torch.nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )

        with torch.no_grad():
            self.gaussian_model._features_dc = _features_dc
            self.gaussian_model._features_rest = _features_rest
        
    def color_update_proj(self, camera_center_path, camera_rotation, styled_img_path, depth_path, point_mask, upscale=1, color_gaussian=True, room_scope_mask=None):
        from internal.utils.pano_utils import direction_to_pano_coord, pano_to_img_coord

        camera_center = np.load(camera_center_path)

        pano_styled = cv2.imread(styled_img_path)
        pano_styled = cv2.resize(pano_styled, (pano_styled.shape[1]*upscale, pano_styled.shape[0]*upscale))
        height, width  = pano_styled.shape[:2]
        pano_styled = cv2.cvtColor(pano_styled, cv2.COLOR_BGR2RGB)

        # pano_depth = np.load(depth_path)
        # pano_depth = cv2.resize(pano_depth, (width, height))

        point_centers = self.gaussian_model.get_xyz
        point_centers = point_centers - torch.from_numpy(camera_center).to(point_centers)

        ### Offset Record ###
        ### Playroom: -90 0 90 ###
        ### Drjohnson: -90 0 180 ###
        
        theta_offset, phi_offset, gamma_offset = -90, 0, 90
        R = camera_rotation @ vtf.SO3.from_rpy_radians(math.radians(phi_offset), math.radians(theta_offset), math.radians(gamma_offset)).as_matrix()
        rotated_point_centers = torch.matmul(point_centers.cpu(), torch.from_numpy(R).float()).to(point_centers)

        pano_coords = direction_to_pano_coord(rotated_point_centers)

        # 最近邻插值
        img_coords = torch.round(pano_to_img_coord(pano_coords, width=width, height=height)).int().detach().cpu()
        point_rgbs = pano_styled[img_coords[...,0], img_coords[...,1]]

        # 双线性插值
        # img_coords = pano_to_img_coord(pano_coords)
        # img_coords_floor = torch.floor(img_coords).int().detach().cpu().numpy()
        # img_coords_ceil = torch.ceil(img_coords).int().detach().cpu().numpy()
        # img_coords = img_coords.detach().cpu().numpy()
        # point_rgbs_x1y1 = pano_styled[img_coords_floor[...,0], img_coords_floor[...,1]]*(img_coords_ceil[...,0]-img_coords[...,0])[...,None]*(img_coords_ceil[...,1]-img_coords[...,1])[...,None]
        # point_rgbs_x2y1 = pano_styled[img_coords_ceil[...,0], img_coords_floor[...,1]]*(img_coords[...,0]-img_coords_floor[...,0])[...,None]*(img_coords_ceil[...,1]-img_coords[...,1])[...,None]
        # point_rgbs_x1y2 = pano_styled[img_coords_floor[...,0], img_coords_ceil[...,1]]*(img_coords_ceil[...,0]-img_coords[...,0])[...,None]*(img_coords[...,1]-img_coords_floor[...,1])[...,None]
        # point_rgbs_x2y2 = pano_styled[img_coords_ceil[...,0], img_coords_ceil[...,1]]*(img_coords[...,0]-img_coords_floor[...,0])[...,None]*(img_coords[...,1]-img_coords_floor[...,1])[...,None]

        # point_rgbs = point_rgbs_x1y1 + point_rgbs_x2y1 + point_rgbs_x1y2 + point_rgbs_x2y2
        # point_rgbs = point_rgbs.clip(0, 255)

        # point_depth = torch.linalg.norm(point_centers, axis=-1).detach().cpu().numpy()
        # pano_depth = pano_depth[...,0][img_coords[...,0], img_coords[...,1]]
        # depth_thres = 1
        # pano_vis = point_depth < (pano_depth+depth_thres)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(point_centers.detach().cpu().numpy())
        # # o3d.io.write_point_cloud("point_cloud.pcd", pcd)
        # diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
        # pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        # _, pt_map = pcd.hidden_point_removal(o3d.core.Tensor(camera_center, o3d.core.float32), diameter*100)
        # pano_vis = pt_map.numpy()

        # pano_vis = np.load(point_mask).astype(np.bool_)
        pano_vis = np.asarray(point_mask).astype(np.bool_)
        point_count = self.gaussian_model.get_xyz.shape[0]
        if pano_vis.shape[0] != point_count:
            raise ValueError(f"point_mask size mismatch: {pano_vis.shape[0]} vs {point_count}")
        if room_scope_mask is not None:
            room_scope_mask = np.asarray(room_scope_mask).astype(np.bool_)
            if room_scope_mask.shape[0] != point_count:
                raise ValueError(f"room_scope_mask size mismatch: {room_scope_mask.shape[0]} vs {point_count}")
            pano_vis = pano_vis & room_scope_mask

        # pano_vis = None
        # masks = np.load(mask_path)
        # for mask in masks:
        #     if pano_vis is None: pano_vis = mask>self.threshold_slider.value
        #     else: pano_vis = pano_vis | (mask>self.threshold_slider.value)

        fused_color = RGB2SH((torch.tensor(point_rgbs).float()/255.).cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.gaussian_model.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        img_coords = torch.round(pano_to_img_coord(pano_coords, width=width, height=height)).int().detach().cpu()

        _features_dc = torch.nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        _features_rest = torch.nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )

        if color_gaussian:
            with torch.no_grad():
                # self.gaussian_model._features_dc = _features_dc
                # self.gaussian_model._features_rest = _features_rest

                self.gaussian_model._features_dc[pano_vis] = _features_dc[pano_vis]
                self.gaussian_model._features_rest[pano_vis] = _features_rest[pano_vis]
        # return _features_dc, _features_rest
        return features[:, :, 0:1].transpose(1, 2), features[:, :, 1:].transpose(1, 2)
    
    def hidden_color_propogation(self, point_mask_path=None, point_mask=None, room_scope_mask=None):
        print("Propogating Color...")

        if point_mask is None:
            if point_mask_path is None:
                raise ValueError("Either point_mask_path or point_mask must be provided")
            point_mask = np.load(point_mask_path).astype(np.bool_)
        else:
            point_mask = np.asarray(point_mask).astype(np.bool_)

        point_count = self.gaussian_model.get_xyz.shape[0]
        if point_mask.shape[0] != point_count:
            raise ValueError(f"point_mask size mismatch: {point_mask.shape[0]} vs {point_count}")

        if room_scope_mask is not None:
            room_scope_mask = np.asarray(room_scope_mask).astype(np.bool_)
            if room_scope_mask.shape[0] != point_count:
                raise ValueError(f"room_scope_mask size mismatch: {room_scope_mask.shape[0]} vs {point_count}")
            visible_mask = point_mask & room_scope_mask
            hidden_mask = (~point_mask) & room_scope_mask
        else:
            visible_mask = point_mask
            hidden_mask = ~point_mask

        if not np.any(visible_mask):
            print("[INFO] skip propagation: no visible points in current scope")
            return
        if not np.any(hidden_mask):
            print("[INFO] skip propagation: no hidden points in current scope")
            return

        # point_centers = self.gaussian_model.get_xyz
        # _, nn_idx = K_nearest_neighbors(point_centers[point_mask], 5, point_centers[~point_mask])
        # nn_feat_dc = self.gaussian_model._features_dc[point_mask][nn_idx]
        # nn_feat_rest = self.gaussian_model._features_rest[point_mask][nn_idx]
        
        if self.knn_number.value <= 0:
            print("[INFO] skip propagation: KNN number <= 0")
            return

        point_color = self.raw_features_dc[:,0,:]
        _, nn_idx = K_nearest_neighbors(point_color[visible_mask], self.knn_number.value, point_color[hidden_mask])
        nn_feat_dc = self.gaussian_model._features_dc[visible_mask][nn_idx]
        nn_feat_rest = self.gaussian_model._features_rest[visible_mask][nn_idx]

        mean_feat_dc = nn_feat_dc.mean(axis=1)#, keepdim=True)
        mean_feat_rest = nn_feat_rest.mean(axis=1)#, keepdim=True)
        if self.knn_number.value==1:
            mean_feat_dc = mean_feat_dc.unsqueeze(1)
            mean_feat_rest = mean_feat_rest.unsqueeze(1)
        with torch.no_grad():
            self.gaussian_model._features_dc[hidden_mask] = mean_feat_dc
            self.gaussian_model._features_rest[hidden_mask] = mean_feat_rest

        # colors = SH2RGB(self.gaussian_model._features_dc)
        # recolored = nnfm_utils.match_colors_for_point_cloud(colors, colors[point_mask])[0]
        # recolored_sh = RGB2SH(recolored)
        # with torch.no_grad():
        #     self.gaussian_model._features_dc[~point_mask] = recolored_sh[~point_mask]
        #     self.gaussian_model._features_rest[~point_mask] = recolored_sh[~point_mask]

        self.update_client()
        print("Propogating Success...")

        # print("KNN Time Cost: " + str(time.time()-start_time))
        pass

    def _zero_feature_grads_outside_room(self, room_scope_mask_t: Optional[torch.Tensor]):
        if room_scope_mask_t is None:
            return

        with torch.no_grad():
            outside_mask = ~room_scope_mask_t
            if self.gaussian_model._features_dc.grad is not None:
                self.gaussian_model._features_dc.grad[outside_mask] = 0
            if self.gaussian_model._features_rest.grad is not None:
                self.gaussian_model._features_rest.grad[outside_mask] = 0

    def retrain_scene(self, save_dir):
        opt = OptimizationParams(
            parser = ArgumentParser(description="Training script parameters"),
            max_steps= self.train_steps.value,
            color_lr_scaler = self.color_lr_scaler.value,
        )
        opt = OmegaConf.create(vars(opt))
        self.gaussian_model.training_setup(opt)
        self.gaussian_model.max_radii2D = torch.zeros((self.gaussian_model.get_xyz.shape[0]), device=self.device)

        def projected_area(scaling):
            s0 = scaling[0]
            s1 = scaling[1]
            return s0 * s1

        def shorten_scaling(scaling, threshold):
            s0 = scaling[0]
            s1 = scaling[1]
            a, b = max(s0, s1), min(s0, s1)
            ratio = a / b
            if ratio > threshold:
                new_a = b + (a - b) * 0.5

                if scaling[0] == a:
                    scaling[0] = new_a
                else:
                    scaling[1] = new_a
        
        print('Number of Gaussians:', self.gaussian_model._features_dc.shape[0])

        viewpoint_stack = None
        n_retraining = 10000000 # a large number
        dont_split_when_above = 3250000
        for retraining in range(n_retraining):
            if self.gaussian_model._features_dc.shape[0] > dont_split_when_above:
                break
            # if pretrain_until_at_least is not None and self.gaussian_model._features_dc.shape[0] > pretrain_until_at_least:
            #     break
        
            progress_bar = tqdm(range(0, 2000), desc=f"Retraining progress {retraining + 1}")

            split_max_n = 0.5
            split_std_threshold = 1.0
            split_std_multiplier = 1.0
            with torch.no_grad():
                if split_max_n > 0.0:
                    scaling = self.gaussian_model.get_scaling.cpu().detach().numpy()
                    n_gaussians = self.gaussian_model._features_dc.shape[0]
                
                    projected_areas = np.zeros(scaling.shape[0], np.float32)
                    for i in range(scaling.shape[0]):
                        projected_areas[i] = projected_area(scaling[i])
                    
                    areas_mean = np.mean(projected_areas)
                    areas_std = np.std(projected_areas)
                    
                    std_split_threshold = areas_mean + areas_std * split_std_threshold * (split_std_multiplier ** retraining)
                    split_counter = np.count_nonzero(projected_areas > std_split_threshold)
                    print('split_counter', split_counter)
                    if split_counter > split_max_n * n_gaussians:
                        sorted_projected_areas = np.sort(projected_areas)
                        percentage_split_threshold = sorted_projected_areas[n_gaussians - int(split_max_n * n_gaussians)]
                        split_threshold = percentage_split_threshold
                    else:
                        split_threshold = std_split_threshold
                    
                    gaussians_to_split = torch.empty(n_gaussians, dtype=torch.bool, device='cpu')
                    for i in range(n_gaussians):
                        gaussians_to_split[i] = bool(projected_areas[i] > split_threshold)
                    gaussians_to_split = gaussians_to_split.cuda()
                    self.gaussian_model.densify_and_split_with_mask(gaussians_to_split, N=4, scaling_factor=0.8)

            self.update_client()

            retrain_iterations = 2000
            shorten_scaling_iter = 250
            shorten_scaling_max_iter = 1000
            shorten_scaling_threshold = 1.5
            lambda_dssim = 0.2
            for iteration in range(1, retrain_iterations + 1):
                if iteration % shorten_scaling_iter == 0 and iteration <= shorten_scaling_max_iter:
                    with torch.no_grad():
                        scaling = self.gaussian_model.get_scaling.cpu().detach().numpy()
                        for i in range(scaling.shape[0]):
                            shorten_scaling(scaling[i], shorten_scaling_threshold)
                        self.gaussian_model._scaling[:, :] = self.gaussian_model.scaling_inverse_activation(torch.from_numpy(scaling))

                self.gaussian_model.optimizer.zero_grad()
                self.gaussian_model.update_learning_rate(30000 - retrain_iterations + iteration)

                # Pick a random Camera
                if not viewpoint_stack:
                    viewpoint_stack = self.colmap_cameras.copy()
                viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1))

                results = self.render(# viewpoint_cam.R, viewpoint_cam.T,
                                        #   viewpoint_cam.FoVx, viewpoint_cam.FoVy,
                                        #   viewpoint_cam.image_width, viewpoint_cam.image_height
                                          viewpoint_cam.R, viewpoint_cam.T,
                                          viewpoint_cam.FovX, viewpoint_cam.FovY,
                                          viewpoint_cam.width, viewpoint_cam.height,
                                    )
                image = results['render'].unsqueeze(0)


                gt_image = torch.from_numpy(np.array(viewpoint_cam.image)).cuda().permute(2,0,1).unsqueeze(0).float()
                Ll1 = l1_loss(image, gt_image)
                content_loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt_image))

                loss = content_loss

                loss.backward()
                self.gaussian_model.optimizer.step()

                with torch.no_grad():
                    # Progress bar
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                    if iteration % 10 == 0:
                        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                        progress_bar.update(10)
                        self.update_client()
                    if iteration == retrain_iterations:
                        progress_bar.close()
                        
        print('Number of Gaussians after pretraining:', self.gaussian_model._features_dc.shape[0])
        self.gaussian_model.save_ply(os.path.join(save_dir, 'scene_retrained.ply'))

    def color_update(self, cam_id, save_dir, room_point_mask=None, room_id=None): #camera_center_path, styled_img_path):
        print("Reading and processing images...")

        room_scope_mask = None
        if room_point_mask is not None:
            room_scope_mask = np.asarray(room_point_mask).astype(np.bool_)
            if room_scope_mask.shape[0] != self.gaussian_model.get_xyz.shape[0]:
                raise ValueError(
                    f"room scope mask size mismatch: {room_scope_mask.shape[0]} vs {self.gaussian_model.get_xyz.shape[0]}"
                )

        room_tag = f"room_{room_id}" if room_id is not None else "all"
        if room_id is not None and room_id != -1 and room_scope_mask is None:
            raise ValueError(f"room_{room_id} stylization requires a valid room_point_mask")
        print(f"[INFO] color_update start for {room_tag}, cameras={len(cam_id)}")

        # sobel_operator = SobelOperator().to(self.device)
        # gray_scale = RGB2Gray().to(self.device)
        # perceptual = PerceptualLoss().eval().to(self.device)

        pers_params = [
            (90, 0,   0),
            (90, 90,  0),
            (90, 180, 0),
            (90, 270, 0),
        
            (90, 0, 90),
            (90, 0, -90),
        ]
        
        def get_pano_imgs_tensor(equ):
            images = []
            for p in pers_params:
                image = equ.GetPerspective(p[0], p[1], p[2], res, res)  # Specify parameters(FOV, theta, phi, height, width)
                image = np.clip(image, 0, 255)
                images.append(image[...,::-1])
            image_tensor = torch.tensor(images).permute(0,3,1,2).float()/255.0
            # image_tensor = torch.tensor(images).float()/255.0
            # if recolor:
            #     style_image = torch.from_numpy(np.array(Image.open(self.style_img))).permute(1,0,2).float().contiguous()
            #     recolored_image = bwhc_to_bcwh(nnfm_utils.match_colors_for_image_set(image_tensor, style_image)[0])
            image_tensor = image_tensor#.to("cuda:1")
            return image_tensor
        
        def get_pano_imgs(equ):
            images = []
            for p in pers_params:
                image = equ.GetPerspective(p[0], p[1], p[2], res, res)  # Specify parameters(FOV, theta, phi, height, width)
                image = np.clip(image, 0, 255)
                images.append(image[...,::-1])
            # image_tensor = torch.tensor(images).permute(0,3,1,2).float()/255.0
            # image_tensor = image_tensor#.to("cuda:1")
            return images
        
        def train_scene(styled_imgs, steps):
            # equ_styled = E2P.Equirectangular(pano_styled)
            # train_steps = 1500 if steps==0 else self.train_steps.value
            train_steps = self.train_steps.value + 20*len(styled_imgs)

            for step in tqdm(range(train_steps)):
                self.gaussian_model.update_learning_rate(step+steps)

                theta = random.random()*360
                phi = (random.random()-0.5)*180

                if step > 200:
                    random_img = random.sample(styled_imgs, 1)[0]
                else:
                    random_img = styled_imgs[-1]
                # random_img = styled_imgs[-1]
                camera_center, equ_styled, equ_raw = random_img['camera_center'], random_img['styled_img'], random_img['raw_img']

                R = vtf.SO3.from_rpy_radians(math.radians(phi), math.radians(theta), math.radians(0)).as_matrix()
                # Rq = tf.SO3.from_rpy_radians(math.radians(phi), math.radians(theta), math.radians(-90)) # For drjohnson
                T = [0,0,0]
                trans = camera_center

                _gt_image = equ_styled.GetPerspective(fov, theta, phi, res, res)  # Specify parameters(FOV, theta, phi, height, width)
                _gt_image = np.clip(_gt_image, 0, 255).astype(np.uint8)
                _gt_image = _gt_image[...,::-1]
                gt_image = torch.from_numpy(np.array(_gt_image)[None,]).float()/255.0
                gt_image = gt_image[0].to("cuda")
                # Image.fromarray((gt_image*255).detach().cpu().numpy().astype(np.uint8)).save("temp2.png")
                # gt_image_gray = gray_scale(gt_image)
                # gt_gradient_magnitude, gt_gradient_direction = sobel_operator(col_render_gray)

                _raw_image = equ_raw.GetPerspective(fov, theta, phi, res, res)  # Specify parameters(FOV, theta, phi, height, width)
                _raw_image = np.clip(_raw_image, 0, 255).astype(np.uint8)
                _raw_image = _raw_image[...,::-1]
                raw_image = torch.from_numpy(np.array(_raw_image)[None,]).float()/255.0
                raw_image = raw_image[0].to("cuda")
                # Image.fromarray((raw_image*255).detach().cpu().numpy().astype(np.uint8)).save("temp1.png")

                # raw_image_gray = gray_scale(raw_image.permute(2,0,1)[None])
                # raw_gradient_magnitude, raw_gradient_direction = sobel_operator(raw_image_gray)

                cam = Simple_Camera(0, R, T, math.radians(90), math.radians(90),
                                    res, res, "", 0, trans=trans)

                render_params = {
                    'active_sh_degree': self.active_sh_degree_slider.value if hasattr(self, 'active_sh_degree_slider') else 0, 
                    'scaling_modifier': self.scale_slider.value, 
                    'depth_ratio': self.depth_ratio_slider.value,
                    'bg_color': self.viewer_renderer.background_color,
                    'sparsity': self.sparsity_slider.value, 
                    'valid_range': None,
                    'show_ptc': self.enable_ptc.value and (self.surfel_mode.value == 'ptc'),
                    'show_disk': self.enable_ptc.value and (self.surfel_mode.value == 'disk'),
                    'point_size': self.point_size.value,
                    # 'override_color': SH2RGB(self.gaussian_model._features_dc.squeeze()),
                }

                results = self.viewer_renderer.render_viewer(
                    cam,
                    **render_params,
                )
                
                # rendered = results['render'].clip(0,1).permute(1, 2, 0)
                rendered = results['render'].permute(1, 2, 0)

                # Image.fromarray((rendered*255).detach().cpu().numpy().astype(np.uint8)).save("temp.png")
                # rendered_gray = gray_scale(rendered.permute(2,0,1)[None])
                # rendered_gradient_magnitude, rendered_gradient_direction = sobel_operator(rendered_gray)

                # sobel_loss = torch.nn.functional.mse_loss(rendered_gradient_magnitude, raw_gradient_magnitude)
                color_loss = torch.nn.functional.l1_loss(rendered.permute(2,0,1)[None], gt_image.permute(2,0,1)[None])
                # perceptual_loss = perceptual(rendered.permute(2,0,1)[None], raw_image.permute(2,0,1)[None])
                loss = color_loss #+ perceptual_loss#+ sobel_loss #+ perceptual_loss

                if step%50==0:
                    print("total_loss:", loss.data,
                          "\t color_loss:", color_loss.data,
                        #   "\t sobel_loss:", sobel_loss.data,
                        #   "\t perceptual_loss:", perceptual_loss.data
                          )
                    self.update_client()
                
                loss.backward(retain_graph=True)

                # # Densification
                # visibility_filter = results["visibility_filter"].to(self.device)
                # viewspace_point_tensor = results["viewspace_points"].to(self.device)
                # radii = results["radii"].to(self.device)
                # if step < train_steps//2:
                #     self.gaussian_model.max_radii2D[visibility_filter] = torch.max(self.gaussian_model.max_radii2D[visibility_filter], radii[visibility_filter])
                #     self.gaussian_model.add_densification_stats(viewspace_point_tensor, visibility_filter)

                #     if step > train_steps//5 and step % train_steps//25 == 0:
                #         size_threshold = 20
                #         self.gaussian_model.densify_and_prune(0.0002, 0.05, self.scene.nerf_normalization["radius"], size_threshold)
                    
                #     if step % train_steps//10 == 0:
                #         self.gaussian_model.reset_opacity()

                self.gaussian_model.optimizer.step()
                self.gaussian_model.optimizer.zero_grad(set_to_none=True)
        
        def train_pano(styled_imgs, feat_dc, feat_rest, steps, random_flag=False):
            def calculate_total_variation_loss(x, p=1):
                batch_size = x.size(0)
        
                # 1. 计算垂直方向的梯度 (H 维度)
                # 上下方向不相连（顶部是北极，底部是南极），所以使用普通的差值计算
                diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
                
                # 2. 计算水平方向的梯度 (W 维度)
                # 全景图左右相连，所以使用 torch.roll 进行循环移位计算差值
                # shifts=-1 表示整个张量向左移动1格，最左侧的列会被移到最右侧
                diff_w = torch.roll(x, shifts=-1, dims=-1) - x
                
                # 计算 Loss
                if p == 1:
                    # L1 范数（通常对保护边缘更有效）
                    tv_h = torch.sum(torch.abs(diff_h))
                    tv_w = torch.sum(torch.abs(diff_w))
                elif p == 2:
                    # L2 范数（平方）
                    tv_h = torch.sum(torch.pow(diff_h, 2))
                    tv_w = torch.sum(torch.pow(diff_w, 2))
                else:
                    raise ValueError("p must be 1 or 2")
                    
                # 平均到每个 batch (也可以根据需求平均到所有像素点)
                tv_loss = (tv_h + tv_w) / batch_size
                
                return tv_loss
            
            def calculate_total_variation_loss_spherical(x, p=1, reduction='mean'):
                B, C, H, W = x.shape
                device = x.device
                dtype = x.dtype

                # ==========================================
                # 1. 计算水平方向的 TV Loss (经度方向，带环形边界)
                # ==========================================
                # 极角 theta 范围 [0, pi]，对应图像的第 0 行到第 H-1 行
                # 取像素中心的极角：(i + 0.5) * (pi / H)
                theta_w = (torch.arange(H, dtype=dtype, device=device) + 0.5) * (math.pi / H)
                # 计算权重 sin(theta) 并调整形状为 (1, 1, H, 1) 以便在 H 维度上广播
                weight_w = torch.sin(theta_w).view(1, 1, H, 1)
                
                # 水平像素差值 (向左平移 1 像素，实现首尾相接)
                diff_w = torch.roll(x, shifts=-1, dims=-1) - x
                
                # ==========================================
                # 2. 计算垂直方向的 TV Loss (纬度方向)
                # ==========================================
                # 垂直方向计算的是第 i 行和第 i+1 行的差值
                # 它们之间的交界线对应的极角为：(i + 1.0) * (pi / H)
                theta_h = (torch.arange(H - 1, dtype=dtype, device=device) + 1.0) * (math.pi / H)
                weight_h = torch.sin(theta_h).view(1, 1, H - 1, 1)
                
                # 垂直像素差值 (上下不相接)
                diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]

                # ==========================================
                # 3. 应用范数与权重
                # ==========================================
                if p == 1:
                    loss_w = torch.abs(diff_w) * weight_w
                    loss_h = torch.abs(diff_h) * weight_h
                elif p == 2:
                    loss_w = torch.pow(diff_w, 2) * weight_w
                    loss_h = torch.pow(diff_h, 2) * weight_h
                else:
                    raise ValueError("Parameter 'p' must be 1 or 2.")

                # ==========================================
                # 4. 汇总与归一化
                # ==========================================
                total_loss = torch.sum(loss_w) + torch.sum(loss_h)

                if reduction == 'mean':
                    # 归一化：除以像素总数，使其对不同的分辨率 (如 1k, 2k, 4k) 具有可比性
                    # 注意：这里我们除以 B*C*H*W，使得损失量级与普通 2D 图像的 mean TV Loss 类似
                    total_loss = total_loss / (B * C * H * W)
                    
                return total_loss
            
            # equ_styled = E2P.Equirectangular(pano_styled)
            # train_steps = 1500 if steps==0 else self.train_steps.value
            train_steps = steps

            room_scope_mask_np = None
            room_scope_mask_t = None
            if room_scope_mask is not None:
                room_scope_mask_np = np.asarray(room_scope_mask).astype(np.bool_)
                if room_scope_mask_np.shape[0] != self.gaussian_model.get_xyz.shape[0]:
                    raise ValueError(
                        f"room_scope_mask size mismatch in train_pano: {room_scope_mask_np.shape[0]} vs {self.gaussian_model.get_xyz.shape[0]}"
                    )
                room_scope_mask_t = torch.from_numpy(room_scope_mask_np).to(
                    device=self.gaussian_model._features_dc.device,
                    dtype=torch.bool,
                )

            # point_centers = self.gaussian_model.get_xyz
            # _, nn_idx = K_nearest_neighbors(point_centers[all_point_mask], 5, point_centers[~all_point_mask])

            visible_mask = all_point_mask
            if room_scope_mask_np is not None:
                visible_mask = visible_mask & room_scope_mask_np
                hidden_mask = room_scope_mask_np & (~visible_mask)
            else:
                hidden_mask = ~visible_mask

            has_visible = bool(np.any(visible_mask))
            has_hidden = bool(np.any(hidden_mask))

            point_color = self.raw_features_dc[:,0,:]
            nn_idx = None
            if has_visible and has_hidden and self.knn_number.value > 0:
                _, nn_idx = K_nearest_neighbors(point_color[visible_mask], self.knn_number.value, point_color[hidden_mask])

            # nn_feat_dc = self.gaussian_model._features_dc[point_mask][nn_idx]
            # nn_feat_rest = self.gaussian_model._features_rest[point_mask][nn_idx]

            # mean_feat_dc = nn_feat_dc.mean(axis=1)
            # mean_feat_rest = nn_feat_rest.mean(axis=1)

            for step in tqdm(range(train_steps)):
                self.gaussian_model.update_learning_rate(step+steps)

                # theta = random.random()*360
                # phi = (random.random()-0.5)*180

                # if step > 50:
                #     random_img = random.sample(styled_imgs, 1)[0]
                # else:
                #     random_img = styled_imgs[-1]
                
                # random_img = random.sample(styled_imgs, 1)[0] if random_flag else styled_imgs[-1]

                if random_flag and step >= 100:
                    random_img = random.sample(styled_imgs, 1)[0]
                else:
                    random_img = styled_imgs[-1]

                # random_img = styled_imgs[-1]

                # camera_center, equ_styled, equ_raw, equ_mask = random_img['camera_center'], random_img['styled_img'], random_img['raw_img'], random_img['pano_mask']
                # camera_center, equ_styled, equ_raw, equ_mask = \
                #     random_img['camera_center'], random_img['styled_img_tensor'].to(self.device), random_img['raw_img_tensor'].to(self.device), random_img['pano_mask_tensor'].to(self.device)
                # cam_mask, feat_dc_visible, feat_rest_visible = \
                #     random_img['cam_mask'], random_img['feat_dc_visible'], random_img['feat_rest_visible']
                # camera_center, equ_styled, equ_raw, equ_mask = \
                #     random_img['camera_center'], random_img['styled_img_tensor'].to(self.device), random_img['raw_img_tensor'].to(self.device), random_img['pano_mask_tensor'].to(self.device)
                camera_center, camera_rotation, equ_styled, equ_mask = \
                    random_img['camera_center'], random_img['camera_rotation'], random_img['styled_img_tensor'], random_img['pano_mask_tensor']

                loss = 0
                color_loss = 0
                ssim_loss = 0
                tv_loss = 0
                propagate_loss = 0
                project_loss = 0
                for idx, params in enumerate(pers_params):
                    fov, theta, phi = params
                    R = camera_rotation @ vtf.SO3.from_rpy_radians(math.radians(phi), math.radians(theta), math.radians(0)).as_matrix()
                    # Rq = tf.SO3.from_rpy_radians(math.radians(phi), math.radians(theta), math.radians(-90)) # For drjohnson
                    T = [0,0,0]
                    trans = camera_center

                    cam = Simple_Camera(0, R, T, math.radians(90), math.radians(90),
                                        res, res, "", 0, trans=trans)

                    render_params = {
                        'active_sh_degree': self.active_sh_degree_slider.value if hasattr(self, 'active_sh_degree_slider') else 0, 
                        'scaling_modifier': self.scale_slider.value, 
                        'depth_ratio': self.depth_ratio_slider.value,
                        'bg_color': self.viewer_renderer.background_color,
                        'sparsity': self.sparsity_slider.value, 
                        'valid_range': None,
                        'show_ptc': self.enable_ptc.value and (self.surfel_mode.value == 'ptc'),
                        'show_disk': self.enable_ptc.value and (self.surfel_mode.value == 'disk'),
                        'point_size': self.point_size.value,
                        'override_color': SH2RGB(self.gaussian_model._features_dc.squeeze()),
                    }

                    
                    results = self.viewer_renderer.render_viewer(
                        cam,
                        **render_params,
                    )
                    rendered = results['render'].permute(1, 2, 0)
                    # Image.fromarray((rendered*255).detach().cpu().numpy().astype(np.uint8)).save("temp.png")
                    # rendered_gray = gray_scale(rendered.permute(2,0,1)[None])
                    # rendered_gradient_magnitude, rendered_gradient_direction = sobel_operator(rendered_gray)

                    # sobel_loss = torch.nn.functional.mse_loss(rendered_gradient_magnitude, raw_gradient_magnitude)

                    color_loss += l1_loss(equ_mask[idx].to(self.device) * rendered.permute(2,0,1)[None], equ_mask[idx].to(self.device) * equ_styled[idx][None].to(self.device))
                    ssim_loss += 1.0 - ssim(equ_mask[idx].to(self.device) * rendered.permute(2,0,1)[None], equ_mask[idx].to(self.device) * equ_styled[idx][None].to(self.device))
                    # color_loss += torch.nn.functional.l1_loss(rendered.permute(2,0,1)[None], equ_styled[idx][None])
                    tv_loss += calculate_total_variation_loss_spherical(rendered.permute(2,0,1)[None])
                    
                    torch.cuda.empty_cache()
                    # perceptual_loss = perceptual(rendered.permute(2,0,1)[None], raw_image.permute(2,0,1)[None])
                    # loss += color_loss #+ perceptual_loss#+ sobel_loss #+ perceptual_loss

                if has_visible:
                    project_loss = l1_loss(self.gaussian_model._features_dc[visible_mask], feat_dc[visible_mask]) + \
                                    l1_loss(self.gaussian_model._features_rest[visible_mask],feat_rest[visible_mask])

                if has_visible and has_hidden and nn_idx is not None:
                    nn_feat_dc = self.gaussian_model._features_dc[visible_mask][nn_idx]
                    nn_feat_rest = self.gaussian_model._features_rest[visible_mask][nn_idx]
                    mean_feat_dc = nn_feat_dc.mean(axis=1)
                    mean_feat_rest = nn_feat_rest.mean(axis=1)
                    if self.knn_number.value==1:
                        mean_feat_dc = mean_feat_dc.unsqueeze(1)
                        mean_feat_rest = mean_feat_rest.unsqueeze(1)
                    propagate_loss = l1_loss(self.gaussian_model._features_dc[hidden_mask], mean_feat_dc) + \
                                    l1_loss(self.gaussian_model._features_rest[hidden_mask], mean_feat_rest)

                loss = 0.8*color_loss/6 + 0.2*ssim_loss/6 + 1*propagate_loss + 1*project_loss + 1e-3*tv_loss/6
                # loss = 1*color_loss + 1*propagate_loss + 10*project_loss

                if step%20==0:
                    print("total_loss:", loss.data,
                          "\t color_loss:", color_loss.data,
                          "\t ssim_loss:", ssim_loss.data,
                          "\t tv_loss:", tv_loss.data,
                          "\t propagate_loss:", propagate_loss.data,
                          "\t project_loss:", project_loss.data,
                        #   "\t sobel_loss:", sobel_loss.data,
                        #   "\t perceptual_loss:", perceptual_loss.data
                          )
                    self.update_client()
                torch.cuda.empty_cache()
                loss.backward(retain_graph=True)
                self._zero_feature_grads_outside_room(room_scope_mask_t)
                self.gaussian_model.optimizer.step()
                self.gaussian_model.optimizer.zero_grad(set_to_none=True)

                # del equ_styled
                # del equ_raw
                # del equ_mask
                # del rendered
        
        def color_matching():
            # viewpoint_stack_for_recoloring = scene.getTrainCameras().copy()

            style_image = Image.open(self.style_img)
            loader = transforms.Compose([transforms.ToTensor()])

            style_image=loader(style_image).unsqueeze(0)
            style_image = style_image.to('cuda', torch.float)

            recolored_viewpoint_renderings = []

            lambda_dssim = 0.2
            
            with torch.no_grad():
                for i in range(len(self.colmap_cameras)):
                    viewpoint_cam = self.colmap_cameras[i]        
                    # bg = torch.rand((3), device="cuda") if opt.random_background else background
                    
                    results = self.render(viewpoint_cam.R, viewpoint_cam.T,
                                          viewpoint_cam.FovX, viewpoint_cam.FovY,
                                          viewpoint_cam.width, viewpoint_cam.height)
                    # Image.fromarray((results['render'].clip(0,1).permute(1, 2, 0)*255).detach().cpu().numpy().astype(np.uint8)).save("capture.png")
                    # print("save current view success.")
                    image = results['render'].unsqueeze(0)

                    recolored_image = bwhc_to_bcwh(nnfm_utils.match_colors_for_image_set(bcwh_to_bwhc(image), bcwh_to_bwhc(style_image)[0])[0])
                    recolored_viewpoint_renderings.append(recolored_image.cpu().detach())
                
                # gaussians._features_secondary_dc.data[:, :, 0] = ((gaussians._features_secondary_dc.data[:, :, 0] + gaussians._features_secondary_dc.data[:, :, 1] + gaussians._features_secondary_dc.data[:, :, 2]) / 3.0)
                # gaussians._features_secondary_dc.data[:, :, 1] = ((gaussians._features_secondary_dc.data[:, :, 0] + gaussians._features_secondary_dc.data[:, :, 1] + gaussians._features_secondary_dc.data[:, :, 2]) / 3.0)
                # gaussians._features_secondary_dc.data[:, :, 2] = ((gaussians._features_secondary_dc.data[:, :, 0] + gaussians._features_secondary_dc.data[:, :, 1] + gaussians._features_secondary_dc.data[:, :, 2]) / 3.0)
                    
            color_matching_optimizer = torch.optim.Adam([self.gaussian_model._features_dc], lr=0.01)
            
            viewpoint_stack_for_recoloring = []
            for recoloring_iteration in range(1, 30000):
                color_matching_optimizer.zero_grad()

                # Pick a random Camera
                if not viewpoint_stack_for_recoloring:
                    viewpoint_stack_for_recoloring = self.colmap_cameras.copy()
                    recolored_renderings = recolored_viewpoint_renderings.copy()
                random_cam_id = random.randint(0, len(viewpoint_stack_for_recoloring)-1)
                viewpoint_cam = viewpoint_stack_for_recoloring.pop(random_cam_id)
                gt_image = recolored_renderings.pop(random_cam_id).cuda()

                # bg = torch.rand((3), device="cuda") if opt.random_background else background

                results = self.render(viewpoint_cam.R, viewpoint_cam.T,
                                          viewpoint_cam.FovX, viewpoint_cam.FovY,
                                          viewpoint_cam.width, viewpoint_cam.height)
                image = results['render'].unsqueeze(0)
                                          
                Ll1 = l1_loss(image, gt_image)
                content_loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt_image))

                loss = content_loss

                loss.backward()
                color_matching_optimizer.step()

        # color_matching()
        
        fov = 90
        res = 1024

        opt = OptimizationParams(
            parser = ArgumentParser(description="Training script parameters"),
            max_steps= self.train_steps.value,
            color_lr_scaler = self.color_lr_scaler.value,
        )
        opt = OmegaConf.create(vars(opt))
        self.gaussian_model.max_radii2D = torch.zeros((self.gaussian_model.get_xyz.shape[0]), device=self.device)

        styled_imgs = []
        camera_center = np.array([0,0,0])
        steps = 0
        feat_dc_visible, feat_rest_visible = None, None
        for i in range(len(cam_id)):
            idx = cam_id[i]
            camera_center = np.load(os.path.join(save_dir, f'cam_{idx}', 'camera_center.npy'))
            camera_rotation = np.load(os.path.join(save_dir, f'cam_{idx}', 'camera_rotation.npy'))
            pano_mask = cv2.imread(os.path.join(save_dir, f'cam_{idx}', 'pano_mask.png'))
            # pano_mask = cv2.imread(os.path.join(save_dir, f'cam_{idx}', 'pano_mask.png'), cv2.IMREAD_GRAYSCALE) if i!=0 else np.zeros((res, res*2), dtype=np.uint8)
            # pano_mask = cv2.erode(pano_mask, np.ones((3,3), np.uint8), iterations=3)
            # pano_mask = cv2.cvtColor(pano_mask, cv2.COLOR_GRAY2RGB)
            cur_point_mask = np.load(os.path.join(save_dir, f'cam_{idx}', 'pano_mask.npy')).astype(np.bool_)
            all_point_mask = np.load(os.path.join(save_dir, f'cam_{idx}', 'pano_mask_all.npy')).astype(np.bool_)
            if room_scope_mask is not None:
                cur_point_mask = cur_point_mask & room_scope_mask
                all_point_mask = all_point_mask & room_scope_mask
            if i != 0:
                prev_all_point_mask = np.load(os.path.join(save_dir, f'cam_{cam_id[i-1]}', 'pano_mask_all.npy')).astype(np.bool_)
                if room_scope_mask is not None:
                    prev_all_point_mask = prev_all_point_mask & room_scope_mask
            else:
                prev_all_point_mask = np.zeros_like(all_point_mask, dtype=np.bool_)

            if not np.any(all_point_mask):
                print(f"[INFO] skip cam_{idx} in {room_tag}: no room points visible")
                continue
            # if not os.path.exists(os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_img.png')):
            #     r_image, r_depth = self.render_panorama(camera_center, save_dir=os.path.join(save_dir, f'cam_{idx}', 'styled'))
            r_image, r_depth = self.check_and_render_panorama(camera_center, camera_rotation, save_dir=os.path.join(save_dir, f'cam_{idx}', 'styled'))
            
            # mask = cv2.imread(os.path.join(save_dir, f'cam_{idx}', 'proj_mask_mid.png'), cv2.IMREAD_GRAYSCALE).astype(np.bool_)
            if os.path.exists(os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_styled_refined.png')):
                styled_img = cv2.imread(os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_styled_refined.png'))
            elif os.path.exists(os.path.join(save_dir, f'cam_{idx}', 'pano_styled_refined.png')):
                styled_img = cv2.imread(os.path.join(save_dir, f'cam_{idx}', 'pano_styled_refined.png'))
            else:
                missing_rate = 1-(pano_mask[...,0].astype(np.bool_).sum() / (pano_mask.shape[0]*pano_mask.shape[1]))
                print(f"Pixel Missing Rate: {missing_rate}")
                if i==0 or missing_rate>0.7:
                    print("Too many pixel missing, regenerating in all....")
                    adain_img = generate_adain(content_path=os.path.join(save_dir, f'cam_{idx}', 'pano_img.png'),
                                            style_path=self.style_img,
                                            save_path=os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_adain.png'))
                    styled_img = generate_image(prompt=self.prompt_text.value,
                                                style_img_path=self.style_img,
                                                input_img_path=os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_adain.png'),
                                                ref_img_path=os.path.join(save_dir, f'cam_{idx}', 'pano_img.png'),
                                                depth_img_path=os.path.join(save_dir, f'cam_{idx}', 'pano_img.png'),
                                                strength=1.0,
                                                )
                    styled_img.save(os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_styled_refined.png'))
                else:
                    ### Diffusers I2I LowStrength ###
                    # styled_img = generate_image(prompt=self.prompt_text.value,
                    #                             style_img_path=self.style_img,
                    #                             input_img_path=os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_img.png'),
                    #                             ref_img_path=os.path.join(save_dir, f'cam_{idx}', 'pano_img.png'),
                    #                             depth_img_path=os.path.join(save_dir, f'cam_{idx}', 'pano_img.png'),
                    #                             strength=0.8,
                    #                             )
                    # styled_img.save(os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_styled_refined.png'))

                    adain_img = generate_adain(content_path=os.path.join(save_dir, f'cam_{idx}', 'pano_img.png'),
                                            style_path=self.style_img,
                                            save_path=os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_adain.png'))
                    # rendered_img = Image.open(os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_img.png'))
                    
                    # rendered_img.paste(adain_img, (0,0), mask=ImageChops.invert(Image.open(os.path.join(save_dir, f'cam_{idx}', 'pano_mask.png'))))
                    # rendered_img.save(os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_pasted.png'))

                    # 将adain结果和原图进行融合，融合区域由pano_mask确定
                    adain_img_np = np.array(adain_img)
                    pano_img_np = cv2.imread(os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_img.png'))
                    pano_img_np = cv2.cvtColor(pano_img_np, cv2.COLOR_BGR2RGB)
                    pano_mask_np = cv2.imread(os.path.join(save_dir, f'cam_{idx}', 'pano_mask.png'), cv2.IMREAD_GRAYSCALE).astype(np.bool_)
                    fused_img_np = pano_img_np.copy()
                    fused_img_np[~pano_mask_np] = adain_img_np[~pano_mask_np]
                    fused_img = Image.fromarray(fused_img_np)
                    fused_img.save(os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_fused.png'))

                    ### Diffusers API ###
                    styled_img, inpainted_img = generate_image(prompt=self.prompt_text.value,
                                                style_img_path=self.style_img,
                                                input_img_path=os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_img.png'),
                                                mask_img_path=os.path.join(save_dir, f'cam_{idx}', 'pano_mask.png'),
                                                ref_img_path=os.path.join(save_dir, f'cam_{idx}', 'pano_img.png'),
                                                depth_img_path=os.path.join(save_dir, f'cam_{idx}', 'pano_img.png'),
                                                strength=0.3,
                                                )
                    styled_img.save(os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_styled_refined.png'))
                    inpainted_img.save(os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_styled_inpainted.png'))

                    ### SDWebUI API ###
                    # styled_img = inpaint(input_path=os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_img.png'),
                    #                     mask_path=os.path.join(save_dir, f'cam_{idx}', 'pano_mask.png'),
                    #                     canny_path=os.path.join(save_dir, f'cam_{idx}', 'pano_img.png'),
                    #                     depth_path=os.path.join(save_dir, f'cam_{idx}', 'pano_img.png'))[0]
                    # styled_img.save(os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_styled.png'))

                    ### Diffusers Refine ###
                    # refine_img = generate_image(prompt=self.prompt_text.value,
                    #                             style_img_path=self.style_img,
                    #                             input_img_path=os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_styled.png'),
                    #                             ref_img_path=os.path.join(save_dir, f'cam_{idx}', 'pano_img.png'),
                    #                             depth_img_path=os.path.join(save_dir, f'cam_{idx}', 'pano_img.png'),
                    #                             strength=0.3)
                    # refine_img.save(os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_styled_refined.png'))

                styled_img = cv2.imread(os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_styled_refined.png'))
            raw_img = cv2.imread(os.path.join(save_dir, f'cam_{idx}', 'pano_img.png'))
            # styled_img = cv2.imread(os.path.join(save_dir, f'cam_{idx}', 'pano_styled.png'))

            ### Pre-Color Gaussian ###
            print("Init Styled Gaussian...")
            print(f"Before save: {self.get_gpu_memory_usage()}")
            # cur_point_mask = point_mask.copy()
            # point_mask = np.logical_or(point_mask, prev_mask)
            cam_mask = (all_point_mask!=prev_all_point_mask) if i!=0 else all_point_mask
            if room_scope_mask is not None:
                cam_mask = cam_mask & room_scope_mask
            # cam_point_mask = np.load(os.path.join(save_dir, f'cam_{idx}', 'mask', f'cam_{idx}', 'point_mask.npy')).astype(np.bool_) if i!=0 else all_point_mask
            # if room_scope_mask is not None:
            #     cam_point_mask = cam_point_mask & room_scope_mask
            # if not np.any(cam_point_mask):
            #     print(f"[INFO] skip cam_{idx} projection update in {room_tag}: empty cam_point_mask")
            #     continue
            _feat_dc_visible, _feat_rest_visible = self.color_update_proj(
                # camera_center_path="/data/hyh/github/2D-GS-Viser-Viewer/preprocess/playroom_cartoon/cam_center/camera_center.npy",
                # styled_img_path="/data/hyh/github/2D-GS-Viser-Viewer/preprocess/playroom_cartoon/cam_center/pano_styled.png",
                # depth_path="/data/hyh/github/2D-GS-Viser-Viewer/preprocess/playroom_cartoon/cam_center/pano_depth.npy",

                camera_center_path=os.path.join(save_dir, f"cam_{idx}", "camera_center.npy"),
                camera_rotation=camera_rotation,
                styled_img_path=os.path.join(save_dir, f"cam_{idx}", "styled", "pano_styled_refined.png"),
                depth_path=os.path.join(save_dir, f"cam_{idx}", "pano_depth.npy"),
                point_mask=cam_mask,
                # point_mask=cam_point_mask,
                upscale=4,
                color_gaussian=True,
                room_scope_mask=room_scope_mask,
            )
            if feat_dc_visible is None:
                feat_dc_visible = _feat_dc_visible
                feat_rest_visible = _feat_rest_visible
            else:
                # feat_dc_visible[cur_point_mask] = (_feat_dc_visible[cur_point_mask] + feat_dc_visible[cur_point_mask])/2
                # feat_rest_visible[cur_point_mask] = (_feat_rest_visible[cur_point_mask] + feat_rest_visible[cur_point_mask])/2
                feat_dc_visible[cam_mask] = _feat_dc_visible[cam_mask]
                feat_rest_visible[cam_mask] = _feat_rest_visible[cam_mask]
                # feat_dc_visible[cam_point_mask] = _feat_dc_visible[cam_point_mask]
                # feat_rest_visible[cam_point_mask] = _feat_rest_visible[cam_point_mask]
                # feat_dc_visible[cur_point_mask] = _feat_dc_visible[cur_point_mask]
                # feat_rest_visible[cur_point_mask] = _feat_rest_visible[cur_point_mask]

            self.viewer_renderer.update_pc_features()

            self.hidden_color_propogation(point_mask=all_point_mask, room_scope_mask=room_scope_mask)
            self.viewer_renderer.update_pc_features()

            # equ_styled_img = E2P.Equirectangular(np.array(styled_img))
            # equ_raw_img = E2P.Equirectangular(raw_img)
            # equ_pano_mask = E2P.Equirectangular(255-pano_mask)

            styled_imgs.append({
                'cam_id': idx,
                'camera_center': camera_center,
                'camera_rotation': camera_rotation,
                'point_mask': cur_point_mask,
                'all_point_mask': all_point_mask,
                'cam_mask': cam_mask,
                # 'feat_dc_visible': torch.nn.Parameter(feat_dc_visible.contiguous()),
                # 'feat_rest_visible': torch.nn.Parameter(feat_rest_visible.contiguous()),
                # 'styled_img': equ_styled_img,
                # 'raw_img': equ_raw_img,
                # 'pano_mask': equ_pano_mask,
                # 'styled_img_tensor': get_pano_imgs_tensor(equ_styled_img),
                # 'raw_img_tensor': get_pano_imgs_tensor(equ_raw_img),
                # 'pano_mask_tensor': get_pano_imgs_tensor(equ_pano_mask),
                'styled_img_tensor': get_pano_imgs_tensor(E2P.Equirectangular(np.array(styled_img))),
                # 'raw_img_tensor': get_pano_imgs_tensor(E2P.Equirectangular(raw_img)),
                'pano_mask_tensor': get_pano_imgs_tensor(E2P.Equirectangular(255-pano_mask)),
            })

            # # Load precomputed distance-based pano masks for current stage.
            # for sidx in range(len(styled_imgs)):
            #     every_cam_id = styled_imgs[sidx]['cam_id']
            #     stage_mask_path = os.path.join(save_dir, f"cam_{idx}", "mask", f"cam_{every_cam_id}", "pano_img.png")
            #     if os.path.exists(stage_mask_path):
            #         stage_mask = cv2.imread(stage_mask_path)
            #         styled_imgs[sidx]['pano_mask_tensor'] = get_pano_imgs_tensor(E2P.Equirectangular(stage_mask))

            print(f"After save: {self.get_gpu_memory_usage()}")
            torch.cuda.empty_cache()

            if not os.path.exists(os.path.join(save_dir, f'cam_{idx}', 'styled', 'scene_styled.ply')):
                print(f"Training scene in cam_{idx}...")
                self.gaussian_model.training_setup(opt)
                train_pano(styled_imgs, feat_dc_visible, feat_rest_visible, 200+25*i, random_flag=True)
                self.gaussian_model.save_ply(os.path.join(save_dir, f'cam_{idx}', 'styled', 'scene_styled.ply'))
            else:
                print(f"Loading {os.path.join(save_dir, f'cam_{idx}', 'styled', 'scene_styled.ply')}")
                self.gaussian_model.load_ply(os.path.join(save_dir, f'cam_{idx}', 'styled', 'scene_styled.ply'))

            r_image, r_depth = self.check_and_render_panorama(camera_center, camera_rotation, save_dir=os.path.join(save_dir, f'cam_{idx}', 'after_styled'))

            steps += self.train_steps.value
            self.update_client()
        
        # print("Training on all styled images")
        # self.gaussian_model.training_setup(opt)
        # train_pano(styled_imgs, feat_dc_visible, feat_rest_visible, 200+25*i, random_flag=True)
        # self.gaussian_model.save_ply(os.path.join(save_dir, 'scene_styled_final.ply'))

    def _setup_general_features_folder(self, server, tabs):
        with tabs.add_tab("General"):
            if self.is_training:
                with server.add_gui_folder("Training Infos"):
                    self.iter = server.add_gui_text('Iteration', initial_value = '0')
                    self.loss = server.add_gui_text('Loss', initial_value = '0.0')
                    self.dist = server.add_gui_text('distortion', initial_value = '0.0')
                    self.norm = server.add_gui_text('normal', initial_value = '0.0')
                    self.gpu_mem = server.add_gui_text(
                        'Memory Usage',
                        initial_value = self.get_gpu_memory_usage()
                    )
                    self.fps = server.add_gui_text(
                        'fps',
                        initial_value = ' frame/sec'
                    )
            else:
                with server.add_gui_folder("Status"):
                    self.gpu_mem = server.add_gui_text(
                        'Memory Usage',
                        initial_value = self.get_gpu_memory_usage()
                    )
                    self.fps = server.add_gui_text(
                        'fps',
                        initial_value = ' frame/sec'
                    )
            with server.add_gui_folder("Image Options"):
                self.max_res_when_static = server.add_gui_slider(
                    "Max Res",
                    min=128,
                    max=3840,
                    step=128,
                    initial_value=1920,
                )
                self.max_res_when_static.on_update(self._handle_option_updated)
                self.max_res_when_moving = server.add_gui_slider(
                    "(when Move)",
                    min=128,
                    max=1920,
                    step=128,
                    initial_value=1024,
                )
                self.jpeg_quality_when_static = server.add_gui_slider(
                    "JPEG Quality",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=100,
                )
                self.jpeg_quality_when_static.on_update(self._handle_option_updated)

                self.jpeg_quality_when_moving = server.add_gui_slider(
                    "(when Move)",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=60,
                )
            
                self.training_view_slider = server.add_gui_slider(
                    "Training View",
                    min=0,
                    max=len(self.camera_poses)-1,
                    step=1,
                    initial_value=0,
                )

                self.save_world_view_button = server.add_gui_button("Render World View")
                self.save_view_button = server.add_gui_button("Render Current View")
                self.render_current_panorama_button = server.add_gui_button("Render Current Panorama")
                self.save_all_training_view_button = server.add_gui_button("Save All Training View")
                self.save_all_training_view_panorama_button = server.add_gui_button("Save All Training View Panorama")

                self.camera_path_text = server.add_gui_text("Camera Path", initial_value="renders/camera_paths/playroom_fast.json", hint="Name of the render")
                self.save_video_button = server.add_gui_button("Save Video")

            with server.add_gui_folder("Panorama Options"):
                self.theta_view_slider = server.add_gui_slider(
                    "View Theta",
                    min=-180,
                    max=180,
                    step=1,
                    initial_value=0,
                )
                self.phi_view_slider = server.add_gui_slider(
                    "View Phi",
                    min=-180,
                    max=180,
                    step=1,
                    initial_value=0,
                )
                self.gamma_view_slider = server.add_gui_slider(
                    "View Gamma",
                    min=-180,
                    max=180,
                    step=1,
                    initial_value=0,
                )

                self.threshold_slider = server.add_gui_slider(
                    "Threshold",
                    min=0,
                    max=1,
                    step=0.1,
                    initial_value=0.5,
                )

                self.camera_gap_slider = server.add_gui_slider(
                    "Camera Gap",
                    min=0,
                    max=20,
                    step=1,
                    initial_value=3,
                )

                self.min_camera_object_dist_slider = server.add_gui_slider(
                    "Min Camera-Object Dist",
                    min=0.0,
                    max=5.0,
                    step=0.1,
                    initial_value=0.3,
                )

                self.knn_number = server.add_gui_slider(
                    "K Neareast Neighbors Number",
                    min=0,
                    max=10,
                    step=1,
                    initial_value=3,
                )

                self.enable_camera_center = server.add_gui_checkbox(
                    "Add Camera Center when Preprocess",
                    initial_value=True,
                )
                self.camera_focus = server.add_gui_checkbox(
                    "Use camera focus",
                    initial_value=True,
                )
                self.sort_camera_center = server.add_gui_checkbox(
                    "Sort by Camera Positions",
                    initial_value=True,
                )
                self.start_from_center = server.add_gui_checkbox(
                    "Start from Center",
                    initial_value=False,
                )
                self.render_test_panorama = server.add_gui_checkbox(
                    "Render Test Panorama",
                    initial_value=False,
                ) 

                self.preprocess_test_button = server.add_gui_button("Preprocess Test")
                self.highlight_preprocess_test_button = server.add_gui_button("Highlight Preprocess Cameras")

                self.preprocess_stylization_button = server.add_gui_button("Preprocess Stylization")
                self.clear_prep_cache_button = server.add_gui_button("Clear Preprocess Cache")
                self.prompt_text = server.add_gui_text("Prompt", initial_value="", hint="prompt of sytle")
                self.start_stylization_button = server.add_gui_button("Start Stylization")
                self.render_panorama_button = server.add_gui_button("Render Panorama")
                self.start_refine_button = server.add_gui_button("Start Refine")
                self.start_pers_button = server.add_gui_button("Start Perspective Stylization")

            with server.add_gui_folder("Manual Room Split"):
                source_options = ("json", "colmap") if len(self.json_camera_poses) > 0 else ("colmap",)
                self.camera_pose_source_selector = server.add_gui_button_group(
                    "Camera Pose Source",
                    source_options,
                )
                self.manual_split_enable = server.add_gui_checkbox(
                    "Enable Manual Camera Split",
                    initial_value=False,
                )
                self.manual_split_select_mode = server.add_gui_button_group(
                    "Select Mode",
                    ("replace", "append"),
                )
                self.manual_split_drag_select_enabled = server.add_gui_checkbox(
                    "Enable Drag Rect Select",
                    initial_value=False,
                )
                self.manual_split_drag_select_target = server.add_gui_button_group(
                    "Drag Select Target",
                    ("camera", "point"),
                )
                self.manual_split_preview_mode = server.add_gui_button_group(
                    "Split Preview Render",
                    ("gaussian", "pointcloud"),
                )
                self.manual_split_room_id = server.add_gui_slider(
                    "Current Room ID",
                    min=0,
                    max=255,
                    step=1,
                    initial_value=1,
                )
                self.manual_split_assign_button = server.add_gui_button("Assign Selection To Room")
                self.manual_split_remove_button = server.add_gui_button("Remove Selection From Room")
                self.manual_split_select_room_button = server.add_gui_button("Select Cameras In Current Room")
                self.manual_split_clear_room_button = server.add_gui_button("Clear Current Room")
                self.manual_split_clear_selection_button = server.add_gui_button("Clear Selection")
                self.manual_split_undo_button = server.add_gui_button("Undo Last Assignment")
                self.manual_split_apply_button = server.add_gui_button("Apply Split And Save")
                self.manual_split_colorize_points_button = server.add_gui_button("Colorize Assigned Points")
                self.manual_split_restore_points_color_button = server.add_gui_button("Restore Point Colors")
                self.manual_split_point_select_mode = server.add_gui_button_group(
                    "Point Select Mode",
                    ("replace", "append", "subtract"),
                )
                self.manual_split_select_points_button = server.add_gui_button("Select Points From Selected Cameras")
                self.manual_split_assign_points_button = server.add_gui_button("Assign Selected Points To Current Room")
                self.manual_split_unassign_points_button = server.add_gui_button("Unassign Selected Points")
                self.manual_split_clear_point_selection_button = server.add_gui_button("Clear Point Selection")
                self.manual_split_save_point_assignments_button = server.add_gui_button("Save Point Assignments")
                self.manual_split_preview_selected_points = server.add_gui_checkbox(
                    "Preview Selected Points",
                    initial_value=True,
                )
                self.manual_split_preview_selected_points_color = server.add_gui_rgb(
                    "Selected Point Color",
                    initial_value=self.manual_split_selected_point_preview_color,
                )
                self.manual_split_status = server.add_gui_text(
                    "Split Status",
                    initial_value="Manual split disabled.",
                )
                self.room_assign_k_nearest_slider = server.add_gui_slider(
                    "Score K Nearest",
                    min=1,
                    max=8,
                    step=1,
                    initial_value=self.room_assign_k_nearest,
                )
                self.room_assign_support_expected_slider = server.add_gui_slider(
                    "Score Support Expected",
                    min=1.0,
                    max=6.0,
                    step=0.1,
                    initial_value=self.room_assign_support_expected,
                )
                self.room_assign_support_penalty_slider = server.add_gui_slider(
                    "Score Support Penalty",
                    min=0.0,
                    max=3.0,
                    step=0.05,
                    initial_value=self.room_assign_support_penalty_lambda,
                )
                self.room_assign_dist_weight_slider = server.add_gui_slider(
                    "Weight Distance",
                    min=0.0,
                    max=3.0,
                    step=0.05,
                    initial_value=self.room_assign_dist_weight,
                )
                self.room_assign_center_weight_slider = server.add_gui_slider(
                    "Weight Room Center",
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    initial_value=self.room_assign_center_weight,
                )
                self.room_assign_color_weight_slider = server.add_gui_slider(
                    "Weight Color",
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    initial_value=self.room_assign_color_weight,
                )
                self.room_assign_support_reward_slider = server.add_gui_slider(
                    "Weight Support Reward",
                    min=0.0,
                    max=1.0,
                    step=0.02,
                    initial_value=self.room_assign_support_reward_weight,
                )
                self.room_assign_confidence_margin_slider = server.add_gui_slider(
                    "Confidence Margin Fallback",
                    min=0.0,
                    max=0.5,
                    step=0.01,
                    initial_value=self.room_assign_conf_margin_threshold,
                )
                self.room_assign_graph_refine_enable_checkbox = server.add_gui_checkbox(
                    "Enable Graph Refine",
                    initial_value=self.room_assign_graph_refine_enable,
                )
                self.room_assign_graph_knn_slider = server.add_gui_slider(
                    "Graph KNN",
                    min=2,
                    max=32,
                    step=1,
                    initial_value=self.room_assign_graph_knn,
                )
                self.room_assign_graph_lambda_slider = server.add_gui_slider(
                    "Graph Smooth Lambda",
                    min=0.0,
                    max=2.0,
                    step=0.02,
                    initial_value=self.room_assign_graph_lambda,
                )
                self.room_assign_graph_sigma_x_slider = server.add_gui_slider(
                    "Graph Sigma X (0=auto)",
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    initial_value=self.room_assign_graph_sigma_x,
                )
                self.room_assign_graph_sigma_c_slider = server.add_gui_slider(
                    "Graph Sigma Color",
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    initial_value=self.room_assign_graph_sigma_c,
                )
                self.room_assign_graph_iters_slider = server.add_gui_slider(
                    "Graph Iterations",
                    min=1,
                    max=20,
                    step=1,
                    initial_value=self.room_assign_graph_iters,
                )
                self.room_assign_graph_only_low_conf_checkbox = server.add_gui_checkbox(
                    "Graph Refine Only Low-Conf",
                    initial_value=self.room_assign_graph_only_low_conf,
                )
                self.room_assign_graph_low_conf_margin_slider = server.add_gui_slider(
                    "Graph Low-Conf Margin",
                    min=0.01,
                    max=0.5,
                    step=0.01,
                    initial_value=self.room_assign_graph_low_conf_margin,
                )
                self.room_assign_graph_use_normal_consistency_checkbox = server.add_gui_checkbox(
                    "Graph Use Normal Consistency",
                    initial_value=self.room_assign_graph_use_normal_consistency,
                )
                self.room_assign_graph_normal_weight_slider = server.add_gui_slider(
                    "Graph Normal Weight",
                    min=0.0,
                    max=3.0,
                    step=0.05,
                    initial_value=self.room_assign_graph_normal_weight,
                )

            with server.add_gui_folder("Render Options"):
                self.render_type = server.add_gui_dropdown(
                    "Render Type", tuple(self.render_type_name.keys())[:-1]
                )
                self.depth_ratio_slider = server.add_gui_slider(
                    "Depth Ratio (mean ~ med)",
                    min=0.,
                    max=1.,
                    step=0.1,
                    initial_value=0.,
                )
            
                with server.add_gui_folder("screen split"):
                    self.enable_split = server.add_gui_checkbox(
                        "use Split",
                        initial_value=False,
                    )
                    self.mode_slider = server.add_gui_slider(
                        "Split Slider",
                        min=0.,
                        max=0.99,
                        step=0.01,
                        initial_value=0.5,
                    )

                    self.render_type1 = server.add_gui_dropdown(
                        "Left Type", tuple(self.render_type_name.keys())[:-1]
                    )
                    self.render_type2 = server.add_gui_dropdown(
                        "Right Type", tuple(self.render_type_name.keys())[:-1]
                    )
            
            with server.add_gui_folder("Gaussian Model"):
                self.enable_ptc = server.add_gui_checkbox(
                    "as Pointcloud",
                    initial_value=False,
                )
                self.surfel_mode = server.add_gui_button_group("View Type", ("ptc", "disk"))
                self.point_size = server.add_gui_slider(
                    "Point Size",
                    min=0.001,
                    max=0.1,
                    initial_value=0.01,
                    step=0.001,
                )
                self.scale_slider = server.add_gui_slider(
                    "Scaler",
                    min=0.1,
                    max=3.,
                    step=0.1,
                    initial_value=1.,
                )
                self.sparsity_slider = server.add_gui_slider(
                    "Sparsity",
                    min=1,
                    max=10,
                    step=1,
                    initial_value=1,
                )
                
                if self.viewer_renderer.gaussian_model.max_sh_degree > 0:
                    self.active_sh_degree_slider = server.add_gui_slider(
                        "SH Degree",
                        min=0,
                        max=self.viewer_renderer.gaussian_model.max_sh_degree,
                        step=1,
                        initial_value=3,
                    )
                    @self.active_sh_degree_slider.on_update
                    def _(event): 
                        with server.atomic(): self._handle_option_updated(_)

            with server.add_gui_folder("Crop Box"):
                self.enable_crop = server.add_gui_checkbox(
                    "use Crop",
                    initial_value=False,
                )
                self.box_x = server.add_gui_multi_slider(
                    'x range', 
                    min = -self.crop_box_size,
                    max = self.crop_box_size,
                    step = 0.1,
                    initial_value= [-4.0, 4.0]
                )
                self.box_y = server.add_gui_multi_slider(
                    'y range', 
                    min = -self.crop_box_size,
                    max = self.crop_box_size,
                    step = 0.1,
                    initial_value=[-4.0, 4.0]
                )
                self.box_z = server.add_gui_multi_slider(
                    'z range', 
                    min = -self.crop_box_size,
                    max = self.crop_box_size,
                    step = 0.1,
                    initial_value=[-4.0, 4.0]
                )
            
            with server.add_gui_folder("Training Option"):
                self.train_steps = server.add_gui_slider(
                    "Total Step", min=0, max=5000, step=100, initial_value=100
                )
                self.color_lr_scaler = server.add_gui_slider(
                    "Color LR", min=0.0, max=10.0, step=0.1, initial_value=3.0
                )

            # add cameras
            self.add_cameras_to_scene(server)
            self._load_manual_split_data()
            self._refresh_manual_split_camera_colors()
            self.manual_split_status.value = self._manual_split_status_text()
            if hasattr(self, "camera_pose_source_selector"):
                self.camera_pose_source_selector.value = self.camera_pose_source_active

            @self.render_type.on_update
            @self.render_type1.on_update
            @self.render_type2.on_update
            @self.enable_split.on_update
            @self.mode_slider.on_update
            @self.depth_ratio_slider.on_update
            @self.scale_slider.on_update
            @self.sparsity_slider.on_update
            @self.enable_ptc.on_update
            @self.surfel_mode.on_click
            @self.point_size.on_update
            @self.enable_crop.on_update
            @self.box_x.on_update 
            @self.box_y.on_update 
            @self.box_z.on_update
            def _(event): 
                with server.atomic(): self._handle_option_updated(_)

            @self.manual_split_enable.on_update
            def _(_event):
                self._update_rect_select_bindings(server)
                self._refresh_manual_split_camera_colors()
                self.manual_split_status.value = self._manual_split_status_text()

            @self.camera_pose_source_selector.on_click
            def _(_event):
                requested_source = self.camera_pose_source_selector.value
                active_source = self._set_active_camera_pose_source(requested_source)
                if active_source != requested_source:
                    self.camera_pose_source_selector.value = active_source
                    print(f"[WARN] pose source '{requested_source}' unavailable, fallback to '{active_source}'")

                for handle in self.camera_handles:
                    try:
                        handle.remove()
                    except Exception:
                        pass
                self.camera_handles = []
                self.camera_handles_by_idx = {}
                self.camera_handle_names = {}
                self.add_cameras_to_scene(server)

                # Keep split assignments valid after source switch.
                self.manual_split_selected_ids = set()
                self.manual_split_assignments = {
                    cam_idx: room_id
                    for cam_idx, room_id in self.manual_split_assignments.items()
                    if 0 <= cam_idx < len(self.camera_poses)
                }

                if hasattr(self, "training_view_slider"):
                    self.training_view_slider.max = max(len(self.camera_poses) - 1, 0)
                    if self.training_view_slider.value > self.training_view_slider.max:
                        self.training_view_slider.value = self.training_view_slider.max

                self._refresh_manual_split_camera_colors()
                self.manual_split_status.value = self._manual_split_status_text()
                print(f"[INFO] camera pose source switched to: {active_source}")

            @self.manual_split_drag_select_enabled.on_update
            def _(_event):
                self._update_rect_select_bindings(server)
                self.manual_split_status.value = self._manual_split_status_text()

            @self.manual_split_preview_selected_points.on_update
            def _(_event):
                self._refresh_manual_split_point_selection_preview()
                self.manual_split_status.value = self._manual_split_status_text()

            @self.manual_split_preview_selected_points_color.on_update
            def _(_event):
                self.manual_split_selected_point_preview_color = tuple(self.manual_split_preview_selected_points_color.value)
                self._refresh_manual_split_point_selection_preview()
                self.manual_split_status.value = self._manual_split_status_text()

            @self.manual_split_preview_mode.on_click
            def _(_event):
                if self.manual_split_preview_mode.value == "pointcloud":
                    self.enable_ptc.value = True
                    self.surfel_mode.value = "ptc"
                else:
                    self.enable_ptc.value = False
                self._handle_option_updated(_)

            @self.manual_split_clear_selection_button.on_click
            def _(_event):
                self.manual_split_selected_ids.clear()
                self._refresh_manual_split_camera_colors()
                self.manual_split_status.value = self._manual_split_status_text()

            @self.manual_split_clear_point_selection_button.on_click
            def _(_event):
                self.manual_split_selected_point_mask = None
                self._refresh_manual_split_point_selection_preview()
                self.manual_split_status.value = self._manual_split_status_text()

            @self.manual_split_assign_button.on_click
            def _(_event):
                if len(self.manual_split_selected_ids) == 0:
                    print("[INFO] no selected cameras to assign")
                    return
                self.manual_split_last_assignments = dict(self.manual_split_assignments)
                room_id = int(self.manual_split_room_id.value)
                for cam_idx in self.manual_split_selected_ids:
                    self.manual_split_assignments[cam_idx] = room_id
                self._refresh_manual_split_camera_colors()
                assigned_count = len(self.manual_split_assignments)
                total_count = len(self.camera_poses)
                base_status = self._manual_split_status_text()
                self.manual_split_status.value = f"{base_status} | Assigned Cameras: {assigned_count}/{total_count}"
                print(f"[INFO] assigned cameras: {assigned_count}/{total_count}")

            @self.manual_split_remove_button.on_click
            def _(_event):
                if len(self.manual_split_selected_ids) == 0:
                    print("[INFO] no selected cameras to remove")
                    return
                self.manual_split_last_assignments = dict(self.manual_split_assignments)
                for cam_idx in list(self.manual_split_selected_ids):
                    self.manual_split_assignments.pop(cam_idx, None)
                self._refresh_manual_split_camera_colors()
                self.manual_split_status.value = self._manual_split_status_text()

            @self.manual_split_select_room_button.on_click
            def _(_event):
                room_id = int(self.manual_split_room_id.value)
                self.manual_split_selected_ids = {
                    cam_idx for cam_idx, rid in self.manual_split_assignments.items() if rid == room_id
                }
                self._refresh_manual_split_camera_colors()
                self.manual_split_status.value = self._manual_split_status_text()

            @self.manual_split_clear_room_button.on_click
            def _(_event):
                room_id = int(self.manual_split_room_id.value)
                self.manual_split_last_assignments = dict(self.manual_split_assignments)
                remove_ids = [cam_idx for cam_idx, rid in self.manual_split_assignments.items() if rid == room_id]
                for cam_idx in remove_ids:
                    self.manual_split_assignments.pop(cam_idx, None)
                self._refresh_manual_split_camera_colors()
                self.manual_split_status.value = self._manual_split_status_text()

            @self.manual_split_undo_button.on_click
            def _(_event):
                if self.manual_split_last_assignments is None:
                    print("[INFO] nothing to undo")
                    return
                self.manual_split_assignments = self.manual_split_last_assignments
                self.manual_split_last_assignments = None
                self._refresh_manual_split_camera_colors()
                self.manual_split_status.value = self._manual_split_status_text()

            @self.manual_split_apply_button.on_click
            def _(_event):
                if self.manual_split_is_applying:
                    self.manual_split_status.value = "Split is already running, please wait..."
                    print("[INFO] split apply is already running")
                    return

                self.manual_split_is_applying = True
                start_t = time.time()
                self.manual_split_status.value = "Applying split and saving..."

                def _run_apply_split_job():
                    try:
                        self._save_manual_split_data()
                        self._colorize_points_by_room()
                        elapsed = time.time() - start_t
                        status = self._manual_split_status_text()
                        self.manual_split_status.value = f"{status} | Split saved in {elapsed:.2f}s"
                    except Exception as err:
                        self.manual_split_status.value = f"Split apply failed: {err}"
                        print(f"[WARN] split apply failed: {err}")
                    finally:
                        self.manual_split_is_applying = False

                threading.Thread(target=_run_apply_split_job, daemon=True).start()

            @self.manual_split_colorize_points_button.on_click
            def _(_event):
                self._colorize_points_by_room()

            @self.manual_split_restore_points_color_button.on_click
            def _(_event):
                self._restore_point_colors_after_room_visualization()

            @self.manual_split_select_points_button.on_click
            def _(_event):
                try:
                    self.manual_split_status.value = "Selecting points from selected cameras..."
                    hit_count = self._manual_split_select_points_from_selected_cameras(
                        mode=self.manual_split_point_select_mode.value,
                    )
                    self.manual_split_status.value = self._manual_split_status_text()
                    print(f"[INFO] selected points by cameras: {hit_count}")
                except Exception as err:
                    self.manual_split_status.value = f"Point selection failed: {err}"
                    print(f"[WARN] point selection failed: {err}")

            @self.manual_split_assign_points_button.on_click
            def _(_event):
                try:
                    room_id = int(self.manual_split_room_id.value)
                    changed = self._manual_split_assign_selected_points_to_room(room_id)
                    self._clear_manual_split_point_selection_preview()
                    self._colorize_points_by_room()
                    self._refresh_manual_split_point_selection_preview()
                    self.manual_split_status.value = self._manual_split_status_text()
                    print(f"[INFO] assigned selected points to room_{room_id}: {changed}")
                except Exception as err:
                    self.manual_split_status.value = f"Point assign failed: {err}"
                    print(f"[WARN] point assign failed: {err}")

            @self.manual_split_unassign_points_button.on_click
            def _(_event):
                try:
                    changed = self._manual_split_unassign_selected_points()
                    self._clear_manual_split_point_selection_preview()
                    self._colorize_points_by_room()
                    self._refresh_manual_split_point_selection_preview()
                    self.manual_split_status.value = self._manual_split_status_text()
                    print(f"[INFO] unassigned selected points: {changed}")
                except Exception as err:
                    self.manual_split_status.value = f"Point unassign failed: {err}"
                    print(f"[WARN] point unassign failed: {err}")

            @self.manual_split_save_point_assignments_button.on_click
            def _(_event):
                try:
                    self._save_manual_split_point_assignments()
                    self.manual_split_status.value = self._manual_split_status_text() + " | Point assignments saved"
                except Exception as err:
                    self.manual_split_status.value = f"Save point assignments failed: {err}"
                    print(f"[WARN] save point assignments failed: {err}")

            @server.on_client_connect
            def _(client: viser.ClientHandle) -> None:
                self._update_rect_select_bindings(server)

            go_to_scene_center = server.add_gui_button("Go to scene center",)
            @go_to_scene_center.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                event.client.camera.position = self.camera_center + np.asarray([2.5, 0., 0.])
                event.client.camera.look_at = self.camera_center
            
            @self.training_view_slider.on_update
            def _(_):
                for client in server.get_clients().values():
                    target_camera = self.camera_poses[self.training_view_slider.value]
                    with client.atomic():
                        client.camera.wxyz = vtf.SO3.from_matrix(np.array(target_camera['rotation'])).wxyz
                        client.camera.position = np.array(target_camera['position'])
                        client.camera.up_direction = vtf.SO3(client.camera.wxyz) @ np.array(
                            [0.0, -1.0, 0.0]
                        )
            
            @self.save_world_view_button.on_click
            def _(_):
                for client in server.get_clients().values():
                    rotation = torch.from_numpy(vtf.SO3.from_rpy_radians(math.radians(-50), math.radians(0), math.radians(0)).as_matrix())
                    target_colmap_camera = self.colmap_cameras[0]
                    results = self.render(rotation, np.array([0,0,0]),
                                          target_colmap_camera.FovX, target_colmap_camera.FovY,
                                          target_colmap_camera.width, target_colmap_camera.height)
                    Image.fromarray((results['render'].clip(0,1).permute(1, 2, 0)*255).detach().cpu().numpy().astype(np.uint8)).save("capture.png")
                    print("save world view success.")
            
            @self.save_view_button.on_click
            def _(_):
                for client in server.get_clients().values():
                    # target_camera = self.camera_poses[self.training_view_slider.value]
                    target_colmap_camera = self.colmap_cameras[self.training_view_slider.value]
                    results = self.render(target_colmap_camera.R, target_colmap_camera.T,
                                          target_colmap_camera.FovX, target_colmap_camera.FovY,
                                          target_colmap_camera.width, target_colmap_camera.height)
                    Image.fromarray((results['render'].clip(0,1).permute(1, 2, 0)*255).detach().cpu().numpy().astype(np.uint8)).save("capture.png")
                    print("save current view success.")
            
            @self.save_all_training_view_button.on_click
            def _(_):
                os.makedirs("output_trainingview/temp", exist_ok=True)
                for idx in tqdm(range(len(self.camera_poses))):
                    # target_camera = self.camera_poses[idx]
                    target_colmap_camera = self.colmap_cameras[idx]
                    results = self.render(target_colmap_camera.R, target_colmap_camera.T,
                                          target_colmap_camera.FovX, target_colmap_camera.FovY,
                                          target_colmap_camera.width, target_colmap_camera.height)
                    Image.fromarray((results['render'].clip(0,1).permute(1, 2, 0)*255).detach().cpu().numpy().astype(np.uint8)).save(f"output_trainingview/temp/view_capture{idx:03d}.png")
        
            
            @self.render_current_panorama_button.on_click
            def _(_):
                for client in server.get_clients().values():
                    target_camera = self.camera_poses[self.training_view_slider.value]
                    image, depth = self.render_panorama(np.array(target_camera['position']), camera_rotation=target_camera['rotation'], save_dir=os.path.join("test_render", f"cam_{self.training_view_slider.value}"))#, render_perspetive=True)

            @self.save_all_training_view_panorama_button.on_click
            def _(_):
                os.makedirs("test_render_pano", exist_ok=True)
                for idx, cam in enumerate(self.camera_poses):
                    target_camera = cam
                    image, depth = self.render_panorama(np.array(target_camera['position']), res=256)
                    Image.fromarray(image.astype(np.uint8)).save(os.path.join("test_render_pano", f'pano_img_cam{idx}.png'))
            
            @self.save_video_button.on_click
            def _(_):
                os.makedirs("test_render_video/temp", exist_ok=True)
                print("Saving Video...")
                with open(self.camera_path_text.value, 'r', encoding='utf-8') as f:
                    camera_data = json.load(f)
                # video = cv2.VideoWriter(f"test_render_video/temp.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (self.colmap_cameras[0].width, self.colmap_cameras[0].height))
                video = cv2.VideoWriter(f"test_render_video/temp.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1024, 1024))
                for idx, cam in tqdm(enumerate(camera_data['camera_path'])):
                    camera_to_world = np.array(cam['camera_to_world']).reshape(4,4)
                    # R = Rt[:3,:3]
                    # T = Rt[:3, 3]
                    # Rt = np.linalg.inv(camera_to_world)
                    Rt = camera_to_world
                    R = (Rt[:3,:3] @ np.linalg.inv(vtf.SO3.from_x_radians(np.pi).as_matrix()))
                    # R = Rt[:3,:3]
                    T = -np.linalg.inv(R) @ Rt[:3, 3]
                    results = self.render(R, T,
                                        #   self.colmap_cameras[0].FovX, self.colmap_cameras[0].FovY,
                                          math.radians(60), math.radians(60),
                                        #   self.colmap_cameras[0].width, self.colmap_cameras[0].height
                                        1024, 1024
                                          )
                    result_img = (results['render'].clip(0,1).permute(1, 2, 0)*255).detach().cpu().numpy().astype(np.uint8)
                    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    # Image.fromarray((results['render'].clip(0,1).permute(1, 2, 0)*255).detach().cpu().numpy().astype(np.uint8)).save(f"test_render_video/temp/capture{idx:03d}.png")
                    video.write(result_img)
                video.release()
                print("Save Video Success.")

            def update_camera():
                for client in server.get_clients().values():
                    theta = self.theta_view_slider.value
                    phi = self.phi_view_slider.value
                    gamma = self.gamma_view_slider.value

                    R = vtf.SO3.from_rpy_radians(math.radians(phi), math.radians(theta), math.radians(gamma))
                    T_world_camera = vtf.SE3.from_rotation_and_translation(
                        R, np.array([0,0,0])
                    ).inverse()
                    wxyz = T_world_camera.rotation().wxyz
                    position = T_world_camera.translation()

                    with client.atomic():
                        client.camera.wxyz = wxyz
                        client.camera.position = position
                        self.fov = [90, 90]
                        client.camera.up_direction = vtf.SO3(client.camera.wxyz) @ np.array(
                            [0.0, -1.0, 0.0]
                        )

            @self.theta_view_slider.on_update
            def _(_):
                self.need_update = True
                update_camera()
            
            @self.phi_view_slider.on_update
            def _(_):
                update_camera()
            
            @self.gamma_view_slider.on_update
            def _(_):
                update_camera()
            
            @self.preprocess_test_button.on_click
            def _(_):
                # all_point_mask = np.load("/raid0/hyh22/github/PanoGS/preprocess/playroom_test/cam_57/pano_mask_after.npy").astype(np.bool_)

                # with torch.no_grad():
                #     self.gaussian_model._features_dc[all_point_mask] = RGB2SH((torch.ones_like(self.gaussian_model._features_dc[all_point_mask]).float()).cuda())
                #     self.gaussian_model._features_rest[all_point_mask] = RGB2SH((torch.ones_like(self.gaussian_model._features_dc[all_point_mask]).float()).cuda())
                #     self.gaussian_model._features_dc[~all_point_mask] = RGB2SH((torch.zeros_like(self.gaussian_model._features_dc[~all_point_mask]).float()).cuda())
                #     self.gaussian_model._features_rest[~all_point_mask] = RGB2SH((torch.zeros_like(self.gaussian_model._features_dc[~all_point_mask]).float()).cuda())
                
                # self.update_client()

                # return
                print(len(self.camera_poses))
                room_groups, camera_id_source = self._resolve_preprocess_room_groups(prefer_manual_split=True)
                self.preprocess_test_room_groups = {
                    int(room_id): [int(cam_idx) for cam_idx in cam_id]
                    for room_id, cam_id in room_groups.items()
                }
                selected_cam_ids: Set[int] = set()
                for cam_id in self.preprocess_test_room_groups.values():
                    for cam_idx in cam_id:
                        if 0 <= cam_idx < len(self.camera_poses):
                            selected_cam_ids.add(cam_idx)
                self.preprocess_test_selected_cam_ids = sorted(selected_cam_ids)

                print(f"[INFO] preprocess_test camera id source: {camera_id_source}")
                for room_id, cam_id in sorted(room_groups.items(), key=lambda x: x[0]):
                    room_label = "all" if room_id == -1 else f"room_{room_id}"
                    print(f"[INFO] preprocess_test {room_label} | Candidate Length: {len(cam_id)} | List: {cam_id}")

                if self.render_test_panorama.value:
                    os.makedirs("test_render_pano", exist_ok=True)
                    for room_id, cam_id in sorted(room_groups.items(), key=lambda x: x[0]):
                        room_label = "all" if room_id == -1 else f"room_{room_id}"
                        room_render_dir = os.path.join("test_render_pano", room_label)
                        os.makedirs(room_render_dir, exist_ok=True)
                        for id in cam_id:
                            if id == -1:
                                camera_center = np.array([0, 0, 0])
                                camera_rotation = np.diag(np.ones((3)))
                            else:
                                target_camera = self.camera_poses[id]
                                camera_center = np.array(target_camera['position'])
                                camera_rotation = np.array(target_camera['rotation'])
                            image, depth = self.render_panorama(camera_center, camera_rotation=camera_rotation, res=256)
                            save_file_path = os.path.join(room_render_dir, f'pano_img_cam{id}.png')
                            Image.fromarray(image.astype(np.uint8)).save(save_file_path)
                            print(f"Save in {save_file_path}")

            @self.highlight_preprocess_test_button.on_click
            def _(_):
                if len(self.preprocess_test_selected_cam_ids) == 0:
                    print("[INFO] preprocess_test results empty, run 'Preprocess Test' first")
                    return
                self.manual_split_selected_ids = set(self.preprocess_test_selected_cam_ids)
                self._refresh_manual_split_camera_colors()
                if hasattr(self, "manual_split_status"):
                    self.manual_split_status.value = self._manual_split_status_text()
                print(
                    f"[INFO] highlighted preprocess_test cameras: {len(self.preprocess_test_selected_cam_ids)} | "
                    f"List: {self.preprocess_test_selected_cam_ids}"
                )

            @self.preprocess_stylization_button.on_click
            def _(_):
                save_dir = self.prep_dir
                os.makedirs(save_dir, exist_ok=True)

                room_groups, camera_id_source = self._resolve_preprocess_room_groups(prefer_manual_split=True)
                if camera_id_source == "auto_candidate_manual_split_grouped":
                    print(f"[INFO] preprocess with manual split rooms: {sorted(room_groups.keys())}")
                else:
                    print("[INFO] preprocess with auto camera candidates")

                preprocess_json = {
                    'save_dir': save_dir,
                    'camera_distance_stats': getattr(self, 'last_preprocess_camera_distance_stats', None),
                    'camera_id_source': camera_id_source,
                }
                if -1 in room_groups:
                    preprocess_json['cam_id'] = room_groups[-1]
                preprocess_json['room_groups'] = {
                    str(room_id): cam_list for room_id, cam_list in sorted(room_groups.items(), key=lambda x: x[0])
                }
                with open(os.path.join(save_dir, "preprocess.json"), 'w', encoding='utf-8') as f:
                    json.dump(preprocess_json, f, ensure_ascii=False, indent=4)

                mask_thres = 0.9
                point_xyz = self.gaussian_model.get_xyz.detach()

                for room_id, cam_id in sorted(room_groups.items(), key=lambda x: x[0]):
                    room_label = "all" if room_id == -1 else f"room_{room_id}"
                    room_save_dir = save_dir if room_id == -1 else os.path.join(save_dir, room_label)
                    os.makedirs(room_save_dir, exist_ok=True)

                    print(f"[INFO] preprocess {room_label}, Candidate Length: {len(cam_id)} | List: {cam_id}")
                    all_point_mask = np.zeros(self.gaussian_model.get_xyz.shape[0], dtype=np.bool_)

                    with torch.no_grad():
                        for idx, id in tqdm(enumerate(cam_id), desc=f"preprocess-{room_label}"):
                            cam_root = os.path.join(room_save_dir, f"cam_{id}")
                            if os.path.exists(os.path.join(cam_root, "pano_mask_all.npy")):
                                continue

                            if id == -1:
                                camera_center = np.array([0,0,0])
                                camera_rotation = np.diag(np.ones((3)))
                            else:
                                cam = self.camera_poses[id]
                                camera_center = np.array(cam['position'])
                                if self.camera_focus.value:
                                    camera_rotation = np.array(cam['rotation'])
                                else:
                                    camera_rotation = np.diag(np.ones((3)))

                            image, depth = self.check_and_render_panorama(camera_center, camera_rotation, save_dir=cam_root)
                            if idx != 0:
                                image_mask, depth_mask = self.check_and_render_panorama(
                                    camera_center,
                                    camera_rotation,
                                    mask=all_point_mask,
                                    save_dir=os.path.join(cam_root, "mask"),
                                )
                                mask = (np.array(image_mask).sum(axis=2)/3/255) > mask_thres
                                Image.fromarray(mask.astype(np.uint8)*255).save(os.path.join(cam_root, "pano_mask.png"))
                                masked_image = np.array(image.copy())
                                masked_image[~mask] = 0
                                Image.fromarray(masked_image.astype(np.uint8)).save(os.path.join(cam_root, "pano_img_masked.png"))

                                mask = cv2.erode(mask.astype(np.uint8), np.ones((3,3), np.uint8), iterations=3)
                                Image.fromarray(mask.astype(np.uint8)*255).save(os.path.join(cam_root, "pano_mask_eroded.png"))
                            else:
                                os.makedirs(os.path.join(cam_root, "mask"), exist_ok=True)
                                Image.fromarray(np.zeros_like(image).astype(np.uint8)*255).save(os.path.join(cam_root, "pano_mask.png"))
                                Image.fromarray(np.zeros_like(image).astype(np.uint8)*255).save(os.path.join(cam_root, "pano_mask_eroded.png"))
                                Image.fromarray(np.zeros_like(image).astype(np.uint8)*255).save(os.path.join(cam_root, "mask", "pano_img.png"))

                            cur_point_map, cur_point_mask = get_hidden_point_mask(self.gaussian_model.get_xyz, camera_center)
                            cur_point_mask = cur_point_mask.astype(np.bool_)
                            np.save(os.path.join(cam_root, "pano_mask.npy"), cur_point_mask)
                            all_point_mask = all_point_mask | cur_point_mask
                            np.save(os.path.join(cam_root, "pano_mask_all.npy"), all_point_mask)

                    # print(f"Precomputing staged distance-based pano masks for {room_label}...")
                    # for stage_i, stage_id in enumerate(tqdm(cam_id, desc=f"stage-mask-{room_label}")):
                    #     stage_root = os.path.join(room_save_dir, f"cam_{stage_id}", "mask")
                    #     os.makedirs(stage_root, exist_ok=True)

                    #     active_cam_ids = cam_id[: stage_i + 1]
                    #     active_centers = []
                    #     active_visibility_masks = []
                    #     for active_id in active_cam_ids:
                    #         center = np.load(os.path.join(room_save_dir, f"cam_{active_id}", "camera_center.npy"))
                    #         active_centers.append(center)
                    #         vis_mask = np.load(os.path.join(room_save_dir, f"cam_{active_id}", "pano_mask.npy")).astype(np.bool_)
                    #         active_visibility_masks.append(vis_mask)

                    #     if len(active_cam_ids) == 1:
                    #         only_id = active_cam_ids[0]
                    #         out_dir = os.path.join(stage_root, f"cam_{only_id}")
                    #         os.makedirs(out_dir, exist_ok=True)
                    #         pano_img = cv2.imread(os.path.join(room_save_dir, f"cam_{stage_id}", "pano_img.png"))
                    #         full_mask = np.ones(pano_img.shape[:2], dtype=np.uint8) * 255
                    #         full_mask = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2RGB)
                    #         Image.fromarray(full_mask).save(os.path.join(out_dir, "pano_mask.png"))
                    #         continue

                    #     active_centers_t = torch.tensor(
                    #         np.stack(active_centers, axis=0),
                    #         dtype=point_xyz.dtype,
                    #         device=point_xyz.device,
                    #     )
                    #     dist_to_active = torch.cdist(point_xyz, active_centers_t)
                    #     visible_mat = torch.from_numpy(np.stack(active_visibility_masks, axis=1)).to(point_xyz.device)
                    #     inf_dist = torch.full_like(dist_to_active, float("inf"))
                    #     visible_dist = torch.where(visible_mat, dist_to_active, inf_dist)
                    #     nearest_cam_idx = torch.argmin(visible_dist, dim=1)
                    #     has_visible_camera = torch.isfinite(torch.min(visible_dist, dim=1).values)

                    #     for local_j, every_id in enumerate(active_cam_ids):
                    #         out_dir = os.path.join(stage_root, f"cam_{every_id}")
                    #         os.makedirs(out_dir, exist_ok=True)

                    #         nearest_point_mask = ((nearest_cam_idx == local_j) & has_visible_camera).detach().cpu().numpy().astype(np.bool_)
                    #         other_visible_mask = np.zeros_like(nearest_point_mask)
                    #         for other_j, other_id in enumerate(active_cam_ids):
                    #             if other_j == local_j:
                    #                 continue
                    #             other_visible_mask = other_visible_mask | active_visibility_masks[other_j]
                    #         point_mask = ~(other_visible_mask & (~nearest_point_mask))

                    #         np.save(os.path.join(out_dir, "point_mask.npy"), point_mask)
                    #         every_center = np.load(os.path.join(room_save_dir, f"cam_{every_id}", "camera_center.npy"))
                    #         every_rotation = np.load(os.path.join(room_save_dir, f"cam_{every_id}", "camera_rotation.npy"))

                    #         mask_pano, _ = self.render_panorama(
                    #             every_center,
                    #             camera_rotation=every_rotation,
                    #             mask=point_mask,
                    #             save_dir=out_dir,
                    #         )
                    #         mask = ~((np.array(mask_pano).sum(axis=2)/3/255) < (1-mask_thres))
                    #         Image.fromarray(mask.astype(np.uint8)*255).save(os.path.join(out_dir, "pano_mask.png"))

                print("Preprocess success...")

            @self.clear_prep_cache_button.on_click
            def _(_):
                if self.prep_dir is None or self.prep_dir == "":
                    print("[WARN] --prep_dir is empty, skip clearing.")
                    return
                cam_count, removed_count = self.clear_prep_cam_folders(self.prep_dir)
                print(f"[INFO] clear done in {self.prep_dir}: {removed_count} folders removed from {cam_count} cam directories.")

            @self.start_stylization_button.on_click
            def _(_):
                # save_dir = "preprocess/drjohnson_room4_refine"

                save_dir = self.prep_dir
                with open(os.path.join(save_dir, "preprocess.json"), 'r', encoding='utf-8') as f:
                    prep_json = json.load(f)
                
                self.raw_features_dc = self.gaussian_model._features_dc.clone()
                self.raw_features_rest = self.gaussian_model._features_rest.clone()

                # self.color_update_proj(
                #     # camera_center_path="/data/hyh/github/2D-GS-Viser-Viewer/preprocess/playroom_cartoon/cam_center/camera_center.npy",
                #     # styled_img_path="/data/hyh/github/2D-GS-Viser-Viewer/preprocess/playroom_cartoon/cam_center/pano_styled.png",
                #     # depth_path="/data/hyh/github/2D-GS-Viser-Viewer/preprocess/playroom_cartoon/cam_center/pano_depth.npy",

                #     camera_center_path=os.path.join(save_dir, f"cam_{cam_id[0]}", "camera_center.npy"),
                #     camera_rotation=np.load(os.path.join(save_dir, f'cam_{cam_id[0]}', 'camera_rotation.npy')),
                #     styled_img_path=os.path.join(save_dir, f"cam_{cam_id[0]}", "styled", "pano_styled_refined.png"),
                #     depth_path=os.path.join(save_dir, f"cam_{cam_id[0]}", "pano_depth.npy"),
                #     point_mask=np.load(os.path.join(save_dir, f"cam_{cam_id[0]}", "pano_mask_all.npy")).astype(np.bool_),
                #     upscale=4,
                # )
                # self.viewer_renderer.update_pc_features()

                # self.hidden_color_propogation(os.path.join(save_dir, f"cam_{cam_id[0]}", "pano_mask_all.npy"))
                # self.viewer_renderer.update_pc_features()

                # camera_center = np.load(os.path.join(save_dir, f'cam_{cam_id[0]}', 'camera_center.npy'))
                # r_image, r_depth = self.check_and_render_panorama(camera_center, save_dir=os.path.join(save_dir, f'cam_{cam_id[0]}', 'after_propagation'))

                # self.retrain_scene(save_dir)

                self.gaussian_model.active_sh_degree = 3

                room_groups = prep_json.get('room_groups', None)
                if isinstance(room_groups, dict) and len(room_groups) > 0:
                    for room_id_text, room_cam_ids in sorted(room_groups.items(), key=lambda x: int(x[0])):
                        room_id = int(room_id_text)
                        room_save_dir = save_dir if room_id == -1 else os.path.join(save_dir, f"room_{room_id}")
                        if not os.path.isdir(room_save_dir):
                            print(f"[WARN] room preprocess dir not found, skip room_{room_id}: {room_save_dir}")
                            continue

                        room_point_mask = None
                        room_mask_path = os.path.join(save_dir, "split", f"room_{room_id}_point_mask.npy")
                        if room_id != -1 and os.path.exists(room_mask_path):
                            room_point_mask = np.load(room_mask_path).astype(np.bool_)
                        elif room_id != -1 and room_id in self.manual_split_room_point_masks:
                            room_point_mask = self.manual_split_room_point_masks[room_id].astype(np.bool_)

                        print(f"[INFO] start stylization for room_{room_id}, cameras={len(room_cam_ids)}")
                        self.color_update(room_cam_ids, room_save_dir, room_point_mask=room_point_mask, room_id=room_id)
                else:
                    cam_id = prep_json['cam_id']
                    self.color_update(cam_id, save_dir)
                self.update_client()
                print("update success.")
            
            @self.start_refine_button.on_click
            def _(_):
                # save_dir = "preprocess/drjohnson_office_refine"
                save_dir = self.prep_dir

                with open(os.path.join(save_dir, "preprocess.json"), 'r', encoding='utf-8') as f:
                    prep_json = json.load(f)
                
                # cam_id = [112,99,10]
                cam_id = prep_json.get('cam_id', None)
                if cam_id is None:
                    room_groups = prep_json.get('room_groups', {})
                    cam_id = []
                    for _room_id, room_cam_ids in sorted(room_groups.items(), key=lambda x: int(x[0])):
                        cam_id.extend(room_cam_ids)

                self.color_refine(cam_id, save_dir)
                self.update_client()
                print("refine success.")
            
            @self.start_pers_button.on_click
            def _(_):
                point_centers = self.gaussian_model.get_xyz.detach().cpu().numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(point_centers)
                o3d.io.write_point_cloud("point_cloud.pcd", pcd)

                # save_dir = self.prep_dir
                # with open(os.path.join(save_dir, "preprocess.json"), 'r', encoding='utf-8') as f:
                #     prep_json = json.load(f)
                # cam_id = prep_json['cam_id']

                # self.color_refine(cam_id, save_dir)
                depth_splatting = DepthSplatting()
                # depth_splatting.init_pts(len(self.colmap_cameras)+6)

                pers_params = [
                    (90, 0,   0),
                    (90, 90,  0),
                    (90, 180, 0),
                    (90, 270, 0),
                
                    (90, 0, 90),
                    (90, 0, -90),
                ]

                # Rs = [torch.from_numpy(vtf.SO3.from_rpy_radians(math.radians(p[2]), math.radians(p[1]), math.radians(0)).as_matrix()) for p in pers_params]
                Rs = [vtf.SO3.from_rpy_radians(math.radians(p[2]), math.radians(p[1]), math.radians(90)).as_matrix() for p in pers_params]
                T = torch.tensor([0,0,0])

                pano_images = np.load("test_render/cam_-1/pano_pimages.npy")
                pano_depthes = np.load("test_render/cam_-1/pano_pdepthes.npy")
                for i in range(len(pano_images)):
                    # color = o3d.t.io.read_image(pano_images).to(self.device)
                    # depth = o3d.t.io.read_image().to(self.device)
                    color = o3d.t.geometry.Image(np.ascontiguousarray(pano_images[i]))
                    depth = o3d.t.geometry.Image(np.ascontiguousarray(pano_depthes[i][...,0]))
                    # rgbd = o3d.geometry.RGBDImage()
                    # rgbd.color = color
                    # rgbd.depth = depth
                    rgbd = o3d.t.geometry.RGBDImage(color, depth)

                    intrinsic = o3d.core.Tensor([[fov2focal(math.radians(90), 1024), 0, 512],
                                                 [0, fov2focal(math.radians(90), 1024), 512],
                                                 [0, 0, 1]])
                    # intrinsic = o3d.camera.PinholeCameraIntrinsic()
                    # intrinsic.set_intrinsics(
                    #     1024,1024,
                    #     fov2focal(math.radians(90), 1024), fov2focal(math.radians(90), 1024),
                    #     512,512
                    # )
                    Rt = np.zeros((4, 4))
                    Rt[:3, :3] = Rs[i]
                    Rt[:3, 3] = T
                    Rt[3, 3] = 1.0
                    extrinsic = o3d.core.Tensor(np.linalg.inv(Rt))
                    depth_splatting.perspective_proj(rgbd, intrinsic, extrinsic, 1024, 1024)
                    i=6
                

                cam_id = list(range(len(self.colmap_cameras)))
                cam_centers = [np.linalg.norm(self.camera_poses[id]['position']) for id in cam_id]
                cam_id = sorted(cam_id, key=lambda x:cam_centers[cam_id.index(x)])
                for id in cam_id:
                    print(f"proprocessing camera {id}")
                    target_colmap_camera = self.colmap_cameras[id]
                    results = self.render(target_colmap_camera.R, target_colmap_camera.T,
                                            target_colmap_camera.FovX, target_colmap_camera.FovY,
                                            target_colmap_camera.width, target_colmap_camera.height)
                    # Image.fromarray((results['render'].clip(0,1).permute(1, 2, 0)*255).detach().cpu().numpy().astype(np.uint8)).save("capture.png")
                    color = (results['render'].clip(0,1).permute(1, 2, 0)*255).detach().cpu().numpy().astype(np.uint8)
                    depth = (results['surf_depth'][0,:,:,0]).detach().cpu().numpy()
                    color = o3d.t.geometry.Image(np.ascontiguousarray(color))
                    depth = o3d.t.geometry.Image(np.ascontiguousarray(depth))
                    rgbd = o3d.t.geometry.RGBDImage(color, depth)
                    # rgbd = o3d.geometry.RGBDImage()
                    # rgbd.color = color
                    # rgbd.depth = depth
                    # intrinsic = o3d.camera.PinholeCameraIntrinsic()
                    # intrinsic.set_intrinsics(
                    #     target_colmap_camera.width, target_colmap_camera.height,
                    #     fov2focal(math.radians(target_colmap_camera.FovX),target_colmap_camera.width),
                    #     fov2focal(math.radians(target_colmap_camera.FovY),target_colmap_camera.height),
                    #     target_colmap_camera.width//2, target_colmap_camera.height//2,
                    # )
                    fx = fov2focal(target_colmap_camera.FovX,target_colmap_camera.width//2)
                    fy = fov2focal(target_colmap_camera.FovY,target_colmap_camera.height//2)
                    cx, cy = target_colmap_camera.width//2, target_colmap_camera.height//2
                    intrinsic = o3d.core.Tensor([[fx, 0, cx],
                                                 [0, fy, cy],
                                                 [0, 0, 1]])
                    Rt = np.zeros((4, 4))
                    Rt[:3, :3] = target_colmap_camera.R
                    Rt[:3, 3] = target_colmap_camera.T
                    Rt[3, 3] = 1.0
                    extrinsic = o3d.core.Tensor(np.linalg.inv(Rt))
                    depth_splatting.perspective_wrap(rgbd, intrinsic, extrinsic, target_colmap_camera.width, target_colmap_camera.height)
                    ij=1


                    # print("save current view success.")

                
                self.update_client()
                print("refine success.")

            @self.render_panorama_button.on_click
            def _(_):
                # self.gaussian_model.load_ply('preprocess/playroom_room3/cam_20/styled/scene_styled.ply')
                # self.viewer_renderer.update_pc_features()
                image, depth = self.render_panorama(np.array([0,0,0]), save_dir=os.path.join("test_render", f"cam_-1"))

    def rerender_for_all_client(self):
        for client_id in self.clients:
            try:
                # switch to low resolution mode first, then notify the client to render
                self.clients[client_id].state = "low"
                self.clients[client_id].render_trigger.set()
            except:
                # ignore errors
                pass

    def _handle_option_updated(self, _):
        """
        Simply push new render to all client
        """
        return self.rerender_for_all_client()

    def _handle_new_client(self, client: viser.ClientHandle) -> None:
        """
        Create and start a thread for every new client
        """

        # create client thread
        client_thread = ClientThread(self, self.viewer_renderer, client)
        client_thread.start()
        # store this thread
        self.clients[client.client_id] = client_thread

    def _handle_client_disconnect(self, client: viser.ClientHandle):
        """
        Destroy client thread when client disconnected
        """

        try:
            if client.client_id in self.manual_split_rect_select_clients:
                self.manual_split_rect_select_clients.remove(client.client_id)
            self.clients[client.client_id].stop()
            del self.clients[client.client_id]
        except Exception as err:
            print(err)

if __name__ == "__main__":
    parser = ArgumentParser()
    # lp = ModelParams(parser)
    # op = OptimizationParams(parser)
    # pp = PipelineParams(parser)
    parser.add_argument("model_path", type=str)
    parser.add_argument("--source_path", "-s", type=str, default="")
    parser.add_argument("--host", "-a", type=str, default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--background_color", "-b",
                        type=str, nargs="+", default=["gray"],
                        help="e.g.: white, gray, black, [0 0 0], [0.5 0.5 0.5], [1 1 1]")
    parser.add_argument("--image_format", "--image-format", "-f", type=str, default="jpeg")
    parser.add_argument("--reorient", "-r", type=str, default="auto",
                        help="whether reorient the scene, available values: auto, enable, disable")
    parser.add_argument("--sh_degree", "--sh-degree", "--sh",
                        type=int, default=0)
    parser.add_argument("--enable_transform", "--enable-transform",
                        action="store_true", default=False,
                        help="Enable transform options on Web UI. May consume more memory")
    parser.add_argument("--show_cameras", "--show-cameras",
                        dest="show_cameras", action="store_true", default=False)
    parser.add_argument("--hide_cameras", "--hide-cameras",
                        dest="show_cameras", action="store_false")
    parser.add_argument("--cameras-json", "--cameras_json", type=str, default=None)
    parser.add_argument("--up", nargs=3, required=False, type=float, default=None)
    parser.add_argument("--default_camera_position", "--dcp", nargs=3, required=False, type=float, default=None)
    parser.add_argument("--default_camera_look_at", "--dcla", nargs=3, required=False, type=float, default=None)

    parser.add_argument("--no_edit_panel", action="store_true", default=False)
    parser.add_argument("--no_render_panel", action="store_true", default=False)

    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--crop_box_size", type=float, default=16.0)
    parser.add_argument("--float32_matmul_precision", "--fp", type=str, default=None)
    parser.add_argument("--from_direct_path", type=str, default=None)

    parser.add_argument("--style_img", type=str, default=None)
    parser.add_argument("--prep_dir", type=str, default=None)
    args, unknown_args = parser.parse_known_args()

    # set torch float32_matmul_precision
    if args.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(args.float32_matmul_precision)
    del args.float32_matmul_precision

    # arguments post process
    if len(args.background_color) == 1 and isinstance(args.background_color[0], str):
        if args.background_color[0] == "white":
            args.background_color = [1., 1., 1.]
        elif args.background_color[0] == "black":
            args.background_color = [0., 0., 0.]
        else:
            args.background_color = [0.5, 0.5, 0.5]
    else:
        args.background_color = tuple([float(i) for i in args.background_color])

    # create viewer
    viewer_init_args = {key: getattr(args, key) for key in vars(args)}
    viewer = Viewer(args, **viewer_init_args)
    viewer.start()
