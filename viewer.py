import os
import sys
import math
import glob
import shutil
import time
import json
import torch
import numpy as np
import viser
import viser.transforms as vtf
import warnings
warnings.filterwarnings('ignore')
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '2d_gaussian_splatting'))

from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple, Literal, List
from viser.theme import TitlebarConfig, TitlebarImage

from internal.viewer import ViewerRenderer, ClientThread
from internal.viewer import GaussianModelforViewer as GaussianModel
from internal.viewer.ui import RenderPanel, TransformPanel, EditPanel

from PIL import Image
import cv2
from internal.utils.sh_utils import RGB2SH, SH2RGB
from internal.cameras.cameras import Cameras

from internal.utils.graphics_utils import fov2focal
from internal.utils.pano_utils import get_depth_distort

import internal.utils.equirec.Equirec2Perspec as E2P
import internal.utils.equirec.multi_Perspec2Equirec as m_P2E
from scene.dataset_readers import sceneLoadTypeCallbacks

from internal.utils.depth_proj import DepthSplatting
import random
from tqdm import tqdm

from arguments import OptimizationParams
from omegaconf import OmegaConf
from scene.cameras import Simple_Camera
from diffusers_inference import generate_image

import open3d as o3d
from internal.utils.point_cloud import get_hidden_point_mask
from internal.utils.knn import K_nearest_neighbors
from internal.utils.loss_utils import *
from internal.utils.adain_utils.adain_api import generate_adain

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

    def _init_camera_poses(self, cameras_json_path):
        if not os.path.exists(cameras_json_path):
            return []
        with open(cameras_json_path, "r") as f:
            camera_poses = json.load(f)
        if camera_poses:
            self.camera_center = np.mean(np.asarray([i["position"] for i in camera_poses]), axis=0)
        self.camera_poses = camera_poses

        col_scene = sceneLoadTypeCallbacks["Colmap"](self.source_path, None, False)
        self.scene = col_scene
        self.colmap_cameras = col_scene.train_cameras

    def get_gpu_memory_usage(self):
        total_memory = torch.cuda.memory_allocated() + torch.cuda.memory_reserved()
        return f"{total_memory / 1024 ** 2:.1f} / {self.total_device_memory:.1f} MB"

    def _record_pano_timing(self, timings):
        if not hasattr(self, "pano_timing_stats"):
            self.pano_timing_stats = {}
        for name, elapsed in timings.items():
            stats = self.pano_timing_stats.setdefault(
                name,
                {"count": 0, "total": 0.0, "min": float("inf"), "max": 0.0},
            )
            elapsed = float(elapsed)
            stats["count"] += 1
            stats["total"] += elapsed
            stats["min"] = min(stats["min"], elapsed)
            stats["max"] = max(stats["max"], elapsed)

    def _print_pano_timing_summary(self, timings):
        self._record_pano_timing(timings)
        print("[PANORAMA TIMING] current:")
        for name, elapsed in timings.items():
            print(f"  {name}: {elapsed:.4f}s")
        print("[PANORAMA TIMING] cumulative min/max/total/avg:")
        for name, stats in self.pano_timing_stats.items():
            avg = stats["total"] / max(stats["count"], 1)
            print(
                f"  {name}: min={stats['min']:.4f}s max={stats['max']:.4f}s "
                f"total={stats['total']:.4f}s avg={avg:.4f}s n={stats['count']}"
            )

    def add_cameras_to_scene(self, viser_server):
        if len(self.camera_poses) == 0:
            return

        self.camera_handles = []
        camera_pose_transform = np.linalg.inv(self.camera_transform.cpu().numpy())
        for camera in self.camera_poses:
            name = camera["img_name"]
            c2w = np.eye(4)
            c2w[:3, :3] = np.asarray(camera["rotation"])
            c2w[:3, 3] = np.asarray(camera["position"])
            c2w[:3, 1:3] *= -1
            c2w = np.matmul(camera_pose_transform, c2w)

            R = vtf.SO3.from_matrix(c2w[:3, :3])
            R = R @ vtf.SO3.from_x_radians(np.pi)

            cx = camera["width"] // 2
            cy = camera["height"] // 2
            fx = camera["fx"]

            camera_handle = viser_server.add_camera_frustum(
                name="cameras/{}".format(name),
                fov=float(2 * np.arctan(cx / fx)),
                scale=0.05,
                aspect=float(cx / cy),
                wxyz=R.wxyz,
                position=c2w[:3, 3],
                color=(255, 255, 0),
            )

            @camera_handle.on_click
            def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz

            self.camera_handles.append(camera_handle)

        self.show_cameras_frustrum = viser_server.add_gui_button("Show Train Cameras")
        self.camera_visible = True
        @self.show_cameras_frustrum.on_click
        def toggle_camera_visibility(_):
            with viser_server.atomic():
                self.camera_visible = not self.camera_visible
                for i in self.camera_handles:
                    i.visible = self.camera_visible

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
        buttons = ()
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
            **pano_camera_params,
        )[0].to_device(self.device)

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
            **pano_camera_params,
        )[0].to_device(self.device)

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
        total_start_time = time.perf_counter()
        stage_start_time = total_start_time
        timings = {}
        if verbose: print(f"generate panorama center in {camera_center}...")

        pers_params = [
            (90, 0,   0),
            (90, 90,  0),
            (90, 180, 0),
            (90, 270, 0),

            (90, 0, 90),
            (90, 0, -90),
        ]

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

        cams = [Simple_Camera(0, R.numpy(), T.numpy(), math.radians(90), math.radians(90),
                            res, res, "", 0, trans=Trans.numpy()) for R in Rs]

        override_color = SH2RGB(self.gaussian_model._features_dc.squeeze()) if mask is None else torch.from_numpy(mask)[..., None].float().repeat(1, 3).to(self.device)
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
            'override_color': override_color,
            'sphere_mode': True,
        }
        timings['prepare_cameras_params'] = time.perf_counter() - stage_start_time

        stage_start_time = time.perf_counter()
        with torch.no_grad():
            results = [self.viewer_renderer.render_viewer(cam,
                **render_params,
            ) for cam in cams]
        torch.cuda.synchronize()
        timings['render_faces'] = time.perf_counter() - stage_start_time

        pano_images = []
        pano_depthes = []

        stage_start_time = time.perf_counter()
        depth_distort = get_depth_distort(res=res).to(results[0]['surf_depth'].device) * math.radians(90)

        if render_perspetive:
            stage_start_time = time.perf_counter()
            pano_images = torch.cat([r['render'].unsqueeze(0) for r in results], dim=0)
            timings['perspective_pack'] = time.perf_counter() - stage_start_time
            stage_start_time = time.perf_counter()
            if save_dir:
                for ridx, r in enumerate(results):
                    pimg = Image.fromarray((r['render'].clip(0,1).permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8))
                    pimg.save(os.path.join(save_dir, f'perspective_img{ridx}.png'))
            timings['save_outputs'] = time.perf_counter() - stage_start_time
            timings['total'] = time.perf_counter() - total_start_time
            self._print_pano_timing_summary(timings)
            return pano_images
        else:
            for idx, result in enumerate(results):
                pano_image_face = (result['render'].clip(0,1).permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8)
                pano_depth_face = (result['surf_depth'][0,:,:,0]+depth_distort)[...,None].detach().cpu().numpy()
                pano_images.append(pano_image_face)
                pano_depthes.append(np.repeat(pano_depth_face, 3, axis=-1))
            timings['face_postprocess'] = time.perf_counter() - stage_start_time

            stage_start_time = time.perf_counter()
            pano_image_depthes = [np.concatenate((image.astype(np.float32), depth[..., :1]), axis=-1) for image, depth in zip(pano_images, pano_depthes)]
            ee_image_depth = m_P2E.Perspective(pano_image_depthes, pers_params)
            timings['pack_image_depth_faces'] = time.perf_counter() - stage_start_time

            print("generating panorama image/depth...")
            stage_start_time = time.perf_counter()
            pano_image_depth = ee_image_depth.GetEquirec(res, res*2)
            timings['stitch_image_depth'] = time.perf_counter() - stage_start_time
            timings.update(getattr(ee_image_depth, 'last_timing', {}))

            stage_start_time = time.perf_counter()
            pano_image = pano_image_depth[..., :3]
            pano_depth = np.repeat(pano_image_depth[..., 3:4], 3, axis=-1)
            timings['split_image_depth'] = time.perf_counter() - stage_start_time

            stage_start_time = time.perf_counter()
            if save_dir:
                print(f"saving into {save_dir}")
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(np.clip(pano_image, 0, 255).astype(np.uint8)).save(os.path.join(save_dir, 'pano_img.png'))
                Image.fromarray((((pano_depth-pano_depth.min())/(pano_depth.max()-pano_depth.min())*0.6+0.2)*255).astype(np.uint8)).save(os.path.join(save_dir, 'pano_depth.png')) # depth 50-200
                np.save(os.path.join(save_dir, 'pano_depth.npy'), pano_depth)
                np.save(os.path.join(save_dir, 'camera_center.npy'), Trans.detach().cpu().numpy())
                np.save(os.path.join(save_dir, 'camera_rotation.npy'), camera_rotation)

                np.save(os.path.join(save_dir, 'pano_pimages.npy'), pano_images)
                np.save(os.path.join(save_dir, 'pano_pdepthes.npy'), pano_depthes)
            timings['save_outputs'] = time.perf_counter() - stage_start_time
            timings['total'] = time.perf_counter() - total_start_time
            self._print_pano_timing_summary(timings)
            return pano_image, pano_depth

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
        target_folders = ("after_styled", "styled")
        prep_path = os.path.abspath(prep_dir)
        if not os.path.isdir(prep_path):
            print(f"[WARN] prep_dir not found: {prep_path}")
            return 0, 0

        cam_dirs = sorted(
            path for path in glob.glob(os.path.join(prep_path, "cam_*")) if os.path.isdir(path)
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

        if self.sort_camera_center.value:
            cam_id = sorted(cam_id, key=lambda idx: np.linalg.norm(self.camera_poses[idx]["position"]))

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

    def color_update_proj(self, camera_center_path, camera_rotation, styled_img_path, depth_path, point_mask, upscale=1, color_gaussian=True):
        from internal.utils.pano_utils import direction_to_pano_coord, pano_to_img_coord

        camera_center = np.load(camera_center_path)

        pano_styled = cv2.imread(styled_img_path)
        pano_styled = cv2.resize(pano_styled, (pano_styled.shape[1]*upscale, pano_styled.shape[0]*upscale))
        height, width  = pano_styled.shape[:2]
        pano_styled = cv2.cvtColor(pano_styled, cv2.COLOR_BGR2RGB)

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

        pano_vis = point_mask

        fused_color = RGB2SH((torch.tensor(point_rgbs).float()/255.).cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.gaussian_model.max_sh_degree + 1) ** 2))
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
                self.gaussian_model._features_rest[pano_vis] = _features_rest[pano_vis]
        return features[:, :, 0:1].transpose(1, 2), features[:, :, 1:].transpose(1, 2)

    def hidden_color_propogation(self, point_mask_path):
        print("Propogating Color...")

        point_mask = np.load(point_mask_path).astype(np.bool_)

        point_centers = self.gaussian_model.get_xyz
        _, nn_idx = K_nearest_neighbors(point_centers[point_mask], self.knn_number.value, point_centers[~point_mask])
        nn_feat_dc = self.gaussian_model._features_dc[point_mask][nn_idx]
        nn_feat_rest = self.gaussian_model._features_rest[point_mask][nn_idx]

        mean_feat_dc = nn_feat_dc.mean(axis=1)
        mean_feat_rest = nn_feat_rest.mean(axis=1)
        if self.knn_number.value==1:
            mean_feat_dc = mean_feat_dc.unsqueeze(1)
            mean_feat_rest = mean_feat_rest.unsqueeze(1)
        with torch.no_grad():
            self.gaussian_model._features_dc[~point_mask] = mean_feat_dc
            self.gaussian_model._features_rest[~point_mask] = mean_feat_rest

        self.update_client()
        print("Propogating Success...")

    def color_update(self, cam_id, save_dir):
        print("Reading and processing images...")

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
            return image_tensor

        def train_pano(styled_imgs, feat_dc, feat_rest, steps, random_flag=False, train_res=None, num_sample_views=6):
            if train_res is None:
                train_res = res

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
                    total_loss = total_loss / (B * C * H * W)

                return total_loss

            train_steps = steps

            point_centers = self.gaussian_model.get_xyz
            _, nn_idx = K_nearest_neighbors(point_centers[all_point_mask], self.knn_number.value, point_centers[~all_point_mask])

            for step in tqdm(range(train_steps)):
                self.gaussian_model.update_learning_rate(step+steps)

                if random_flag and step >= 100:
                    random_img = random.sample(styled_imgs, 1)[0]
                else:
                    random_img = styled_imgs[-1]

                camera_center, camera_rotation, equ_styled, equ_mask = \
                    random_img['camera_center'], random_img['camera_rotation'], random_img['styled_img_tensor'], random_img['pano_mask_tensor']

                if train_res != res:
                    equ_styled = torch.nn.functional.interpolate(equ_styled, size=(train_res, train_res), mode='bilinear', align_corners=False)
                    equ_mask = torch.nn.functional.interpolate(equ_mask, size=(train_res, train_res), mode='nearest')

                loss = 0
                color_loss = 0
                ssim_loss = 0
                tv_loss = 0
                propagate_loss = 0
                project_loss = 0
                sampled_indices = random.sample(range(len(pers_params)), min(num_sample_views, len(pers_params)))
                for idx in sampled_indices:
                    fov, theta, phi = pers_params[idx]
                    R = camera_rotation @ vtf.SO3.from_rpy_radians(math.radians(phi), math.radians(theta), math.radians(0)).as_matrix()
                    T = [0,0,0]
                    trans = camera_center

                    cam = Simple_Camera(0, R, T, math.radians(90), math.radians(90),
                                        train_res, train_res, "", 0, trans=trans)

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
                        'sphere_mode': True,
                    }

                    results = self.viewer_renderer.render_viewer(
                        cam,
                        **render_params,
                    )
                    rendered = results['render'].permute(1, 2, 0)

                    color_loss += l1_loss(equ_mask[idx].to(self.device) * rendered.permute(2,0,1)[None], equ_mask[idx].to(self.device) * equ_styled[idx][None].to(self.device))
                    ssim_loss += 1.0 - ssim(equ_mask[idx].to(self.device) * rendered.permute(2,0,1)[None], equ_mask[idx].to(self.device) * equ_styled[idx][None].to(self.device))

                    tv_loss += calculate_total_variation_loss_spherical(rendered.permute(2,0,1)[None])

                    torch.cuda.empty_cache()

                project_loss = l1_loss(self.gaussian_model._features_dc[all_point_mask], feat_dc[all_point_mask]) + \
                                l1_loss(self.gaussian_model._features_rest[all_point_mask],feat_rest[all_point_mask])

                nn_feat_dc = self.gaussian_model._features_dc[all_point_mask][nn_idx]
                nn_feat_rest = self.gaussian_model._features_rest[all_point_mask][nn_idx]
                mean_feat_dc = nn_feat_dc.mean(axis=1)
                mean_feat_rest = nn_feat_rest.mean(axis=1)
                if self.knn_number.value==1:
                    mean_feat_dc = mean_feat_dc.unsqueeze(1)
                    mean_feat_rest = mean_feat_rest.unsqueeze(1)
                propagate_loss = l1_loss(self.gaussian_model._features_dc[~all_point_mask], mean_feat_dc) + \
                                l1_loss(self.gaussian_model._features_rest[~all_point_mask], mean_feat_rest)

                color_term     = 0.8 * color_loss / num_sample_views
                ssim_term      = 0.2 * ssim_loss / num_sample_views
                tv_term        = 1e-3 * tv_loss / num_sample_views
                propagate_term = 1.0 * propagate_loss
                project_term   = 1.0 * project_loss

                loss = color_term + ssim_term + propagate_term + project_term + tv_term

                if step%20==0:
                    print("total_loss:", loss.data,
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
        feat_dc_visible, feat_rest_visible = None, None
        total_cams = len(cam_id)
        for i in range(total_cams):
            idx = cam_id[i]
            camera_center = np.load(os.path.join(save_dir, f'cam_{idx}', 'camera_center.npy'))
            camera_rotation = np.load(os.path.join(save_dir, f'cam_{idx}', 'camera_rotation.npy'))
            pano_mask = cv2.imread(os.path.join(save_dir, f'cam_{idx}', 'pano_mask.png'))
            prev_all_point_mask = np.load(os.path.join(save_dir, f'cam_{cam_id[i-1]}', 'pano_mask_all.npy')).astype(np.bool_)
            cur_point_mask = np.load(os.path.join(save_dir, f'cam_{idx}', 'pano_mask.npy')).astype(np.bool_)
            all_point_mask = np.load(os.path.join(save_dir, f'cam_{idx}', 'pano_mask_all.npy')).astype(np.bool_)
            r_image, r_depth = self.check_and_render_panorama(camera_center, camera_rotation, save_dir=os.path.join(save_dir, f'cam_{idx}', 'styled'))

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
                                                strength=1.0,
                                                )
                    styled_img.save(os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_styled_refined.png'))
                else:
                    adain_img = generate_adain(content_path=os.path.join(save_dir, f'cam_{idx}', 'pano_img.png'),
                                            style_path=self.style_img,
                                            save_path=os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_adain.png'))

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
                                                strength=0.3,
                                                )
                    styled_img.save(os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_styled_refined.png'))
                    inpainted_img.save(os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_styled_inpainted.png'))

                styled_img = cv2.imread(os.path.join(save_dir, f'cam_{idx}', 'styled', 'pano_styled_refined.png'))

            ### Pre-Color Gaussian ###
            print("Init Styled Gaussian...")
            print(f"Before save: {self.get_gpu_memory_usage()}")
            cam_mask = (all_point_mask!=prev_all_point_mask) if i!=0 else all_point_mask
            _feat_dc_visible, _feat_rest_visible = self.color_update_proj(
                camera_center_path=os.path.join(save_dir, f"cam_{idx}", "camera_center.npy"),
                camera_rotation=camera_rotation,
                styled_img_path=os.path.join(save_dir, f"cam_{idx}", "styled", "pano_styled_refined.png"),
                depth_path=os.path.join(save_dir, f"cam_{idx}", "pano_depth.npy"),
                point_mask=cam_mask,
                upscale=4,
                color_gaussian=True,
            )
            if feat_dc_visible is None:
                feat_dc_visible = _feat_dc_visible
                feat_rest_visible = _feat_rest_visible
            else:
                feat_dc_visible[cam_mask] = _feat_dc_visible[cam_mask]
                feat_rest_visible[cam_mask] = _feat_rest_visible[cam_mask]

            self.viewer_renderer.update_pc_features()

            self.hidden_color_propogation(os.path.join(save_dir, f"cam_{idx}", "pano_mask_all.npy"))
            self.viewer_renderer.update_pc_features()

            styled_imgs.append({
                'cam_id': idx,
                'camera_center': camera_center,
                'camera_rotation': camera_rotation,
                'point_mask': cur_point_mask,
                'all_point_mask': all_point_mask,
                'cam_mask': cam_mask,
                'styled_img_tensor': get_pano_imgs_tensor(E2P.Equirectangular(np.array(styled_img))),
                'pano_mask_tensor': get_pano_imgs_tensor(E2P.Equirectangular(255-pano_mask)),
            })

            print(f"After save: {self.get_gpu_memory_usage()}")
            torch.cuda.empty_cache()

            if not os.path.exists(os.path.join(save_dir, f'cam_{idx}', 'styled', 'scene_styled.ply')):
                print(f"Training scene in cam_{idx}...")
                self.gaussian_model.training_setup(opt)
                train_pano(styled_imgs, feat_dc_visible, feat_rest_visible, 200+25*i, random_flag=True, train_res=512)
                self.gaussian_model.save_ply(os.path.join(save_dir, f'cam_{idx}', 'styled', 'scene_styled.ply'))
            else:
                print(f"Loading {os.path.join(save_dir, f'cam_{idx}', 'styled', 'scene_styled.ply')}")
                self.gaussian_model.load_ply(os.path.join(save_dir, f'cam_{idx}', 'styled', 'scene_styled.ply'))

            r_image, r_depth = self.check_and_render_panorama(camera_center, camera_rotation, save_dir=os.path.join(save_dir, f'cam_{idx}', 'after_styled'))

            self.update_client()

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
                    initial_value=0.2,
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

                self.preprocess_stylization_button = server.add_gui_button("Preprocess Stylization")
                self.clear_prep_cache_button = server.add_gui_button("Clear Preprocess Cache")
                self.prompt_text = server.add_gui_text("Prompt", initial_value="", hint="prompt of sytle")
                self.start_stylization_button = server.add_gui_button("Start Stylization")
                self.render_panorama_button = server.add_gui_button("Render Panorama")
                self.start_refine_button = server.add_gui_button("Start Refine")
                self.start_pers_button = server.add_gui_button("Start Perspective Stylization")

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
            if self.show_cameras:
                self.add_cameras_to_scene(server)

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
                    target_colmap_camera = self.colmap_cameras[idx]
                    results = self.render(target_colmap_camera.R, target_colmap_camera.T,
                                          target_colmap_camera.FovX, target_colmap_camera.FovY,
                                          target_colmap_camera.width, target_colmap_camera.height)
                    Image.fromarray((results['render'].clip(0,1).permute(1, 2, 0)*255).detach().cpu().numpy().astype(np.uint8)).save(f"output_trainingview/temp/view_capture{idx:03d}.png")


            @self.render_current_panorama_button.on_click
            def _(_):
                for client in server.get_clients().values():
                    target_camera = self.camera_poses[self.training_view_slider.value]
                    image, depth = self.render_panorama(np.array(target_camera['position']), camera_rotation=target_camera['rotation'], save_dir=os.path.join("test_render", f"cam_{self.training_view_slider.value}"))

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
                video = cv2.VideoWriter(f"test_render_video/temp.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1024, 1024))
                for idx, cam in tqdm(enumerate(camera_data['camera_path'])):
                    camera_to_world = np.array(cam['camera_to_world']).reshape(4,4)
                    Rt = camera_to_world
                    R = (Rt[:3,:3] @ np.linalg.inv(vtf.SO3.from_x_radians(np.pi).as_matrix()))
                    T = -np.linalg.inv(R) @ Rt[:3, 3]
                    results = self.render(R, T,
                                          math.radians(60), math.radians(60),
                                        1024, 1024
                                          )
                    result_img = (results['render'].clip(0,1).permute(1, 2, 0)*255).detach().cpu().numpy().astype(np.uint8)
                    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
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
                print(len(self.camera_poses))
                cam_id = self._get_preprocess_candidate_cam_ids()
                print(f"Candidate Length: {len(cam_id)} | List: {cam_id}")
                if self.render_test_panorama.value:
                    os.makedirs("test_render_pano", exist_ok=True)
                    for id in cam_id:
                        target_camera = self.camera_poses[id]
                        image, depth = self.render_panorama(np.array(target_camera['position']), camera_rotation=np.array(target_camera['rotation']), res=256)
                        save_file_path = os.path.join("test_render_pano", f'pano_img_cam{id}.png')
                        Image.fromarray(image.astype(np.uint8)).save(save_file_path)
                        print(f"Save in {save_file_path}")

            @self.preprocess_stylization_button.on_click
            def _(_):
                cam_id = self._get_preprocess_candidate_cam_ids()

                print(f"Candidate Length: {len(cam_id)} | List: {cam_id}")

                save_dir = self.prep_dir
                os.makedirs(save_dir, exist_ok=True)
                preprocess_json = {
                    'cam_id': cam_id,
                    'save_dir': save_dir,
                    'camera_distance_stats': getattr(self, 'last_preprocess_camera_distance_stats', None),
                }
                with open(os.path.join(save_dir, "preprocess.json"), 'w', encoding='utf-8') as f:
                    json.dump(preprocess_json, f, ensure_ascii=False, indent=4)

                mask_thres = 0.9
                all_point_mask = np.zeros(self.gaussian_model.get_xyz.shape[0], dtype=np.uint8)
                with torch.no_grad():
                    for idx, id in tqdm(enumerate(cam_id)):
                        if os.path.exists(os.path.join(save_dir, f"cam_{id}", "pano_mask_all.npy")): continue
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

                        image, depth = self.check_and_render_panorama(camera_center, camera_rotation, save_dir=os.path.join(save_dir, f"cam_{id}"))
                        if idx!=0:
                            image_mask, depth_mask = self.check_and_render_panorama(camera_center, camera_rotation, mask=all_point_mask, save_dir=os.path.join(save_dir, f"cam_{id}", "mask"))
                            mask = (np.array(image_mask).sum(axis=2)/3/255)>mask_thres
                            Image.fromarray(mask.astype(np.uint8)*255).save(os.path.join(save_dir, f"cam_{id}", "pano_mask.png"))
                            masked_image = np.array(image.copy())
                            masked_image[~mask] = 0
                            Image.fromarray(masked_image.astype(np.uint8)).save(os.path.join(save_dir, f"cam_{id}", "pano_img_masked.png"))

                            mask = cv2.erode(mask.astype(np.uint8), np.ones((3,3), np.uint8), iterations=3)
                            Image.fromarray(mask.astype(np.uint8)*255).save(os.path.join(save_dir, f"cam_{id}", "pano_mask_eroded.png"))
                        else:
                            os.makedirs(os.path.join(save_dir, f"cam_{id}", "mask"), exist_ok=True)
                            Image.fromarray(np.zeros_like(image).astype(np.uint8)*255).save(os.path.join(save_dir, f"cam_{id}", "pano_mask.png"))
                            Image.fromarray(np.zeros_like(image).astype(np.uint8)*255).save(os.path.join(save_dir, f"cam_{id}", "pano_mask_eroded.png"))
                            Image.fromarray(np.zeros_like(image).astype(np.uint8)*255).save(os.path.join(save_dir, f"cam_{id}", "mask", "pano_img.png"))

                        cur_point_map, cur_point_mask = get_hidden_point_mask(self.gaussian_model.get_xyz, camera_center)
                        np.save(os.path.join(save_dir, f"cam_{id}", "pano_mask.npy"), cur_point_mask)
                        all_point_mask = all_point_mask | cur_point_mask
                        np.save(os.path.join(save_dir, f"cam_{id}", "pano_mask_all.npy"), all_point_mask)

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
                save_dir = self.prep_dir
                with open(os.path.join(save_dir, "preprocess.json"), 'r', encoding='utf-8') as f:
                    prep_json = json.load(f)
                cam_id = prep_json['cam_id']

                self.raw_features_dc = self.gaussian_model._features_dc.clone()
                self.raw_features_rest = self.gaussian_model._features_rest.clone()

                self.gaussian_model.active_sh_degree = 3

                self.color_update(cam_id, save_dir)
                self.update_client()
                print("update success.")

            @self.start_refine_button.on_click
            def _(_):
                save_dir = self.prep_dir

                with open(os.path.join(save_dir, "preprocess.json"), 'r', encoding='utf-8') as f:
                    prep_json = json.load(f)

                cam_id = prep_json['cam_id']

                self.color_refine(cam_id, save_dir)
                self.update_client()
                print("refine success.")

            @self.start_pers_button.on_click
            def _(_):
                point_centers = self.gaussian_model.get_xyz.detach().cpu().numpy()
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(point_centers)
                o3d.io.write_point_cloud("point_cloud.pcd", pcd)

                depth_splatting = DepthSplatting()

                pers_params = [
                    (90, 0,   0),
                    (90, 90,  0),
                    (90, 180, 0),
                    (90, 270, 0),

                    (90, 0, 90),
                    (90, 0, -90),
                ]

                Rs = [vtf.SO3.from_rpy_radians(math.radians(p[2]), math.radians(p[1]), math.radians(90)).as_matrix() for p in pers_params]
                T = torch.tensor([0,0,0])

                pano_images = np.load("test_render/cam_-1/pano_pimages.npy")
                pano_depthes = np.load("test_render/cam_-1/pano_pdepthes.npy")
                for i in range(len(pano_images)):
                    color = o3d.t.geometry.Image(np.ascontiguousarray(pano_images[i]))
                    depth = o3d.t.geometry.Image(np.ascontiguousarray(pano_depthes[i][...,0]))
                    rgbd = o3d.t.geometry.RGBDImage(color, depth)

                    intrinsic = o3d.core.Tensor([[fov2focal(math.radians(90), 1024), 0, 512],
                                                 [0, fov2focal(math.radians(90), 1024), 512],
                                                 [0, 0, 1]])
                    Rt = np.zeros((4, 4))
                    Rt[:3, :3] = Rs[i]
                    Rt[:3, 3] = T
                    Rt[3, 3] = 1.0
                    extrinsic = o3d.core.Tensor(np.linalg.inv(Rt))
                    depth_splatting.perspective_proj(rgbd, intrinsic, extrinsic, 1024, 1024)


                cam_id = list(range(len(self.colmap_cameras)))
                cam_centers = [np.linalg.norm(self.camera_poses[id]['position']) for id in cam_id]
                cam_id = sorted(cam_id, key=lambda x:cam_centers[cam_id.index(x)])
                for id in cam_id:
                    print(f"proprocessing camera {id}")
                    target_colmap_camera = self.colmap_cameras[id]
                    results = self.render(target_colmap_camera.R, target_colmap_camera.T,
                                            target_colmap_camera.FovX, target_colmap_camera.FovY,
                                            target_colmap_camera.width, target_colmap_camera.height)
                    color = (results['render'].clip(0,1).permute(1, 2, 0)*255).detach().cpu().numpy().astype(np.uint8)
                    depth = (results['surf_depth'][0,:,:,0]).detach().cpu().numpy()
                    color = o3d.t.geometry.Image(np.ascontiguousarray(color))
                    depth = o3d.t.geometry.Image(np.ascontiguousarray(depth))
                    rgbd = o3d.t.geometry.RGBDImage(color, depth)
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

                self.update_client()
                print("refine success.")

            @self.render_panorama_button.on_click
            def _(_):
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
            self.clients[client.client_id].stop()
            del self.clients[client.client_id]
        except Exception as err:
            print(err)

if __name__ == "__main__":
    parser = ArgumentParser()
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
                        action="store_true")
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
