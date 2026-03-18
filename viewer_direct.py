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
from typing import Tuple, Literal, List
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

        # class GroupParams:
        #     def __init__(self):
        #         self.sh_degree = 0 #1 #3
        #         self.source_path = ""
        #         self.model_path = ""
        #         self.path_style = ""
        #         self.images = "images"
        #         self.resolution = -1
        #         self.forward_facing = False
        #         self.white_background = False
        #         self.data_device = "cuda"
        #         self.eval = False
        #         self.starting_iter = ""
        # gargs = GroupParams()
        # for arg in vars(self.args).items():
        #     setattr(gargs, arg[0], arg[1])
        # self.scene = Scene(gargs, self.gaussian_model)
        # self.colmap_cameras = scene.getTrainCameras()
        col_scene = sceneLoadTypeCallbacks["Colmap"](self.source_path, None, False)
        self.scene = col_scene
        self.colmap_cameras = col_scene.train_cameras

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

    def clear_prep_cam_folders(self, prep_dir, target_folders=("after_styled", "styled")):
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
        
    def color_update_proj(self, camera_center_path, camera_rotation, styled_img_path, depth_path, point_mask, upscale=1, color_gaussian=True):
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
        pano_vis = point_mask

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
    
    def hidden_color_propogation(self, point_mask_path):
        print("Propogating Color...")

        point_mask = np.load(point_mask_path).astype(np.bool_)

        # point_centers = self.gaussian_model.get_xyz
        # _, nn_idx = K_nearest_neighbors(point_centers[point_mask], 5, point_centers[~point_mask])
        # nn_feat_dc = self.gaussian_model._features_dc[point_mask][nn_idx]
        # nn_feat_rest = self.gaussian_model._features_rest[point_mask][nn_idx]
        
        point_color = self.raw_features_dc[:,0,:]
        _, nn_idx = K_nearest_neighbors(point_color[point_mask], self.knn_number.value, point_color[~point_mask])
        nn_feat_dc = self.gaussian_model._features_dc[point_mask][nn_idx]
        nn_feat_rest = self.gaussian_model._features_rest[point_mask][nn_idx]

        mean_feat_dc = nn_feat_dc.mean(axis=1)#, keepdim=True)
        mean_feat_rest = nn_feat_rest.mean(axis=1)#, keepdim=True)
        if self.knn_number.value==1:
            mean_feat_dc = mean_feat_dc.unsqueeze(1)
            mean_feat_rest = mean_feat_rest.unsqueeze(1)
        with torch.no_grad():
            self.gaussian_model._features_dc[~point_mask] = mean_feat_dc
            self.gaussian_model._features_rest[~point_mask] = mean_feat_rest

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

    def color_update(self, cam_id, save_dir): #camera_center_path, styled_img_path):
        print("Reading and processing images...")

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
            
            # equ_styled = E2P.Equirectangular(pano_styled)
            # train_steps = 1500 if steps==0 else self.train_steps.value
            train_steps = steps

            # point_centers = self.gaussian_model.get_xyz
            # _, nn_idx = K_nearest_neighbors(point_centers[all_point_mask], 5, point_centers[~all_point_mask])

            point_color = self.raw_features_dc[:,0,:]
            _, nn_idx = K_nearest_neighbors(point_color[all_point_mask], self.knn_number.value, point_color[~all_point_mask])

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
                random_img = random.sample(styled_imgs, 1)[0] if random_flag else styled_imgs[-1]
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
                    
                    torch.cuda.empty_cache()
                    # perceptual_loss = perceptual(rendered.permute(2,0,1)[None], raw_image.permute(2,0,1)[None])
                    # loss += color_loss #+ perceptual_loss#+ sobel_loss #+ perceptual_loss

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

                loss = 0.8*color_loss + 0.2*ssim_loss + 1*propagate_loss + 1*project_loss
                # loss = 1*color_loss + 1*propagate_loss + 10*project_loss

                if step%20==0:
                    print("total_loss:", loss.data,
                          "\t color_loss:", color_loss.data,
                          "\t ssim_loss:", ssim_loss.data,
                          "\t propagate_loss:", propagate_loss.data,
                          "\t project_loss:", project_loss.data,
                        #   "\t sobel_loss:", sobel_loss.data,
                        #   "\t perceptual_loss:", perceptual_loss.data
                          )
                    self.update_client()
                torch.cuda.empty_cache()
                loss.backward(retain_graph=True)
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
            prev_all_point_mask = np.load(os.path.join(save_dir, f'cam_{cam_id[i-1]}', 'pano_mask_all.npy')).astype(np.bool_)
            cur_point_mask = np.load(os.path.join(save_dir, f'cam_{idx}', 'pano_mask.npy')).astype(np.bool_)
            all_point_mask = np.load(os.path.join(save_dir, f'cam_{idx}', 'pano_mask_all.npy')).astype(np.bool_)
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
            cam_point_mask = np.load(os.path.join(save_dir, f'cam_{idx}', 'mask', f'cam_{idx}', 'point_mask.npy')).astype(np.bool_) if i!=0 else all_point_mask
            

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

            # Load precomputed distance-based pano masks for current stage.
            for sidx in range(len(styled_imgs)):
                every_cam_id = styled_imgs[sidx]['cam_id']
                stage_mask_path = os.path.join(save_dir, f"cam_{idx}", "mask", f"cam_{every_cam_id}", "pano_img.png")
                if os.path.exists(stage_mask_path):
                    stage_mask = cv2.imread(stage_mask_path)
                    stage_cam_point_mask = np.load(os.path.join(save_dir, f'cam_{idx}', 'mask', f'cam_{every_cam_id}', 'point_mask.npy')).astype(np.bool_) if len(styled_imgs)<=1 else np.ones_like(cam_point_mask, dtype=np.bool_)
                    styled_imgs[sidx]['pano_mask_tensor'] = get_pano_imgs_tensor(E2P.Equirectangular(stage_mask))
                    _feat_dc_visible, _feat_rest_visible = self.color_update_proj(
                        camera_center_path=os.path.join(save_dir, f"cam_{every_cam_id}", "camera_center.npy"),
                        camera_rotation=np.load(os.path.join(save_dir, f'cam_{every_cam_id}', 'camera_rotation.npy')),
                        styled_img_path=os.path.join(save_dir, f"cam_{every_cam_id}", "styled", "pano_styled_refined.png"),
                        depth_path=os.path.join(save_dir, f"cam_{every_cam_id}", "pano_depth.npy"),
                        point_mask=stage_cam_point_mask,
                        upscale=4,
                        color_gaussian=True,
                    )
                    if feat_dc_visible is None:
                        feat_dc_visible = _feat_dc_visible
                        feat_rest_visible = _feat_rest_visible
                    else:
                        # feat_dc_visible[cur_point_mask] = (_feat_dc_visible[cur_point_mask] + feat_dc_visible[cur_point_mask])/2
                        # feat_rest_visible[cur_point_mask] = (_feat_rest_visible[cur_point_mask] + feat_rest_visible[cur_point_mask])/2
                        # feat_dc_visible[cam_mask] = _feat_dc_visible[cam_mask]
                        # feat_rest_visible[cam_mask] = _feat_rest_visible[cam_mask]
                        feat_dc_visible[stage_cam_point_mask] = _feat_dc_visible[stage_cam_point_mask]
                        feat_rest_visible[stage_cam_point_mask] = _feat_rest_visible[stage_cam_point_mask]
                        # feat_dc_visible[cur_point_mask] = _feat_dc_visible[cur_point_mask]
                        # feat_rest_visible[cur_point_mask] = _feat_rest_visible[cur_point_mask]

                    self.viewer_renderer.update_pc_features()

            self.hidden_color_propogation(os.path.join(save_dir, f"cam_{idx}", "pano_mask_all.npy"))
            self.viewer_renderer.update_pc_features()

            print(f"After save: {self.get_gpu_memory_usage()}")
            torch.cuda.empty_cache()

            if not os.path.exists(os.path.join(save_dir, f'cam_{idx}', 'styled', 'scene_styled.ply')):
                # print(f"Training scene in cam_{idx}...")
                # self.gaussian_model.training_setup(opt)
                # train_pano(styled_imgs, feat_dc_visible, feat_rest_visible, 200+25*i, random_flag=True)
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
                    max=10,
                    step=1,
                    initial_value=2,
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
                self.sort_camera_pointnum = server.add_gui_checkbox(
                    "Sort by Point Number",
                    initial_value=False,
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
                self.clear_stylization_cache_button = server.add_gui_button("Clear Stylization Cache")
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
                        initial_value=0,
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
                cam_id = list(range(len(self.colmap_cameras)))
                selected_cam = np.array([0,0,0])[None]

                if self.sort_camera_pointnum.value:
                    print("Sorting cameras by point number...")
                    point_xyz = self.gaussian_model.get_xyz
                    cam_pointnums = []
                    with torch.no_grad():
                        for id in tqdm(cam_id, desc="Sort by Point Number"):
                            center = np.array(self.camera_poses[id]['position'])
                            _, point_mask = get_hidden_point_mask(point_xyz, center)
                            cam_pointnums.append((id, int(np.count_nonzero(point_mask))))
                    cam_id = [id for id, _ in sorted(cam_pointnums, key=lambda x: x[1], reverse=True)]
                elif self.sort_camera_center.value:
                    cam_centers = [np.linalg.norm(self.camera_poses[id]['position']) for id in cam_id]
                    cam_id = sorted(cam_id, key=lambda x:cam_centers[cam_id.index(x)])
                if self.start_from_center.value:
                    cam_centers = [np.linalg.norm(self.camera_poses[id]['position']) for id in cam_id]
                    sorted_id = sorted(cam_id, key=lambda x:cam_centers[cam_id.index(x)])
                    for i in range(len(sorted_id)):
                        center = np.array(self.camera_poses[sorted_id[i]]['position'])
                        if not np.all(np.linalg.norm(selected_cam-center, axis=1)>self.camera_gap_slider.value):
                            pass
                        else:
                            nearest_id = sorted_id[i]
                            break
                    cam_id = cam_id[nearest_id:] + cam_id[:nearest_id]
                
                ### Delete Near Camera All ###
                flag = []
                for i in range(len(cam_id)):
                    center = np.array(self.camera_poses[cam_id[i]]['position'])
                    too_near_selected_cam = not np.all(np.linalg.norm(selected_cam-center, axis=1)>self.camera_gap_slider.value)
                    too_close_to_object = self.is_camera_center_too_close(center)
                    if too_near_selected_cam or too_close_to_object:
                        flag.append(False)
                    else:
                        flag.append(True)
                        selected_cam = np.concatenate((selected_cam, center[None]), axis=0)
                cam_id = [cam_id[idx] for idx,tf in enumerate(flag) if tf==True]
                if self.enable_camera_center.value:
                    cam_id.insert(0, -1)
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
                # self.pano_distort = self.get_pano_depth_distort()

                # sample_camera_number = 5

                ### Playroom ###
                # cam_id = [99,98,22,101,147]
                # cam_id = [-1,173,15,200,29]
                cam_id = list(range(len(self.colmap_cameras)))
                selected_cam = np.array([0,0,0])[None]

                if self.sort_camera_pointnum.value:
                    print("Sorting cameras by point number...")
                    point_xyz = self.gaussian_model.get_xyz
                    cam_pointnums = []
                    with torch.no_grad():
                        for id in tqdm(cam_id, desc="Sort by Point Number"):
                            center = np.array(self.camera_poses[id]['position'])
                            _, point_mask = get_hidden_point_mask(point_xyz, center)
                            cam_pointnums.append((id, int(np.count_nonzero(point_mask))))
                    cam_id = [id for id, _ in sorted(cam_pointnums, key=lambda x: x[1], reverse=True)]
                elif self.sort_camera_center.value:
                    cam_centers = [np.linalg.norm(self.camera_poses[id]['position']) for id in cam_id]
                    cam_id = sorted(cam_id, key=lambda x:cam_centers[cam_id.index(x)])
                if self.start_from_center.value:
                    cam_centers = [np.linalg.norm(self.camera_poses[id]['position']) for id in cam_id]
                    sorted_id = sorted(cam_id, key=lambda x:cam_centers[cam_id.index(x)])
                    for i in range(len(sorted_id)):
                        center = np.array(self.camera_poses[sorted_id[i]]['position'])
                        if np.all(np.linalg.norm(selected_cam-center, axis=1)>self.camera_gap_slider.value):
                            nearest_id = sorted_id[i]
                            break
                    cam_id = cam_id[nearest_id:] + cam_id[:nearest_id]
                
                # cam_centers = sorted(camera_center)

                ### Delete Near Camera ###
                # last_center = np.array([0,0,0])
                # flag = []
                # for i in range(len(cam_id)):
                #     center = np.array(self.camera_poses[cam_id[i]]['position'])
                #     if np.linalg.norm(center - last_center) < 1:
                #         flag.append(False)
                #     else:
                #         flag.append(True)
                #         last_center = center
                # cam_id = [cam_id[idx] for idx,tf in enumerate(flag) if tf==True]
                # print(cam_id)

                ### Delete Near Camera All ###
                flag = []
                for i in range(len(cam_id)):
                    center = np.array(self.camera_poses[cam_id[i]]['position'])
                    too_near_selected_cam = not np.all(np.linalg.norm(selected_cam-center, axis=1)>self.camera_gap_slider.value)
                    too_close_to_object = self.is_camera_center_too_close(center)
                    if too_near_selected_cam or too_close_to_object:
                        flag.append(False)
                    else:
                        flag.append(True)
                        selected_cam = np.concatenate((selected_cam, center[None]), axis=0)
                cam_id = [cam_id[idx] for idx,tf in enumerate(flag) if tf==True]

                if self.enable_camera_center.value:
                    cam_id.insert(0, -1)
                print(f"Candidate Length: {len(cam_id)} | List: {cam_id}")
                

                # print("test render...")
                # test_render_dir = "test_render_preprocess"
                # os.makedirs(test_render_dir, exist_ok=True)
                # with torch.no_grad():
                #     for idx, id in tqdm(enumerate(cam_id)):
                #         if id == -1:
                #             camera_center = np.array([0,0,0])
                #             camera_rotation = np.eye(3)
                #         else:
                #             cam = self.camera_poses[id]
                #             camera_center = np.array(cam['position'])
                #             camera_rotation = np.array(cam['rotation'])
                        
                #         image, depth = self.render_panorama(camera_center, res=256)
                #         Image.fromarray(image.astype(np.uint8)).save(os.path.join(test_render_dir, f'pano_img_cam{idx}.png'))
                
                # save_dir = 'preprocess/drjohnson_office_refine'
                save_dir = self.prep_dir
                os.makedirs(save_dir, exist_ok=True)
                preprocess_json = {
                    'cam_id': cam_id,
                    'save_dir': save_dir,
                }
                with open(os.path.join(save_dir, "preprocess.json"), 'w', encoding='utf-8') as f:
                    json.dump(preprocess_json, f, ensure_ascii=False)

                mask_thres = 0.9
                all_point_mask = np.zeros(self.gaussian_model.get_xyz.shape[0], dtype=np.uint8)
                # cur_point_mask = np.zeros(self.gaussian_model.get_xyz.shape[0], dtype=np.uint8)
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

                print("Precomputing staged distance-based pano masks...")
                point_xyz = self.gaussian_model.get_xyz.detach()
                for stage_i, stage_id in enumerate(tqdm(cam_id)):
                    stage_root = os.path.join(save_dir, f"cam_{stage_id}", "mask")
                    stage_all_point_mask = np.load(os.path.join(save_dir, f"cam_{stage_id}", "pano_mask_all.npy")).astype(np.bool_)
                    os.makedirs(stage_root, exist_ok=True)

                    active_cam_ids = cam_id[: stage_i + 1]
                    active_centers = []
                    active_visibility_masks = []
                    for active_id in active_cam_ids:
                        center = np.load(os.path.join(save_dir, f"cam_{active_id}", "camera_center.npy"))
                        active_centers.append(center)
                        vis_mask = np.load(os.path.join(save_dir, f"cam_{active_id}", "pano_mask.npy")).astype(np.bool_)
                        active_visibility_masks.append(vis_mask)

                    # if len(active_cam_ids) == 1:
                    #     only_id = active_cam_ids[0]
                    #     out_dir = os.path.join(stage_root, f"cam_{only_id}")
                    #     os.makedirs(out_dir, exist_ok=True)
                    #     pano_img = cv2.imread(os.path.join(save_dir, f"cam_{stage_id}", "pano_img.png"))
                    #     full_mask = np.ones(pano_img.shape[:2], dtype=np.uint8) * 255
                    #     full_mask = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2RGB)
                    #     Image.fromarray(full_mask).save(os.path.join(out_dir, "pano_mask.png"))
                    #     continue

                    active_centers_t = torch.tensor(
                        np.stack(active_centers, axis=0),
                        dtype=point_xyz.dtype,
                        device=point_xyz.device,
                    )
                    dist_to_active = torch.cdist(point_xyz, active_centers_t)
                    visible_mat = torch.from_numpy(np.stack(active_visibility_masks, axis=1)).to(point_xyz.device)
                    inf_dist = torch.full_like(dist_to_active, float("inf"))
                    visible_dist = torch.where(visible_mat, dist_to_active, inf_dist)
                    nearest_cam_idx = torch.argmin(visible_dist, dim=1)
                    has_visible_camera = torch.isfinite(torch.min(visible_dist, dim=1).values)

                    for local_j, every_id in enumerate(active_cam_ids):
                        out_dir = os.path.join(stage_root, f"cam_{every_id}")
                        os.makedirs(out_dir, exist_ok=True)

                        point_mask = ((nearest_cam_idx == local_j) & has_visible_camera).detach().cpu().numpy().astype(np.bool_) & stage_all_point_mask
                        np.save(os.path.join(out_dir, "point_mask.npy"), point_mask)
                        every_center = np.load(os.path.join(save_dir, f"cam_{every_id}", "camera_center.npy"))
                        every_rotation = np.load(os.path.join(save_dir, f"cam_{every_id}", "camera_rotation.npy"))

                        mask_pano, _ = self.render_panorama(
                            every_center,
                            camera_rotation=every_rotation,
                            mask=point_mask,
                            save_dir=out_dir,
                        )
                        mask = (np.array(mask_pano).sum(axis=2)/3/255)>mask_thres
                        Image.fromarray(mask.astype(np.uint8)*255).save(os.path.join(out_dir, "pano_mask.png"))
                
                print("Preprocess success...")

            @self.clear_stylization_cache_button.on_click
            def _(_):
                if self.prep_dir is None or self.prep_dir == "":
                    print("[WARN] --prep_dir is empty, skip clearing.")
                    return
                cam_count, removed_count = self.clear_prep_cam_folders(self.prep_dir, target_folders=("styled", "after_styled"))
                print(f"[INFO] stylization cache clear done in {self.prep_dir}: {removed_count} folders removed from {cam_count} cam directories.")

            @self.clear_prep_cache_button.on_click
            def _(_):
                if self.prep_dir is None or self.prep_dir == "":
                    print("[WARN] --prep_dir is empty, skip clearing.")
                    return
                cam_count, removed_count = self.clear_prep_cam_folders(self.prep_dir, target_folders=("mask",))
                print(f"[INFO] preprocess cache clear done in {self.prep_dir}: {removed_count} folders removed from {cam_count} cam directories.")

            @self.start_stylization_button.on_click
            def _(_):
                # save_dir = "preprocess/drjohnson_room4_refine"

                save_dir = self.prep_dir
                with open(os.path.join(save_dir, "preprocess.json"), 'r', encoding='utf-8') as f:
                    prep_json = json.load(f)
                cam_id = prep_json['cam_id']
                
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

                self.gaussian_model.active_sh_degree = 0

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
