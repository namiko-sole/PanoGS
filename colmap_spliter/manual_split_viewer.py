#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import trimesh
import viser
import viser.transforms as vtf

from split_rooms import export_room_colmap_submodel, read_points3d_binary_full


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


@dataclass
class CameraEntry:
    image_id: int
    camera_id: int
    name: str
    qvec: np.ndarray
    tvec: np.ndarray
    center: np.ndarray
    width: int
    height: int
    fx: float
    fy: float
    handle: Optional[viser.CameraFrustumHandle] = None


class ManualSplitViewer:
    def __init__(
        self,
        model_path: Path,
        input_root: Path,
        sparse_rel: str,
        output_dir: Path,
        host: str,
        port: int,
        camera_scale: float,
        point_size: float,
        max_points: int,
    ):
        self.model_path = model_path
        self.input_root = input_root
        self.sparse_dir = (input_root / sparse_rel).resolve()
        self.output_dir = output_dir
        self.host = host
        self.port = port
        self.camera_scale = camera_scale
        self.point_size = point_size
        self.max_points = max_points

        self.repo_root = Path(__file__).resolve().parents[1]
        self.colmap_loader = None
        self.images: Dict[int, object] = {}
        self.cameras: Dict[int, object] = {}
        self.point_records: Dict[int, object] = {}
        self.camera_entries: List[CameraEntry] = []

        self.selected_ids: Set[int] = set()
        self.assignments: Dict[int, int] = {}
        self.last_assignments: Optional[Dict[int, int]] = None

        self.point_handle: Optional[viser.PointCloudHandle] = None

    def _load_colmap_loader(self):
        scene_dir = self.repo_root / "2d_gaussian_splatting" / "scene"
        if not scene_dir.exists():
            raise FileNotFoundError(f"Cannot find COLMAP loader directory: {scene_dir}")
        sys.path.insert(0, str(scene_dir))
        import colmap_loader  # type: ignore

        return colmap_loader

    @staticmethod
    def _resolve_ply_path(model_path: Path) -> Path:
        if model_path.is_file() and model_path.suffix.lower() == ".ply":
            return model_path

        if model_path.is_dir():
            cand = model_path / "point_cloud.ply"
            if cand.exists():
                return cand

            iter_root = model_path / "point_cloud"
            if iter_root.exists():
                matches = sorted(iter_root.glob("iteration_*/point_cloud.ply"))
                if matches:
                    def _iter_num(p: Path) -> int:
                        try:
                            return int(p.parent.name.split("_")[-1])
                        except ValueError:
                            return -1

                    matches.sort(key=_iter_num)
                    return matches[-1]

        raise FileNotFoundError(
            f"Cannot find point cloud .ply from --model-path={model_path}. "
            "Expected a .ply file or model dir containing point_cloud/iteration_*/point_cloud.ply."
        )

    @staticmethod
    def _camera_center_from_image(colmap_loader, image_obj) -> np.ndarray:
        r = colmap_loader.qvec2rotmat(image_obj.qvec)
        t = image_obj.tvec.reshape(3)
        return (-r.T @ t).astype(np.float64)

    @staticmethod
    def _get_fx_fy(camera_obj) -> Tuple[float, float]:
        model = camera_obj.model
        p = camera_obj.params

        if model in ("PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"):
            return float(p[0]), float(p[1])
        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE", "FOV", "THIN_PRISM_FISHEYE"):
            return float(p[0]), float(p[0])
        return float(p[0]), float(p[0])

    @staticmethod
    def _room_color(room_id: int) -> Tuple[int, int, int]:
        return ROOM_PALETTE[room_id % len(ROOM_PALETTE)]

    def _load_data(self):
        colmap_loader = self._load_colmap_loader()
        self.colmap_loader = colmap_loader

        images_bin = self.sparse_dir / "images.bin"
        cameras_bin = self.sparse_dir / "cameras.bin"
        points_bin = self.sparse_dir / "points3D.bin"

        if not images_bin.exists() or not cameras_bin.exists() or not points_bin.exists():
            raise FileNotFoundError(
                f"COLMAP binaries not found under {self.sparse_dir}. "
                "Need images.bin, cameras.bin, points3D.bin."
            )

        self.images = colmap_loader.read_extrinsics_binary(str(images_bin))
        self.cameras = colmap_loader.read_intrinsics_binary(str(cameras_bin))
        self.point_records = read_points3d_binary_full(points_bin)

        self.camera_entries = []
        for image_id in sorted(self.images.keys()):
            img = self.images[image_id]
            cam = self.cameras[int(img.camera_id)]
            fx, fy = self._get_fx_fy(cam)
            center = self._camera_center_from_image(colmap_loader, img)

            self.camera_entries.append(
                CameraEntry(
                    image_id=int(img.id),
                    camera_id=int(img.camera_id),
                    name=img.name,
                    qvec=np.asarray(img.qvec, dtype=np.float64),
                    tvec=np.asarray(img.tvec, dtype=np.float64),
                    center=center,
                    width=int(cam.width),
                    height=int(cam.height),
                    fx=fx,
                    fy=fy,
                )
            )

    def _load_point_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        ply_path = self._resolve_ply_path(self.model_path)
        mesh = trimesh.load(str(ply_path), process=False)

        if isinstance(mesh, trimesh.Scene):
            vertices_list = []
            colors_list = []
            for geom in mesh.geometry.values():
                if not hasattr(geom, "vertices"):
                    continue
                vertices_list.append(np.asarray(geom.vertices))
                if hasattr(geom.visual, "vertex_colors") and len(geom.visual.vertex_colors) == len(geom.vertices):
                    colors_list.append(np.asarray(geom.visual.vertex_colors)[:, :3])
                else:
                    colors_list.append(np.full((len(geom.vertices), 3), 200, dtype=np.uint8))
            if not vertices_list:
                raise RuntimeError(f"No vertices found in {ply_path}")
            points = np.concatenate(vertices_list, axis=0)
            colors = np.concatenate(colors_list, axis=0)
        else:
            points = np.asarray(mesh.vertices)
            if hasattr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors) == len(mesh.vertices):
                colors = np.asarray(mesh.visual.vertex_colors)[:, :3]
            else:
                colors = np.full((len(points), 3), 200, dtype=np.uint8)

        if self.max_points > 0 and points.shape[0] > self.max_points:
            ids = np.random.choice(points.shape[0], size=self.max_points, replace=False)
            points = points[ids]
            colors = colors[ids]

        return points.astype(np.float32), colors.astype(np.uint8)

    @staticmethod
    def _project_world_to_screen01(
        point_world: np.ndarray,
        cam_wxyz: np.ndarray,
        cam_pos: np.ndarray,
        cam_fov: float,
        cam_aspect: float,
    ) -> Optional[np.ndarray]:
        T_camera_world = vtf.SE3.from_rotation_and_translation(
            vtf.SO3(cam_wxyz), cam_pos
        ).inverse()
        p_cam_h = T_camera_world.as_matrix() @ np.array([
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

        # Convert to normalized OpenCV screen coords: (0,0)=top-left, (1,1)=bottom-right.
        xy01 = (1.0 + xy) * 0.5
        return np.array([xy01[0], xy01[1], z], dtype=np.float64)

    def _apply_rect_select(
        self,
        client: viser.ClientHandle,
        screen_pos: Tuple[Tuple[float, float], Tuple[float, float]],
        mode: str,
    ) -> int:
        (x0, y0), (x1, y1) = screen_pos
        x_min, x_max = float(min(x0, x1)), float(max(x0, x1))
        y_min, y_max = float(min(y0, y1)), float(max(y0, y1))

        cam = client.camera
        cam_pos = np.asarray(cam.position, dtype=np.float64)
        cam_wxyz = np.asarray(cam.wxyz, dtype=np.float64)
        cam_fov = float(cam.fov)
        cam_aspect = float(cam.aspect)

        hits: Set[int] = set()
        for e in self.camera_entries:
            proj = self._project_world_to_screen01(
                point_world=e.center,
                cam_wxyz=cam_wxyz,
                cam_pos=cam_pos,
                cam_fov=cam_fov,
                cam_aspect=cam_aspect,
            )
            if proj is None:
                continue
            if x_min <= proj[0] <= x_max and y_min <= proj[1] <= y_max:
                hits.add(e.image_id)

        if mode == "replace":
            self.selected_ids = hits
        else:
            self.selected_ids |= hits

        self._refresh_camera_colors()
        return len(hits)

    def _refresh_camera_colors(self):
        for entry in self.camera_entries:
            if entry.handle is None:
                continue
            if entry.image_id in self.selected_ids:
                entry.handle.color = SELECTED_COLOR
                continue
            room_id = self.assignments.get(entry.image_id)
            if room_id is None:
                entry.handle.color = UNASSIGNED_COLOR
            else:
                entry.handle.color = self._room_color(room_id)

    def _room_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for room_id in self.assignments.values():
            counts[room_id] = counts.get(room_id, 0) + 1
        return counts

    def _status_text(self) -> str:
        counts = self._room_counts()
        if not counts:
            room_msg = "none"
        else:
            room_msg = ", ".join([f"room_{k}:{counts[k]}" for k in sorted(counts.keys())])
        return (
            f"Selected: {len(self.selected_ids)} | Assigned: {len(self.assignments)} / {len(self.camera_entries)} | "
            f"Rooms: {room_msg}"
        )

    def _save_outputs(self):
        if not self.assignments:
            raise RuntimeError("No room assignment found. Assign at least one camera before saving.")

        room_to_entries: Dict[int, List[CameraEntry]] = {}
        for entry in self.camera_entries:
            room_id = self.assignments.get(entry.image_id)
            if room_id is None:
                continue
            room_to_entries.setdefault(room_id, []).append(entry)

        if not room_to_entries:
            raise RuntimeError("No valid room assignment found.")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_root": str(self.input_root),
            "sparse_dir": str(self.sparse_dir),
            "model_path": str(self.model_path),
            "num_rooms": len(room_to_entries),
            "num_assigned_images": len(self.assignments),
            "num_total_images": len(self.camera_entries),
            "rooms": [],
            "unassigned_image_ids": sorted(
                [entry.image_id for entry in self.camera_entries if entry.image_id not in self.assignments]
            ),
        }

        images_root = self.input_root / "images"

        for room_id in sorted(room_to_entries.keys()):
            entries = sorted(room_to_entries[room_id], key=lambda x: x.image_id)
            room_dir = self.output_dir / f"room_{room_id}"
            room_dir.mkdir(parents=True, exist_ok=True)

            # 1) images.txt
            (room_dir / "images.txt").write_text("\n".join([e.name for e in entries]) + "\n", encoding="utf-8")

            # 2) metadata.json
            metadata = {
                "room": f"room_{room_id}",
                "room_id": int(room_id),
                "num_images": len(entries),
                "images": [
                    {
                        "image_id": int(e.image_id),
                        "camera_id": int(e.camera_id),
                        "name": e.name,
                        "center": [float(v) for v in e.center.tolist()],
                        "qvec": [float(v) for v in e.qvec.tolist()],
                        "tvec": [float(v) for v in e.tvec.tolist()],
                        "width": int(e.width),
                        "height": int(e.height),
                        "fx": float(e.fx),
                        "fy": float(e.fy),
                    }
                    for e in entries
                ],
            }
            with open(room_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            # 3) copy perspective images
            room_img_root = room_dir / "images"
            room_img_root.mkdir(exist_ok=True)
            copied = 0
            missing = []
            for e in entries:
                src = images_root / e.name
                dst = room_img_root / e.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.exists():
                    shutil.copy2(src, dst)
                    copied += 1
                else:
                    missing.append(e.name)

            # 4) colmap submodel
            export_info = export_room_colmap_submodel(
                room_dir=room_dir,
                input_root=self.input_root,
                room_image_ids=[e.image_id for e in entries],
                images_dict=self.images,
                cameras_dict=self.cameras,
                point_records=self.point_records,
            )

            summary["rooms"].append(
                {
                    "room": f"room_{room_id}",
                    "room_id": int(room_id),
                    "num_images": len(entries),
                    "num_copied_images": int(copied),
                    "missing_images": missing,
                    "colmap_submodel": export_info,
                }
            )

        with open(self.output_dir / "manual_split_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        with open(self.output_dir / "assignment.json", "w", encoding="utf-8") as f:
            json.dump({str(k): int(v) for k, v in sorted(self.assignments.items())}, f, ensure_ascii=False, indent=2)

    def run(self):
        self._load_data()
        points, colors = self._load_point_cloud()

        server = viser.ViserServer(host=self.host, port=self.port)

        self.point_handle = server.add_point_cloud(
            "/scene/point_cloud",
            points=points,
            colors=colors,
            point_size=self.point_size,
        )

        name_collision_count: Dict[str, int] = {}
        for entry in self.camera_entries:
            r_cw = vtf.SO3.from_matrix(self.colmap_loader.qvec2rotmat(entry.qvec))
            r_wc = r_cw.inverse()
            r_vis = r_wc @ vtf.SO3.from_x_radians(np.pi)
            fov = float(2.0 * math.atan((entry.width * 0.5) / max(entry.fx, 1e-6)))
            aspect = float(entry.width / max(entry.height, 1))

            base_name = entry.name.replace("\\", "/")
            name_collision_count[base_name] = name_collision_count.get(base_name, 0) + 1
            scene_name = base_name
            if name_collision_count[base_name] > 1:
                scene_name = f"{base_name}#{entry.image_id}"

            handle = server.add_camera_frustum(
                name=f"/colmap/{scene_name}",
                fov=fov,
                aspect=aspect,
                scale=self.camera_scale,
                wxyz=r_vis.wxyz,
                position=entry.center,
                color=UNASSIGNED_COLOR,
            )
            entry.handle = handle

            @handle.on_click
            def _on_click(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle], image_id=entry.image_id):
                if image_id in self.selected_ids:
                    self.selected_ids.remove(image_id)
                else:
                    self.selected_ids.add(image_id)
                self._refresh_camera_colors()
                selection_status.value = self._status_text()

        with server.add_gui_folder("Display"):
            show_points = server.add_gui_checkbox("Show Point Cloud", initial_value=True)
            show_cameras = server.add_gui_checkbox("Show Cameras", initial_value=True)
            point_size_slider = server.add_gui_slider(
                "Point Size",
                min=0.001,
                max=0.05,
                step=0.001,
                initial_value=self.point_size,
            )

        with server.add_gui_folder("Selection"):
            selection_mode = server.add_gui_button_group("Select Mode", ("replace", "append"))
            drag_select_enabled = server.add_gui_checkbox("Enable Drag Rect Select", initial_value=True)
            drag_select_hint = server.add_gui_markdown(
                "拖拽鼠标在 3D 视图中框选相机。"
                "使用 Select Mode 选择替换/追加模式。"
            )
            clear_selection_button = server.add_gui_button("Clear Selection")

        with server.add_gui_folder("Room Assignment"):
            room_id_slider = server.add_gui_slider("Current Room ID", min=0, max=255, step=1, initial_value=1)
            assign_button = server.add_gui_button("Assign Selection To Room")
            remove_button = server.add_gui_button("Remove Selection From Room")
            select_room_button = server.add_gui_button("Select All Cameras In Current Room")
            clear_room_button = server.add_gui_button("Clear Current Room")
            undo_button = server.add_gui_button("Undo Last Assignment")
            selection_status = server.add_gui_text("Status", initial_value="")
            selection_status.value = self._status_text()

        with server.add_gui_folder("Save"):
            output_text = server.add_gui_text("Output Dir", initial_value=str(self.output_dir))
            save_button = server.add_gui_button("Save Split Result")
            save_log = server.add_gui_text("Save Log", initial_value="")

        @show_points.on_update
        def _(_event):
            if self.point_handle is not None:
                self.point_handle.visible = bool(show_points.value)

        @show_cameras.on_update
        def _(_event):
            visible = bool(show_cameras.value)
            for e in self.camera_entries:
                if e.handle is not None:
                    e.handle.visible = visible

        @point_size_slider.on_update
        def _(_event):
            if self.point_handle is not None:
                self.point_handle.point_size = float(point_size_slider.value)

        @server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            @client.scene.on_pointer_event(event_type="rect-select")
            def _rect_select(event: viser.ScenePointerEvent) -> None:
                if not drag_select_enabled.value:
                    return
                if len(event.screen_pos) < 2:
                    return

                hits = self._apply_rect_select(
                    client=event.client,
                    screen_pos=(event.screen_pos[0], event.screen_pos[1]),
                    mode=selection_mode.value,
                )
                selection_status.value = self._status_text()
                save_log.value = f"Drag-box selected {hits} cameras."

        @clear_selection_button.on_click
        def _(_event):
            self.selected_ids.clear()
            self._refresh_camera_colors()
            selection_status.value = self._status_text()

        @assign_button.on_click
        def _(_event):
            if not self.selected_ids:
                save_log.value = "No selected cameras to assign."
                return
            self.last_assignments = dict(self.assignments)
            room_id = int(room_id_slider.value)
            for image_id in self.selected_ids:
                self.assignments[image_id] = room_id
            self._refresh_camera_colors()
            selection_status.value = self._status_text()
            save_log.value = f"Assigned {len(self.selected_ids)} cameras to room_{room_id}."

        @remove_button.on_click
        def _(_event):
            if not self.selected_ids:
                save_log.value = "No selected cameras to remove."
                return
            self.last_assignments = dict(self.assignments)
            for image_id in list(self.selected_ids):
                self.assignments.pop(image_id, None)
            self._refresh_camera_colors()
            selection_status.value = self._status_text()
            save_log.value = f"Removed assignment for {len(self.selected_ids)} cameras."

        @select_room_button.on_click
        def _(_event):
            room_id = int(room_id_slider.value)
            self.selected_ids = {k for k, v in self.assignments.items() if v == room_id}
            self._refresh_camera_colors()
            selection_status.value = self._status_text()
            save_log.value = f"Selected {len(self.selected_ids)} cameras in room_{room_id}."

        @clear_room_button.on_click
        def _(_event):
            room_id = int(room_id_slider.value)
            self.last_assignments = dict(self.assignments)
            remove_ids = [k for k, v in self.assignments.items() if v == room_id]
            for image_id in remove_ids:
                self.assignments.pop(image_id, None)
            self._refresh_camera_colors()
            selection_status.value = self._status_text()
            save_log.value = f"Cleared room_{room_id}, removed {len(remove_ids)} cameras."

        @undo_button.on_click
        def _(_event):
            if self.last_assignments is None:
                save_log.value = "Nothing to undo."
                return
            self.assignments = self.last_assignments
            self.last_assignments = None
            self._refresh_camera_colors()
            selection_status.value = self._status_text()
            save_log.value = "Undo success."

        @save_button.on_click
        def _(_event):
            try:
                self.output_dir = Path(output_text.value).resolve()
                self._save_outputs()
                save_log.value = f"Saved to {self.output_dir}"
            except Exception as err:
                save_log.value = f"Save failed: {err}"

        print(f"[INFO] loaded cameras: {len(self.camera_entries)}")
        print(f"[INFO] loaded points: {len(points)}")
        print(f"[INFO] server running at {self.host}:{self.port}")

        while True:
            time.sleep(3600)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual room splitting tool for 2DGS big-room scenes.")
    parser.add_argument("--model-path", required=True, help="2DGS model dir or .ply path")
    parser.add_argument("--input", required=True, help="COLMAP dataset root containing sparse/0 and images")
    parser.add_argument("--sparse", default="sparse/0", help="relative sparse folder under --input")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--camera-scale", type=float, default=0.08)
    parser.add_argument("--point-size", type=float, default=0.005)
    parser.add_argument("--max-points", type=int, default=250000)
    return parser


def main():
    args = build_parser().parse_args()
    app = ManualSplitViewer(
        model_path=Path(args.model_path).resolve(),
        input_root=Path(args.input).resolve(),
        sparse_rel=args.sparse,
        output_dir=Path(args.output).resolve(),
        host=args.host,
        port=args.port,
        camera_scale=args.camera_scale,
        point_size=args.point_size,
        max_points=args.max_points,
    )
    app.run()


if __name__ == "__main__":
    main()
