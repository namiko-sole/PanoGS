#!/usr/bin/env python3
import argparse
import json
import math
import struct
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


def _load_colmap_readers(repo_root: Path):
    scene_dir = repo_root / "2d_gaussian_splatting" / "scene"
    if not scene_dir.exists():
        raise FileNotFoundError(f"Cannot find COLMAP loader directory: {scene_dir}")
    sys.path.insert(0, str(scene_dir))
    import colmap_loader  # type: ignore

    return colmap_loader


def _read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3d_binary_full(path_to_model_file: Path):
    points = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = _read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3d_id = int(props[0])
            xyz = np.array(props[1:4], dtype=np.float64)
            rgb = np.array(props[4:7], dtype=np.uint8)
            error = float(props[7])
            track_length = _read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = _read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            track = []
            for i in range(track_length):
                img_id = int(track_elems[2 * i])
                p2d_idx = int(track_elems[2 * i + 1])
                track.append((img_id, p2d_idx))
            points[point3d_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track,
            }
    return points


def camera_center_from_image(colmap_loader, image_obj):
    # COLMAP stores world->camera as R,t. Camera center in world: C = -R^T t
    r = colmap_loader.qvec2rotmat(image_obj.qvec)
    t = image_obj.tvec.reshape(3)
    c = -r.T @ t
    return c.astype(np.float64)


def build_point_sets(images_dict):
    image_ids = sorted(images_dict.keys())
    point_sets = {}
    for image_id in image_ids:
        pids = images_dict[image_id].point3D_ids
        valid = pids[pids >= 0]
        point_sets[image_id] = set(valid.tolist())
    return image_ids, point_sets


def pairwise_similarity(image_ids, point_sets, centers, min_shared_points=80):
    n = len(image_ids)
    sim = np.zeros((n, n), dtype=np.float32)
    shared = np.zeros((n, n), dtype=np.int32)

    for i in range(n):
        sim[i, i] = 1.0
        shared[i, i] = len(point_sets[image_ids[i]])
        for j in range(i + 1, n):
            a = point_sets[image_ids[i]]
            b = point_sets[image_ids[j]]
            if not a or not b:
                continue
            inter = len(a & b)
            if inter < min_shared_points:
                continue
            uni = len(a | b)
            if uni == 0:
                continue
            iou = inter / float(uni)

            # Mild geometric prior to avoid connecting remote rooms by accidental overlap.
            d = np.linalg.norm(centers[i] - centers[j])
            geom = math.exp(-d / 8.0)
            s = float(iou * (0.7 + 0.3 * geom))

            sim[i, j] = s
            sim[j, i] = s
            shared[i, j] = inter
            shared[j, i] = inter
    return sim, shared


def estimate_adaptive_thresholds(
    shared,
    sim,
    shared_quantile=0.70,
    sim_quantile=0.85,
    shared_floor=5,
    shared_cap=200,
):
    tri = np.triu_indices(shared.shape[0], k=1)
    shared_vals = shared[tri]
    sim_vals = sim[tri]

    shared_pos = shared_vals[shared_vals > 0]
    sim_pos = sim_vals[sim_vals > 0]

    if shared_pos.size == 0:
        min_shared = int(shared_floor)
    else:
        min_shared = int(np.quantile(shared_pos, np.clip(shared_quantile, 0.0, 1.0)))
        min_shared = int(np.clip(min_shared, shared_floor, shared_cap))

    if sim_pos.size == 0:
        edge_threshold = 0.01
    else:
        edge_threshold = float(np.quantile(sim_pos, np.clip(sim_quantile, 0.0, 1.0)))
        edge_threshold = float(np.clip(edge_threshold, 1e-4, 0.20))

    return min_shared, edge_threshold


def build_adjacency(sim, shared, k=8, edge_threshold=0.04, min_shared_points=80):
    n = sim.shape[0]
    adjacency = [set() for _ in range(n)]

    for i in range(n):
        scores = sim[i].copy()
        scores[i] = 0.0
        if np.all(scores <= 0):
            continue
        nn_idx = np.argsort(scores)[-k:]
        for j in nn_idx:
            if j == i:
                continue
            if shared[i, j] >= min_shared_points and scores[j] >= edge_threshold:
                adjacency[i].add(int(j))
                adjacency[j].add(int(i))

    return adjacency


def connected_components(adjacency):
    n = len(adjacency)
    visited = np.zeros(n, dtype=bool)
    comps = []

    for i in range(n):
        if visited[i]:
            continue
        q = deque([i])
        visited[i] = True
        comp = []
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adjacency[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
        comps.append(sorted(comp))

    return comps


def merge_small_components(components, centers, min_room_size=5):
    if not components:
        return components

    large = [c for c in components if len(c) >= min_room_size]
    small = [c for c in components if len(c) < min_room_size]

    if not large:
        return components

    large_centers = [np.mean(centers[c], axis=0) for c in large]

    for s in small:
        s_center = np.mean(centers[s], axis=0)
        dists = [np.linalg.norm(s_center - lc) for lc in large_centers]
        tgt = int(np.argmin(dists))
        large[tgt].extend(s)
        large[tgt] = sorted(set(large[tgt]))
        large_centers[tgt] = np.mean(centers[large[tgt]], axis=0)

    return [sorted(c) for c in large]


def _component_point_sets(components, image_ids, point_sets):
    comp_points = []
    for comp in components:
        s = set()
        for cam_local_idx in comp:
            s.update(point_sets[image_ids[cam_local_idx]])
        comp_points.append(s)
    return comp_points


def _component_centers(components, centers):
    return [np.mean(centers[c], axis=0) for c in components]


def merge_components_two_stage(
    components,
    image_ids,
    point_sets,
    centers,
    sim,
    adjacency,
    target_rooms=0,
    merge_score_threshold=0.03,
    max_iter=4096,
):
    comps = [sorted(set(c)) for c in components]
    merge_history = []

    for _ in range(max_iter):
        if len(comps) <= 1:
            break
        if target_rooms > 0 and len(comps) <= target_rooms:
            break

        cam_to_comp = {}
        for ci, comp in enumerate(comps):
            for cam_idx in comp:
                cam_to_comp[cam_idx] = ci

        pair_sims = defaultdict(list)
        for u in range(len(adjacency)):
            cu = cam_to_comp.get(u)
            if cu is None:
                continue
            for v in adjacency[u]:
                if v <= u:
                    continue
                cv = cam_to_comp.get(v)
                if cv is None or cu == cv:
                    continue
                a, b = (cu, cv) if cu < cv else (cv, cu)
                pair_sims[(a, b)].append(float(sim[u, v]))

        comp_points = _component_point_sets(comps, image_ids, point_sets)
        comp_centers = _component_centers(comps, centers)

        candidate_pairs = set(pair_sims.keys())
        if target_rooms > 0:
            # In target-room mode, allow disconnected components to be merged
            # using global camera similarity as a fallback.
            for a in range(len(comps)):
                for b in range(a + 1, len(comps)):
                    if (a, b) in candidate_pairs:
                        continue
                    vals = sim[np.ix_(comps[a], comps[b])]
                    vmax = float(np.max(vals)) if vals.size > 0 else 0.0
                    if vmax > 0.0:
                        pair_sims[(a, b)] = [vmax]
                        candidate_pairs.add((a, b))

        if not candidate_pairs:
            break

        # Adaptive distance scale from current component layout.
        dvals = []
        for i in range(len(comp_centers)):
            for j in range(i + 1, len(comp_centers)):
                dvals.append(float(np.linalg.norm(comp_centers[i] - comp_centers[j])))
        dist_scale = float(np.median(dvals)) if dvals else 1.0
        dist_scale = max(1e-4, dist_scale)

        best = None
        for a, b in sorted(candidate_pairs):
            edge_vals = pair_sims.get((a, b), [])
            pa = comp_points[a]
            pb = comp_points[b]
            if not pa or not pb:
                continue

            inter = len(pa & pb)
            overlap_small = inter / float(max(1, min(len(pa), len(pb))))
            edge_strength = float(np.mean(edge_vals)) if edge_vals else 0.0
            d = float(np.linalg.norm(comp_centers[a] - comp_centers[b]))
            geo = math.exp(-d / dist_scale)

            score = 0.55 * overlap_small + 0.35 * edge_strength + 0.10 * geo
            if best is None or score > best[0]:
                best = (score, a, b, overlap_small, edge_strength, geo, inter)

        if best is None:
            break

        score, a, b, overlap_small, edge_strength, geo, inter = best
        if target_rooms <= 0 and score < merge_score_threshold:
            break

        merged = sorted(set(comps[a]) | set(comps[b]))
        if a < b:
            first, second = a, b
        else:
            first, second = b, a
        comps[first] = merged
        comps.pop(second)

        merge_history.append({
            "score": float(score),
            "overlap_small": float(overlap_small),
            "edge_strength": float(edge_strength),
            "geo": float(geo),
            "shared_points": int(inter),
            "num_components_after_merge": int(len(comps)),
        })

    return comps, merge_history


def assign_points_to_rooms(image_ids, point_sets, components):
    room_point_votes = defaultdict(lambda: defaultdict(int))

    for room_idx, comp in enumerate(components):
        for cam_local_idx in comp:
            img_id = image_ids[cam_local_idx]
            for pid in point_sets[img_id]:
                room_point_votes[pid][room_idx] += 1

    point_room = {}
    for pid, votes in room_point_votes.items():
        room_idx = max(votes.items(), key=lambda kv: kv[1])[0]
        point_room[int(pid)] = int(room_idx)

    return point_room


def _write_cameras_txt(path: Path, cameras_dict, camera_ids):
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(camera_ids)}",
    ]
    for cid in sorted(camera_ids):
        cam = cameras_dict[cid]
        params_str = " ".join(str(float(v)) for v in cam.params.tolist())
        lines.append(f"{int(cam.id)} {cam.model} {int(cam.width)} {int(cam.height)} {params_str}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_images_txt(path: Path, room_image_ids, images_dict, allowed_point_ids):
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(room_image_ids)}",
    ]
    for img_id in sorted(room_image_ids):
        img = images_dict[img_id]
        q = img.qvec
        t = img.tvec
        lines.append(
            f"{int(img.id)} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {int(img.camera_id)} {img.name}"
        )

        triples = []
        for idx in range(len(img.point3D_ids)):
            x = float(img.xys[idx][0])
            y = float(img.xys[idx][1])
            pid = int(img.point3D_ids[idx])
            if pid not in allowed_point_ids:
                pid = -1
            triples.append(f"{x} {y} {pid}")
        lines.append(" ".join(triples))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_points3d_txt(path: Path, point_ids, point_records, room_image_id_set):
    lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        f"# Number of points: {len(point_ids)}",
    ]
    for pid in sorted(point_ids):
        p = point_records.get(pid)
        if p is None:
            continue
        xyz = p["xyz"]
        rgb = p["rgb"]
        err = p["error"]
        track = [tp for tp in p["track"] if tp[0] in room_image_id_set]
        if len(track) < 2:
            continue
        track_str = " ".join(f"{int(img_id)} {int(p2d_idx)}" for img_id, p2d_idx in track)
        lines.append(
            f"{int(pid)} {xyz[0]} {xyz[1]} {xyz[2]} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])} {err} {track_str}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_room_colmap_submodel(
    room_dir: Path,
    input_root: Path,
    room_image_ids,
    images_dict,
    cameras_dict,
    point_records,
):
    room_image_ids = sorted(room_image_ids)
    room_image_set = set(room_image_ids)

    room_camera_ids = sorted({int(images_dict[i].camera_id) for i in room_image_ids})
    room_point_ids = set()
    for img_id in room_image_ids:
        pids = images_dict[img_id].point3D_ids
        valid = pids[pids >= 0]
        room_point_ids.update(int(v) for v in valid.tolist())

    # Remove points with less than 2 tracks inside this room to keep valid COLMAP structure.
    filtered_point_ids = set()
    for pid in room_point_ids:
        p = point_records.get(pid)
        if p is None:
            continue
        track = [tp for tp in p["track"] if tp[0] in room_image_set]
        if len(track) >= 2:
            filtered_point_ids.add(pid)

    submodel_root = room_dir / "colmap_submodel"
    sparse0 = submodel_root / "sparse" / "0"
    sparse0.mkdir(parents=True, exist_ok=True)

    images_src = input_root / "images"
    images_dst = submodel_root / "images"
    if not images_dst.exists():
        try:
            images_dst.symlink_to(images_src)
        except Exception:
            # Fallback: keep a text pointer if symlink is unavailable.
            (submodel_root / "IMAGES_SOURCE.txt").write_text(str(images_src) + "\n", encoding="utf-8")

    _write_cameras_txt(sparse0 / "cameras.txt", cameras_dict, room_camera_ids)
    _write_images_txt(sparse0 / "images.txt", room_image_ids, images_dict, filtered_point_ids)
    _write_points3d_txt(sparse0 / "points3D.txt", filtered_point_ids, point_records, room_image_set)

    return {
        "submodel_root": str(submodel_root),
        "num_cameras": len(room_camera_ids),
        "num_images": len(room_image_ids),
        "num_points": len(filtered_point_ids),
    }


def save_results(
    output_dir: Path,
    input_root: Path,
    image_ids,
    images_dict,
    cameras_dict,
    point_records,
    centers,
    components,
    point_room,
    sim,
    adjacency,
    split_config=None,
    merge_history=None,
    boundary_purity_threshold=0.65,
    export_submodels=True,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort rooms by size desc while keeping mapping from old component index to new room index.
    comp_with_old = [(idx, sorted(c)) for idx, c in enumerate(components)]
    comp_with_old.sort(key=lambda x: len(x[1]), reverse=True)
    components = [c for _, c in comp_with_old]
    old_to_new_room = {old_idx: new_idx for new_idx, (old_idx, _) in enumerate(comp_with_old)}

    camera_room_map = {}
    for ridx0, comp in enumerate(components):
        for local_cam_idx in comp:
            camera_room_map[int(local_cam_idx)] = int(ridx0 + 1)

    remapped_point_room = {}
    for pid, old_room in point_room.items():
        if old_room in old_to_new_room:
            remapped_point_room[pid] = int(old_to_new_room[old_room] + 1)

    summary = {
        "num_rooms": len(components),
        "num_images": len(image_ids),
        "num_point_assignments": len(remapped_point_room),
        "rooms": [],
    }

    if split_config is not None:
        summary["split_config"] = split_config
    if merge_history is not None:
        summary["merge_history"] = merge_history

    for ridx, comp in enumerate(components, start=1):
        room_name = f"room_{ridx}"
        room_dir = output_dir / room_name
        room_dir.mkdir(exist_ok=True)

        image_lines = []
        room_images = []
        for local_cam_idx in comp:
            img_id = image_ids[local_cam_idx]
            img = images_dict[img_id]
            c = centers[local_cam_idx]
            image_lines.append(img.name)
            room_images.append({
                "image_id": int(img_id),
                "name": img.name,
                "camera_id": int(img.camera_id),
                "center": [float(c[0]), float(c[1]), float(c[2])],
            })

        (room_dir / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
        with open(room_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({
                "room": room_name,
                "num_images": len(room_images),
                "images": room_images,
            }, f, ensure_ascii=False, indent=2)

        summary["rooms"].append({
            "room": room_name,
            "num_images": len(room_images),
            "images_file": str((room_dir / "images.txt").name),
            "metadata_file": str((room_dir / "metadata.json").name),
        })

        if export_submodels:
            export_info = export_room_colmap_submodel(
                room_dir=room_dir,
                input_root=input_root,
                room_image_ids=[image_ids[idx] for idx in comp],
                images_dict=images_dict,
                cameras_dict=cameras_dict,
                point_records=point_records,
            )
            summary["rooms"][-1]["colmap_submodel"] = export_info

    with open(output_dir / "room_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Save debug artifacts.
    np.save(output_dir / "camera_similarity.npy", sim)
    np.save(output_dir / "camera_centers.npy", centers)

    edge_list = []
    for i, nbrs in enumerate(adjacency):
        for j in sorted(nbrs):
            if i < j:
                edge_list.append([int(i), int(j)])
    with open(output_dir / "camera_graph_edges.json", "w", encoding="utf-8") as f:
        json.dump(edge_list, f, ensure_ascii=False, indent=2)

    with open(output_dir / "point_room_assignment.json", "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in remapped_point_room.items()}, f, ensure_ascii=False)

    # Camera-level confidence and boundary report based on point-room consistency.
    camera_conf = []
    boundary_cameras = []
    for local_cam_idx, img_id in enumerate(image_ids):
        assigned_room = camera_room_map.get(local_cam_idx)
        if assigned_room is None:
            continue
        pids = images_dict[img_id].point3D_ids
        valid = pids[pids >= 0]
        counts = defaultdict(int)
        for pid in valid.tolist():
            rid = remapped_point_room.get(int(pid))
            if rid is not None:
                counts[rid] += 1
        total = int(sum(counts.values()))
        if total <= 0:
            purity = 0.0
            dominant_room = assigned_room
        else:
            dominant_room = max(counts.items(), key=lambda kv: kv[1])[0]
            purity = float(counts.get(assigned_room, 0)) / float(total)

        rec = {
            "image_id": int(img_id),
            "name": images_dict[img_id].name,
            "assigned_room": int(assigned_room),
            "dominant_room": int(dominant_room),
            "purity": float(purity),
            "num_counted_points": int(total),
        }
        camera_conf.append(rec)
        if purity < boundary_purity_threshold or dominant_room != assigned_room:
            boundary_cameras.append(rec)

    camera_conf.sort(key=lambda x: x["image_id"])
    boundary_cameras.sort(key=lambda x: (x["purity"], x["image_id"]))
    with open(output_dir / "camera_room_confidence.json", "w", encoding="utf-8") as f:
        json.dump(camera_conf, f, ensure_ascii=False, indent=2)
    with open(output_dir / "boundary_cameras.json", "w", encoding="utf-8") as f:
        json.dump(boundary_cameras, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Split multi-room COLMAP scene into room clusters")
    parser.add_argument("--input", required=True, help="Dataset root containing sparse/0")
    parser.add_argument("--sparse", default="sparse/0", help="Relative sparse model directory")
    parser.add_argument("--output", required=True, help="Output directory for room split result")
    parser.add_argument("--min-shared-points", type=int, default=80, help="Minimum shared 3D points to consider an edge")
    parser.add_argument("--edge-threshold", type=float, default=0.04, help="Similarity threshold for camera graph edges")
    parser.add_argument("--knn", type=int, default=8, help="KNN neighbors in camera graph")
    parser.add_argument("--min-room-size", type=int, default=5, help="Minimum images per room before merging")
    parser.add_argument("--auto-adaptive", action="store_true", help="Enable adaptive two-stage splitting")
    parser.add_argument("--shared-quantile", type=float, default=0.70, help="Quantile for adaptive min shared points")
    parser.add_argument("--sim-quantile", type=float, default=0.85, help="Quantile for adaptive similarity threshold")
    parser.add_argument("--shared-floor", type=int, default=5, help="Lower bound for adaptive min shared points")
    parser.add_argument("--shared-cap", type=int, default=200, help="Upper bound for adaptive min shared points")
    parser.add_argument("--target-rooms", type=int, default=0, help="Optional target room count for stage-2 merging (0 disables)")
    parser.add_argument("--merge-score-threshold", type=float, default=0.03, help="Stop merging when best score falls below this (if target-rooms=0)")
    parser.add_argument("--boundary-purity-threshold", type=float, default=0.65, help="Boundary camera purity threshold")
    parser.add_argument("--no-export-submodels", action="store_true", help="Disable per-room COLMAP submodel export")
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    sparse_dir = (input_root / args.sparse).resolve()
    output_dir = Path(args.output).resolve()

    repo_root = Path(__file__).resolve().parents[1]
    colmap_loader = _load_colmap_readers(repo_root)

    images_bin = sparse_dir / "images.bin"
    cameras_bin = sparse_dir / "cameras.bin"
    points3d_bin = sparse_dir / "points3D.bin"
    if not images_bin.exists():
        raise FileNotFoundError(f"Missing images.bin: {images_bin}")
    if not cameras_bin.exists():
        raise FileNotFoundError(f"Missing cameras.bin: {cameras_bin}")
    if not points3d_bin.exists():
        raise FileNotFoundError(f"Missing points3D.bin: {points3d_bin}")

    images = colmap_loader.read_extrinsics_binary(str(images_bin))
    cameras = colmap_loader.read_intrinsics_binary(str(cameras_bin))
    point_records = read_points3d_binary_full(points3d_bin)
    if len(images) == 0:
        raise RuntimeError("No registered images found in COLMAP model.")

    image_ids, point_sets = build_point_sets(images)
    centers = np.stack([camera_center_from_image(colmap_loader, images[i]) for i in image_ids], axis=0)

    sim, shared = pairwise_similarity(
        image_ids,
        point_sets,
        centers,
        min_shared_points=1,
    )
    min_shared_points = max(1, args.min_shared_points)
    edge_threshold = max(0.0, args.edge_threshold)
    knn = max(1, args.knn)

    split_config = {
        "auto_adaptive": bool(args.auto_adaptive),
    }

    if args.auto_adaptive:
        min_shared_points, edge_threshold = estimate_adaptive_thresholds(
            shared=shared,
            sim=sim,
            shared_quantile=args.shared_quantile,
            sim_quantile=args.sim_quantile,
            shared_floor=max(1, args.shared_floor),
            shared_cap=max(args.shared_floor, args.shared_cap),
        )
        knn = max(knn, min(32, max(12, int(round(math.sqrt(len(image_ids)))))))
        split_config.update({
            "shared_quantile": float(args.shared_quantile),
            "sim_quantile": float(args.sim_quantile),
            "shared_floor": int(args.shared_floor),
            "shared_cap": int(args.shared_cap),
            "adapted_min_shared_points": int(min_shared_points),
            "adapted_edge_threshold": float(edge_threshold),
            "adapted_knn": int(knn),
            "target_rooms": int(args.target_rooms),
            "merge_score_threshold": float(args.merge_score_threshold),
        })
    else:
        split_config.update({
            "min_shared_points": int(min_shared_points),
            "edge_threshold": float(edge_threshold),
            "knn": int(knn),
            "target_rooms": int(args.target_rooms),
            "merge_score_threshold": float(args.merge_score_threshold),
        })

    adjacency = build_adjacency(
        sim,
        shared,
        k=knn,
        edge_threshold=edge_threshold,
        min_shared_points=min_shared_points,
    )
    components = connected_components(adjacency)
    components = merge_small_components(components, centers, min_room_size=max(1, args.min_room_size))

    merge_history = []
    if args.auto_adaptive or args.target_rooms > 0:
        components, merge_history = merge_components_two_stage(
            components=components,
            image_ids=image_ids,
            point_sets=point_sets,
            centers=centers,
            sim=sim,
            adjacency=adjacency,
            target_rooms=max(0, args.target_rooms),
            merge_score_threshold=max(0.0, args.merge_score_threshold),
        )

    point_room = assign_points_to_rooms(image_ids, point_sets, components)
    save_results(
        output_dir=output_dir,
        input_root=input_root,
        image_ids=image_ids,
        images_dict=images,
        cameras_dict=cameras,
        point_records=point_records,
        centers=centers,
        components=components,
        point_room=point_room,
        sim=sim,
        adjacency=adjacency,
        split_config=split_config,
        merge_history=merge_history,
        boundary_purity_threshold=float(np.clip(args.boundary_purity_threshold, 0.0, 1.0)),
        export_submodels=not args.no_export_submodels,
    )

    print(f"[OK] cameras: {len(image_ids)}")
    print(f"[OK] rooms: {len(components)}")
    print(f"[OK] output: {output_dir}")
    for idx, comp in enumerate(sorted(components, key=lambda c: len(c), reverse=True), start=1):
        print(f"  - room_{idx}: {len(comp)} images")

if __name__ == "__main__":
    main()
