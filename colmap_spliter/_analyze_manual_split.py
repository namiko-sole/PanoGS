#!/usr/bin/env python3
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path('/nas1/hyh22/backup/PanoGS')
    input_root = root / 'data_big_room_d918a_2dgs'
    out = root / 'colmap_spliter' / 'output_rooms_d918a'

    assignment = json.loads((out / 'assignment.json').read_text(encoding='utf-8'))
    assignment = {int(k): int(v) for k, v in assignment.items()}
    rooms = sorted(set(assignment.values()))

    scene_dir = root / '2d_gaussian_splatting' / 'scene'
    sys.path.insert(0, str(scene_dir))
    import colmap_loader  # type: ignore

    sys.path.insert(0, str(root / 'colmap_spliter'))
    from split_rooms import read_points3d_binary_full, camera_center_from_image

    images = colmap_loader.read_extrinsics_binary(str(input_root / 'sparse/0/images.bin'))
    points = read_points3d_binary_full(input_root / 'sparse/0/points3D.bin')

    room_images = defaultdict(list)
    for img_id, rid in assignment.items():
        room_images[rid].append(img_id)
    for rid in rooms:
        room_images[rid].sort()

    room_centers = {}
    for rid in rooms:
        centers = []
        for img_id in room_images[rid]:
            centers.append(camera_center_from_image(colmap_loader, images[img_id]))
        room_centers[rid] = np.stack(centers)

    print('=== Rooms Basic ===')
    for rid in rooms:
        ids = room_images[rid]
        arr = room_centers[rid]
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        size = maxs - mins
        runs = 1
        for a, b in zip(ids[:-1], ids[1:]):
            if b != a + 1:
                runs += 1
        print(
            f'room_{rid}: images={len(ids)}, id_range=[{ids[0]},{ids[-1]}], '
            f'runs={runs}, bbox_size=({size[0]:.2f},{size[1]:.2f},{size[2]:.2f})'
        )

    room_point_sets = {rid: set() for rid in rooms}
    for rid in rooms:
        s = room_point_sets[rid]
        for img_id in room_images[rid]:
            pids = images[img_id].point3D_ids
            valid = pids[pids >= 0]
            s.update(int(v) for v in valid.tolist())

    print('\n=== Pairwise Room Point Overlap (IoU / overlap wrt smaller) ===')
    for i, r1 in enumerate(rooms):
        for r2 in rooms[i + 1 :]:
            a = room_point_sets[r1]
            b = room_point_sets[r2]
            inter = len(a & b)
            uni = len(a | b)
            iou = inter / uni if uni else 0.0
            ov_small = inter / min(len(a), len(b)) if min(len(a), len(b)) else 0.0
            if inter > 0:
                print(
                    f'room_{r1}-room_{r2}: inter={inter}, IoU={iou:.4f}, '
                    f'overlap_small={ov_small:.4f}'
                )

    single = 0
    multi = 0
    room_combo_counter = Counter()
    pair_counter = Counter()
    for _pid, rec in points.items():
        rs = sorted(
            {assignment.get(int(img_id)) for img_id, _ in rec['track'] if int(img_id) in assignment}
        )
        rs = [r for r in rs if r is not None]
        if not rs:
            continue
        if len(rs) == 1:
            single += 1
        else:
            multi += 1
            room_combo_counter[tuple(rs)] += 1
            for i in range(len(rs)):
                for j in range(i + 1, len(rs)):
                    pair_counter[(rs[i], rs[j])] += 1

    print('\n=== Point Track Purity ===')
    total = single + multi
    print(
        f'total_points_seen={total}, single_room={single} ({single/total:.2%}), '
        f'multi_room={multi} ({multi/total:.2%})'
    )
    print('Top multi-room combos:')
    for combo, c in room_combo_counter.most_common(10):
        print(f'  rooms={combo}: {c}')
    print('Top boundary pairs by shared tracks:')
    for pair, c in pair_counter.most_common(10):
        print(f'  pair={pair}: {c}')

    point_owner = {}
    for pid, rec in points.items():
        cnt = Counter()
        for img_id, _ in rec['track']:
            rid = assignment.get(int(img_id))
            if rid is not None:
                cnt[rid] += 1
        if cnt:
            point_owner[pid] = cnt.most_common(1)[0][0]

    ambiguous_cams = []
    for img_id, rid in assignment.items():
        cnt = Counter()
        pids = images[img_id].point3D_ids
        for pid in pids[pids >= 0]:
            own = point_owner.get(int(pid))
            if own is not None:
                cnt[own] += 1
        if not cnt:
            continue
        top_room, top_n = cnt.most_common(1)[0]
        total_n = sum(cnt.values())
        purity = top_n / total_n
        if rid != top_room or purity < 0.65:
            ambiguous_cams.append((img_id, rid, top_room, purity, total_n))

    ambiguous_cams.sort(key=lambda x: x[3])
    print('\n=== Potential Boundary Cameras ===')
    print(f'count={len(ambiguous_cams)} / {len(assignment)}')
    for row in ambiguous_cams[:20]:
        print(
            f'  img={row[0]} assign={row[1]} dominant={row[2]} '
            f'purity={row[3]:.3f} pts={row[4]}'
        )


if __name__ == '__main__':
    main()
