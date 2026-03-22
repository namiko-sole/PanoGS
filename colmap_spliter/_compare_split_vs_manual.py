#!/usr/bin/env python3
import itertools
import json
from collections import defaultdict
from pathlib import Path


def load_manual_assignment(path: Path):
    d = json.loads(path.read_text(encoding='utf-8'))
    return {int(k): int(v) for k, v in d.items()}


def load_auto_assignment(output_dir: Path):
    assign = {}
    for room_dir in sorted(output_dir.glob('room_*')):
        m = room_dir / 'metadata.json'
        if not m.exists():
            continue
        room_name = room_dir.name
        rid = int(room_name.split('_')[-1])
        data = json.loads(m.read_text(encoding='utf-8'))
        for it in data['images']:
            assign[int(it['image_id'])] = rid
    return assign


def compare(manual, auto, title):
    ids = sorted(set(manual.keys()) & set(auto.keys()))
    manual_rooms = sorted(set(manual[i] for i in ids))
    auto_rooms = sorted(set(auto[i] for i in ids))

    # confusion counts
    conf = defaultdict(int)
    for i in ids:
        conf[(manual[i], auto[i])] += 1

    print(f'=== {title} ===')
    print(f'num_common_images={len(ids)}, manual_rooms={manual_rooms}, auto_rooms={auto_rooms}')

    if len(auto_rooms) == len(manual_rooms) and len(auto_rooms) <= 9:
        best = None
        for perm in itertools.permutations(manual_rooms):
            mp = {auto_rooms[idx]: perm[idx] for idx in range(len(auto_rooms))}
            correct = sum(1 for i in ids if manual[i] == mp[auto[i]])
            if best is None or correct > best[0]:
                best = (correct, mp)
        correct, mp = best
        acc = correct / len(ids)
        print(f'best_label_mapping={mp}')
        print(f'best_mapped_accuracy={acc:.4%} ({correct}/{len(ids)})')
    else:
        # greedy many-to-one mapping
        mp = {}
        for ar in auto_rooms:
            cands = [(conf[(mr, ar)], mr) for mr in manual_rooms]
            cands.sort(reverse=True)
            mp[ar] = cands[0][1]
        correct = sum(1 for i in ids if manual[i] == mp[auto[i]])
        acc = correct / len(ids)
        print(f'greedy_mapping={mp}')
        print(f'greedy_accuracy={acc:.4%} ({correct}/{len(ids)})')

    print('top overlaps (manual, auto, count):')
    top = sorted(((k[0], k[1], v) for k, v in conf.items()), key=lambda x: x[2], reverse=True)[:15]
    for t in top:
        print(' ', t)

    # Auto cluster purity (weighted by cluster size)
    auto_sizes = defaultdict(int)
    auto_major = defaultdict(int)
    for ar in auto_rooms:
        cands = [conf[(mr, ar)] for mr in manual_rooms]
        auto_sizes[ar] = sum(cands)
        auto_major[ar] = max(cands) if cands else 0
    weighted_purity = sum(auto_major[a] for a in auto_rooms) / len(ids)
    print(f'weighted_auto_purity={weighted_purity:.4%}')

    # Manual room fragmentation: how many auto clusters needed to cover 95% images per room
    print('manual_room_fragmentation (num_auto_clusters_for_95%):')
    for mr in manual_rooms:
        overlaps = sorted([conf[(mr, ar)] for ar in auto_rooms], reverse=True)
        total = sum(overlaps)
        accum = 0
        k95 = 0
        for v in overlaps:
            if v <= 0:
                continue
            accum += v
            k95 += 1
            if accum >= 0.95 * total:
                break
        dominant = overlaps[0] / total if total else 0.0
        print(f'  manual_room_{mr}: clusters95={k95}, dominant_ratio={dominant:.4%}, total={total}')
    print()


def main():
    root = Path('/nas1/hyh22/backup/PanoGS')
    manual = load_manual_assignment(root / 'colmap_spliter/output_rooms_d918a/assignment.json')

    auto_eval = load_auto_assignment(root / 'colmap_spliter/output_rooms_d918a_auto_eval')
    compare(manual, auto_eval, 'AUTO tuned (30,0.015,20,20)')

    auto_default = load_auto_assignment(root / 'colmap_spliter/output_rooms_d918a_auto_default')
    compare(manual, auto_default, 'AUTO default')


if __name__ == '__main__':
    main()
