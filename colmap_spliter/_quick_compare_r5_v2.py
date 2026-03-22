#!/usr/bin/env python3
import itertools
import json
from pathlib import Path

root = Path('/nas1/hyh22/backup/PanoGS')
manual = json.loads((root / 'colmap_spliter/output_rooms_d918a/assignment.json').read_text(encoding='utf-8'))
manual = {int(k): int(v) for k, v in manual.items()}

auto = {}
out = root / 'colmap_spliter/output_rooms_d918a_auto_adaptive_r5_v2'
for room_dir in sorted(out.glob('room_[0-9]*')):
    rid = int(room_dir.name.split('_')[-1])
    d = json.loads((room_dir / 'metadata.json').read_text(encoding='utf-8'))
    for it in d['images']:
        auto[int(it['image_id'])] = rid

ids = sorted(set(manual) & set(auto))
rooms_m = sorted(set(manual[i] for i in ids))
rooms_a = sorted(set(auto[i] for i in ids))

best = (0, None)
for perm in itertools.permutations(rooms_m):
    mp = {rooms_a[i]: perm[i] for i in range(len(rooms_a))}
    c = sum(1 for i in ids if manual[i] == mp[auto[i]])
    if c > best[0]:
        best = (c, mp)

print('best_accuracy', best[0] / len(ids), best[0], len(ids))
print('mapping', best[1])
