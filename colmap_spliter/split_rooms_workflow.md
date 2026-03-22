# split_rooms.py 工作原理说明

本文档详细说明 `split_rooms.py` 的设计目标、数据流、核心算法、导出逻辑、参数调优与常见问题，便于后续维护与扩展。

## 1. 脚本目标

`split_rooms.py` 用于将一个多房间 COLMAP 稀疏重建场景自动切分为 `room_1`, `room_2`, ...，并可选导出每个房间的独立 COLMAP 子模型，便于后续“按房间单独训练”。

输入是一个标准 COLMAP 数据目录（默认包含 `sparse/0` 和 `images`）。

输出包含两部分：

1. 房间聚类结果
- 每个房间的图像清单和元数据
- 全局相机相似度、图边等调试文件

2. 子模型导出（默认开启）
- `room_i/colmap_submodel/sparse/0/cameras.txt`
- `room_i/colmap_submodel/sparse/0/images.txt`
- `room_i/colmap_submodel/sparse/0/points3D.txt`
- `room_i/colmap_submodel/images`（指向原始 `images` 的软链接）

---

## 2. 处理流程总览

脚本主流程在 `main()` 中，按如下顺序执行：

1. 读取输入参数
2. 加载 COLMAP 读取器（复用工程内 `2d_gaussian_splatting/scene/colmap_loader.py`）
3. 读取 COLMAP 二进制模型：
- `images.bin`
- `cameras.bin`
- `points3D.bin`
4. 计算每个图像可见的 3D 点集合
5. 构建相机两两相似度矩阵（点集 IoU + 轻量几何先验）
6. 基于 KNN + 阈值构建相机图
7. 对相机图做连通域分解得到初始房间簇
8. 将过小簇并入最近大簇
9. （可选）二阶段自动合并：按共享点比例 + 图连边强度 + 几何接近度合并簇
10. 统计点属于哪个房间（投票）
11. 写出结果文件（含置信度与边界相机）
12. 若开启导出，写出每个房间 COLMAP 子模型

---

## 3. 核心模块说明

## 3.1 模型读取

### `_load_colmap_readers(repo_root)`
动态导入工程内 `colmap_loader`，避免重复实现 `images.bin`、`cameras.bin` 的解析。

### `read_points3d_binary_full(path_to_model_file)`
自定义解析 `points3D.bin`，因为子模型导出需要完整 track 信息。

每个 3D 点记录字段：
- `xyz`
- `rgb`
- `error`
- `track = [(image_id, point2D_idx), ...]`

该函数比仅返回 xyz/rgb 的轻量读取更完整，是子模型导出的关键依赖。

---

## 3.2 相机表示与可见性

### `camera_center_from_image(colmap_loader, image_obj)`
将 COLMAP 外参转换为世界系相机中心：

- COLMAP 给的是 world->camera 的 `R, t`
- 相机中心为：`C = -R^T t`

### `build_point_sets(images_dict)`
为每个图像提取其有效 3D 点 ID 集合（过滤 `-1`）。

输出：
- `image_ids`（排序后的 image_id 列表）
- `point_sets[image_id] = set(point3d_id, ...)`

---

## 3.3 相机相似度建图

### `pairwise_similarity(image_ids, point_sets, centers, min_shared_points)`
对每对相机 `(i, j)` 计算：

1. 交并比
- `inter = |Pi ∩ Pj|`
- `union = |Pi ∪ Pj|`
- `iou = inter / union`

2. 几何先验（抑制远距离误连接）
- `d = ||Ci - Cj||`
- `geom = exp(-d / 8.0)`

3. 最终相似度
- `s = iou * (0.7 + 0.3 * geom)`

并且要求 `inter >= min_shared_points` 才允许连边。

该函数输出：
- `sim`：相似度矩阵
- `shared`：共享点数矩阵（当前用于调试）

### `estimate_adaptive_thresholds(shared, sim, ...)`
根据当前场景统计分布自动估计：

- `min_shared_points`
- `edge_threshold`

避免固定参数在新场景失效。

### `build_adjacency(sim, shared, k, edge_threshold, min_shared_points)`
构建无向相机图：

- 每个相机仅保留 top-k 邻居候选
- 仅连接同时满足以下条件的边：
  - `shared >= min_shared_points`
  - `sim >= edge_threshold`

作用：
- 抑制全连接图噪声
- 保持图稀疏，便于稳定分簇

---

## 3.4 房间聚类

### `connected_components(adjacency)`
在相机图上做连通域分解，每个连通块视作一个房间候选簇。

### `merge_small_components(components, centers, min_room_size)`
将小簇（图像数 < `min_room-size`）并入最近的大簇，避免碎片化房间。

并入规则：
- 计算簇中心（该簇相机中心均值）
- 小簇并到最近大簇

### `merge_components_two_stage(...)`
这是“尽可能自动化”的核心步骤：

1. 第一阶段先过分割（保证簇纯度）
2. 第二阶段按合并分数自动合并，分数由以下项加权：
  - `overlap_small`：两簇点集交集占较小簇点数比例
  - `edge_strength`：跨簇相机图连边相似度
  - `geo`：簇中心距离衰减项

当 `--target-rooms > 0` 时，会继续合并到目标房间数；
否则在分数低于 `--merge-score-threshold` 时停止。

---

## 3.5 点到房间分配

### `assign_points_to_rooms(image_ids, point_sets, components)`
基于“可见图像投票”给每个点分房间：

- 某点被某房间内多少张相机看到，就给该房间多少票
- 点归属给票数最多的房间

用途：
- 用于统计与调试（`point_room_assignment.json`）
- 可为后续房间内点云处理提供先验

---

## 4. 子模型导出逻辑

导出在 `export_room_colmap_submodel(...)` 中完成。

## 4.1 导出目标

每个房间输出一个独立 COLMAP 文本模型（text format）：

- `cameras.txt`
- `images.txt`
- `points3D.txt`

并创建 `images` 软链接到原始图像目录。

## 4.2 关键过滤策略

为了保持模型一致性，脚本做了两层过滤：

1. 点必须属于该房间图像可见集合
2. 点在该房间内 track 长度必须 >= 2

原因：
- COLMAP 的 3D 点若在子模型里只剩单观测，几何意义弱且易引入噪声。

## 4.3 文件写入函数

### `_write_cameras_txt(...)`
输出该房间涉及到的内参集合（通常一个数据集可能只有 1 个 camera_id）。

### `_write_images_txt(...)`
输出该房间图像位姿与二维观测。

- 若某个二维观测点对应的 3D 点不在本房间允许集合中，则将该观测的 `POINT3D_ID` 置为 `-1`。

### `_write_points3d_txt(...)`
输出该房间保留的 3D 点及其 track（只保留房间内图像上的观测索引）。

---

## 5. 输出目录结构

以 `--output output_xxx` 为例：

- `output_xxx/room_split_summary.json`
- `output_xxx/camera_similarity.npy`
- `output_xxx/camera_centers.npy`
- `output_xxx/camera_graph_edges.json`
- `output_xxx/point_room_assignment.json`
- `output_xxx/camera_room_confidence.json`
- `output_xxx/boundary_cameras.json`
- `output_xxx/room_1/`
  - `images.txt`
  - `metadata.json`
  - `colmap_submodel/`
    - `images`（软链接）
    - `sparse/0/cameras.txt`
    - `sparse/0/images.txt`
    - `sparse/0/points3D.txt`
- `output_xxx/room_2/...`
- ...

---

## 6. 参数说明与调参建议

## 6.1 关键参数

- `--min-shared-points`
  - 相机对最小共享点数阈值
  - 越大越保守，房间数倾向更多（易碎片化）

- `--edge-threshold`
  - 相机图连边相似度阈值
  - 越大越难连边，房间数可能增多

- `--knn`
  - 每个相机最多保留的邻居数
  - 越大图越稠密，房间倾向更少（更容易合并）

- `--min-room-size`
  - 小簇并入阈值
  - 越大越倾向得到少数大房间

- `--no-export-submodels`
  - 关闭子模型导出，仅保留聚类结果

- `--auto-adaptive`
  - 开启自适应二阶段自动分割（推荐）

- `--shared-quantile`
  - 自适应 `min_shared_points` 的分位数

- `--sim-quantile`
  - 自适应 `edge_threshold` 的分位数

- `--shared-floor`, `--shared-cap`
  - 自适应共享点阈值上下界

- `--target-rooms`
  - 目标房间数（0 表示不强制）

- `--merge-score-threshold`
  - 二阶段自动合并停止阈值（仅当 `target-rooms=0` 生效）

- `--boundary-purity-threshold`
  - 边界相机判定阈值（用于输出 `boundary_cameras.json`）

## 6.2 实战建议

如果出现“房间太碎”（自动模式下房间数偏多）：
- 降低 `--min-shared-points`
- 降低 `--edge-threshold`
- 增大 `--knn`
- 增大 `--min-room-size`
- 或者直接设置 `--target-rooms`

如果出现“房间合并过度”（自动模式下房间数偏少）：
- 增大 `--min-shared-points`
- 增大 `--edge-threshold`
- 减小 `--knn`
- 减小 `--min-room-size`
- 或提高 `--shared-quantile` / `--sim-quantile`

---

## 7. 典型命令

### 7.1 含子模型导出（默认）

```bash
python colmap_spliter/split_rooms.py \
  --input /path/to/dataset \
  --output /path/to/output \
  --min-shared-points 30 \
  --edge-threshold 0.015 \
  --knn 20 \
  --min-room-size 20
```

### 7.2 推荐：自适应二阶段（不指定房间数）

```bash
python colmap_spliter/split_rooms.py \
  --input /path/to/dataset \
  --output /path/to/output \
  --auto-adaptive \
  --no-export-submodels
```

### 7.3 推荐：自适应二阶段 + 目标房间数

```bash
python colmap_spliter/split_rooms.py \
  --input /path/to/dataset \
  --output /path/to/output \
  --auto-adaptive \
  --target-rooms 5
```

### 7.4 批量一键脚本（推荐）

新增脚本：`colmap_spliter/adaptive_split.sh`

单数据集：

```bash
bash colmap_spliter/adaptive_split.sh \
  --input /path/to/dataset \
  --output-root /path/to/output_root \
  --target-rooms 5
```

批量数据集：

```bash
bash colmap_spliter/adaptive_split.sh \
  --input-list /path/to/dataset_list.txt \
  --output-root /path/to/output_root \
  --target-rooms 0
```

`dataset_list.txt` 每行一个数据集根目录（需包含 `sparse/0` 和 `images`）。

---

## 8. 局限性与后续可改进方向

当前方法本质是“相机图连通性聚类”，优点是稳定、简单、无需学习；但仍有局限：

1. 对“长走廊 + 开放空间”场景，可能将多个空间连成同一房间。
2. 相机重叠稀疏时，可能产生小碎簇。
3. 当前二阶段合并仍是启发式，不是全局最优划分。

可改进方向：

1. 用 Louvain/Leiden 替代连通域，提升分区质量。
2. 引入“门洞/遮挡”几何约束，减少跨房间误连边。
3. 在点级别做二次平滑，让点房间标签更连续。
4. 增加可视化（例如导出每房间相机中心 ply），便于人工核验。

---

## 9. 与后续训练对接建议

子模型目录已经满足常见训练脚本对 COLMAP text 模型的读取需求。若训练代码默认读取 `sparse/0/*.bin`，可选择：

1. 直接修改训练入口支持 txt（你的工程已经支持）
2. 用 COLMAP `model_converter` 再转为 bin（可选）

推荐流程：

1. 对每个 `room_i/colmap_submodel` 独立训练
2. 训练后做跨房间边界一致性微调（可选）
3. 最终进行全局融合

以上即当前脚本的完整工作原理。
