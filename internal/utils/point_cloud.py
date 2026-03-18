import open3d as o3d
import numpy as np

def get_hidden_point_mask(point_cloud, camera_center):
    print("predicting hidden points...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.detach().cpu().numpy())
    # o3d.io.write_point_cloud("point_cloud.pcd", pcd)
    diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
    _, pt_map = pcd.hidden_point_removal(o3d.core.Tensor(camera_center, o3d.core.float32), diameter*100)
    pt_map = pt_map.numpy()
    mask = np.zeros(point_cloud.shape[0], dtype=np.uint8)
    mask[pt_map] = 1
    return pt_map, mask

