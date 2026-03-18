import os
from PIL import Image
import numpy as np
import torch
import cv2
import math
import open3d as o3d

def depth_image_to_point_cloud(depth, cx, cy, fx, fy):
    u = range(0, depth.shape[1])
    v = range(0, depth.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    Z = depth.astype(float)
    # X = (u - K[0, 2]) * Z / K[0, 0]
    # Y = (v - K[1, 2]) * Z / K[1, 1]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    valid = Z > 0

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    position = np.vstack((X, Y, Z, np.ones(len(X))))
    # position = np.dot(pose, position)

    # R = np.ravel(rgb[:, :, 0])[valid]
    # G = np.ravel(rgb[:, :, 1])[valid]
    # B = np.ravel(rgb[:, :, 2])[valid]

    # points = np.transpose(np.vstack((position[0:3, :], R, G, B))).tolist()

    return position

def point_cloud_to_depth_image(points, cx, cy, fx, fy, w, h):
    depth = points[..., 2]
    X, Y = points[..., 0], points[..., 1]
    u = X * fx / depth + cx
    v = Y * fy / depth + cy
    sorted_idx = np.argsort(depth)[::-1]
    depth = depth[sorted_idx]
    u = u[sorted_idx]
    v = v[sorted_idx]

    depth_img = np.full((w, h), np.inf)
    depth_img[u,v] = depth
    return depth_img

def get_pts(center, depth, width, height, mask=None):
    x,y = np.meshgrid(np.linspace(-180,180,width), np.linspace(-90,90,height))

    x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
    y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
    z_map = np.sin(np.radians(y))

    xyz = np.stack((x_map,y_map,z_map),axis=2)
    
    if mask is not None:
        pts = xyz[mask]*depth[mask] + center
    else:
        pts = (xyz*depth + center)
    return pts

def get_pts_inter(s_R, s_T, depth, width, height):
    x,y = np.meshgrid(np.linspace(-180,180,width), np.linspace(-90,90,height))

    x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
    y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
    z_map = np.sin(np.radians(y))

    xyz = np.stack((x_map,y_map,z_map),axis=2)
    # xyz = xyz@np.linalg.inv(s_R)
    
    pts = (xyz*depth + s_T)
    return pts

def direction_to_pano_coord(dirs):
    dirs = dirs / torch.linalg.norm(dirs, 2, -1, True)
    beta = torch.arcsin(dirs[..., 2])
    xy = dirs[..., :2] / torch.cos(beta)[..., None]
    alpha = torch.view_as_complex(xy).angle()   # [-np.pi., np.pi]
    return torch.stack([torch.rad2deg(beta), torch.rad2deg(alpha)], -1)

def pano_coord_to_direction(coords):
    beta, alpha = coords[..., 0], coords[..., 1]
    dirs = torch.stack([torch.cos(alpha) * torch.cos(beta),
                        torch.sin(alpha) * torch.cos(beta),
                        torch.sin(beta)], dim=-1)
    return dirs

def pano_to_img_coord(coords, width=2048, height=1024):
    y, x = coords[..., 0], coords[..., 1]
    return torch.stack([(y+90)/180*(height-1), (x+180)/360*(width-1)], -1)

def img_to_pano_coord(coords):
    y, x = coords[..., 0], coords[..., 1]
    return torch.stack([-(y - .5) * np.pi, -(x - .5) * 2. * np.pi], -1)

def img_coord_to_sample_coord(coords):
    return torch.stack([coords[..., 1], coords[..., 0]], -1) * 2. - 1.

def direction_to_img_coord(dirs):
    return pano_to_img_coord(direction_to_pano_coord(dirs))

def img_coord_to_pano_direction(coords):
    return pano_coord_to_direction(img_to_pano_coord(coords))


class DepthSplatting():
    def __init__(self, upscale=1):
        self.pts = None
        self.colors = None
        self.upscale = upscale
        self.width = None
        self.height = None
        self.len = 0

        self.depthes = []
        self.camera_Rs = []
        self.camera_Ts = []
        self.cams = []
        self.data = []
        self.pcd = []

    def splat(self, camera_center:np.array, pano_img:np.array, depth_img:np.array, mask=None):
        if not self.width: self.width = pano_img.shape[1]
        if not self.height: self.height = pano_img.shape[0]
        upscale_pano_img = cv2.resize(pano_img, (self.width*self.upscale, self.height*self.upscale))
        upscale_pano_depth = cv2.resize(depth_img, (self.width*self.upscale, self.height*self.upscale))

        if mask is None: mask = np.ones((self.height*self.upscale, self.width*self.upscale), np.bool_)
        else: mask = cv2.resize(mask.astype(np.uint8), (self.width*self.upscale, self.height*self.upscale)).astype(np.bool_)
        pts = get_pts(camera_center, upscale_pano_depth, self.width*self.upscale, self.height*self.upscale, mask)

        self.colors = np.concatenate((self.colors, upscale_pano_img[mask]), axis=0) if self.pts is not None else upscale_pano_img[mask]
        self.pts = np.concatenate((self.pts, pts), axis=0) if self.pts is not None else pts

    def project(self, camera_center:np.array, depth_img:np.array, save_dir=None):
        # dirs = torch.from_numpy(self.pts - camera_center)

        # pano_coords = direction_to_pano_coord(dirs)
        # img_coords = torch.round(pano_to_img_coord(pano_coords, self.width, self.height)).int()

        # img_coords_np = img_coords.detach().cpu().numpy()

        # # mask = np.zeros((1024, 2048))
        # img = np.zeros((self.height, self.width, 3))
        # img[img_coords_np.reshape(-1,2)[...,0], img_coords_np.reshape(-1,2)[...,1]] = self.colors

        # mask = (np.linalg.norm(img,axis=-1)>0)

        # kernel_l = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # mask = cv2.dilate((~mask).astype(np.uint8)*255, kernel=kernel_s)
        # mask_mid = mask
        # mask = cv2.erode(mask, kernel=kernel_l)

        # if save_dir is not None:
        #     Image.fromarray(img.astype(np.uint8)).save(os.path.join(save_dir, "proj_img.png"))
        #     Image.fromarray(mask).save(os.path.join(save_dir, "proj_mask.png"))
        
        # return cv2.resize(mask, (self.width//self.upscale, self.height//self.upscale)).astype(np.bool_)

        print("projecting pts...")
        dirs = torch.from_numpy(self.pts - camera_center)

        pano_coords = direction_to_pano_coord(dirs)
        img_coords = torch.round(pano_to_img_coord(pano_coords, self.width, self.height)).int()
        img_coords_np = img_coords.detach().cpu().numpy()

        proj_depth = torch.linalg.norm(dirs, axis=-1).detach().cpu().numpy()
        pano_depth = depth_img[...,0][img_coords_np.reshape(-1,2)[...,0], img_coords_np.reshape(-1,2)[...,1]]
        depth_thres = 1
        depth_mask = proj_depth < (pano_depth+depth_thres)
        img_coords_np = img_coords_np[depth_mask]

        img = np.zeros((self.height, self.width, 3))
        img[img_coords_np.reshape(-1,2)[...,0], img_coords_np.reshape(-1,2)[...,1]] = self.colors[depth_mask]

        mask = np.zeros((self.height, self.width))
        mask[img_coords_np.reshape(-1,2)[...,0], img_coords_np.reshape(-1,2)[...,1]] = 1

        # mask = (np.linalg.norm(img,axis=-1)>0)
        mask = mask.astype(np.bool_)
        print(f"pixel misssing rate: {1-(mask.sum()/(self.height*self.width))}")

        raw_mask = ~mask.copy()

        kernel_l = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        
        mask = (~mask).astype(np.uint8)*255

        mask = cv2.dilate(mask, kernel=kernel_s)
        mask = cv2.erode(mask, kernel=kernel_s)

        mask_mid = mask.astype(np.bool_)
        mask = mask.astype(np.bool_)

        mask = mask|raw_mask
        
        print(f"mask rate: {1-(~mask.sum()/(self.height*self.width))}")

        if save_dir is not None:
            Image.fromarray(img.astype(np.uint8)).save(os.path.join(save_dir, "proj_img.png"))
            Image.fromarray(mask).save(os.path.join(save_dir, "proj_mask.png"))
            Image.fromarray(mask_mid).save(os.path.join(save_dir, "proj_mask_mid.png"))
    
    def project_and_splat(self, camera_center:np.array, pano_img:np.array, depth_img:np.array, save_dir=None):
        if not self.width: self.width = pano_img.shape[1]
        if not self.height: self.height = pano_img.shape[0]

        mask = np.ones((self.width, self.height)).astype(np.bool_)

        # print("projecting pts...")
        if self.pts is not None:
            dirs = torch.from_numpy(self.pts - camera_center)

            pano_coords = direction_to_pano_coord(dirs)
            img_coords = torch.round(pano_to_img_coord(pano_coords, self.width, self.height)).int()
            img_coords_np = img_coords.detach().cpu().numpy()

            proj_depth = torch.linalg.norm(dirs, axis=-1).detach().cpu().numpy()
            pano_depth = depth_img[...,0][img_coords_np.reshape(-1,2)[...,0], img_coords_np.reshape(-1,2)[...,1]]
            depth_thres = 10
            depth_mask = proj_depth < (pano_depth+depth_thres)
            
            sorted_depth_mask = depth_mask[np.argsort(proj_depth)[::-1]]
            img_coords_np = img_coords_np[sorted_depth_mask]

            img = np.zeros((self.height, self.width, 3))
            img[img_coords_np.reshape(-1,2)[...,0], img_coords_np.reshape(-1,2)[...,1]] = self.colors[sorted_depth_mask]

            mask = np.zeros((self.height, self.width))
            mask[img_coords_np.reshape(-1,2)[...,0], img_coords_np.reshape(-1,2)[...,1]] = 1

            # mask = (np.linalg.norm(img,axis=-1)>0)
            mask = mask.astype(np.bool_)
            print(f"pixel misssing rate: {1-(mask.sum()/(self.height*self.width))}")

            kernel_l = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31))
            kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19,19))
            
            mask = (~mask).astype(np.uint8)*255

            mask = cv2.dilate(mask, kernel=kernel_s)
            mask_mid = mask.astype(np.bool_)
            mask_splat = cv2.dilate(mask, kernel=kernel_s).astype(np.bool_)
            # mask = cv2.erode(mask, kernel=kernel_s)
            mask_mid = mask_splat

            # mask = cv2.erode(mask, kernel=kernel_l)
            # mask = cv2.dilate(mask, kernel=kernel_l)

            mask = mask.astype(np.bool_)
            
            mask_rate = mask.sum()/(self.height*self.width)

            if save_dir is not None:
                Image.fromarray(img.astype(np.uint8)).save(os.path.join(save_dir, "proj_img.png"))

                _save_img = pano_img.copy()
                _save_img[mask] = 0
                Image.fromarray(mask).save(os.path.join(save_dir, "proj_mask.png"))
                Image.fromarray(_save_img.astype(np.uint8)).save(os.path.join(save_dir, "pano_img_masked.png"))

                _save_img = pano_img.copy()
                _save_img[mask_mid] = 0
                Image.fromarray(mask_mid).save(os.path.join(save_dir, "proj_mask_mid.png"))
                Image.fromarray(_save_img.astype(np.uint8)).save(os.path.join(save_dir, "pano_img_masked_mid.png"))

                # _save_img = pano_img.copy()
                # _save_img[mask_splat] = 0
                # Image.fromarray(mask_splat).save(os.path.join(save_dir, "proj_mask_splat.png"))
                # Image.fromarray(_save_img.astype(np.uint8)).save(os.path.join(save_dir, "pano_img_masked_splat.png"))
            
            print(f"mask rate: {mask_rate}")
            # if mask_rate<0.1 or mask_rate>0.5:
            if mask_rate<0.05:
                return False

        print("Splatting pts...")
        upscale_pano_img = cv2.resize(pano_img, (self.width*self.upscale, self.height*self.upscale))
        upscale_pano_depth = cv2.resize(depth_img, (self.width*self.upscale, self.height*self.upscale))
        upscale_mask = cv2.resize(mask.astype(np.uint8), (self.width*self.upscale, self.height*self.upscale)).astype(np.bool_)
        # upscale_mask = np.ones((self.height*self.upscale, self.width*self.upscale)).astype(np.bool_)

        pts = get_pts(camera_center, upscale_pano_depth, self.width*self.upscale, self.height*self.upscale, upscale_mask)
        self.colors = np.concatenate((self.colors, upscale_pano_img[upscale_mask]), axis=0) if self.pts is not None else upscale_pano_img[upscale_mask]
        self.pts = np.concatenate((self.pts, pts), axis=0) if self.pts is not None else pts

        # pts = get_pts(camera_center, upscale_pano_depth, self.width*self.upscale, self.height*self.upscale)
        # self.colors = np.concatenate((self.colors, upscale_pano_img), axis=0)
        # self.pts = np.concatenate((self.pts, pts), axis=0)
        return True
    
    def wrap_and_project(self, camera_R:np.array, camera_T:np.array, pano_img:np.array, depth_img:np.array, save_dir=None):
        if not self.width: self.width = pano_img.shape[1]
        if not self.height: self.height = pano_img.shape[0]

        if len(self.depthes) != 0:
            pts = get_pts_inter(camera_R, camera_T, depth_img, self.width, self.height)

            mask = None
            for t_depth, t_R, t_T in zip(self.depthes, self.camera_Rs, self.camera_Ts):
                # dirs = torch.from_numpy(pts@t_R + t_T)
                dirs = torch.from_numpy(pts - t_T)
                pano_coords = direction_to_pano_coord(dirs)
                img_coords = torch.round(pano_to_img_coord(pano_coords, self.width, self.height)).int()
                img_coords_np = img_coords.detach().cpu().numpy()


                s_depth = depth_img[...,0][img_coords_np.reshape(-1,2)[...,0], img_coords_np.reshape(-1,2)[...,1]]
                ref_depth = t_depth[...,0][img_coords_np.reshape(-1,2)[...,0], img_coords_np.reshape(-1,2)[...,1]]
                _mask = (s_depth < ref_depth+0.1).reshape((self.height, self.width))
                # _mask = np.ones((self.height, self.width)).astype(np.bool_)

                mask = _mask if mask is None else mask | _mask

            if save_dir is not None:
                img = np.zeros((self.height, self.width, 3))
                img[img_coords_np.reshape(-1,2)[...,0], img_coords_np.reshape(-1,2)[...,1]] = depth_img.reshape(-1,3)
                # Image.fromarray(img.astype(np.uint8)).save(os.path.join(save_dir, "proj_img.png"))
                Image.fromarray(((img-img.min())/(img.max()-img.min()+1e-9)*255).astype(np.uint8)).save(os.path.join(save_dir, "proj_img.png"))

                _save_img = pano_img.copy()
                _save_img[mask] = 0
                Image.fromarray(mask).save(os.path.join(save_dir, "proj_mask.png"))
                Image.fromarray(_save_img.astype(np.uint8)).save(os.path.join(save_dir, "pano_img_masked.png"))
            
        self.depthes.append(depth_img)
        self.camera_Rs.append(camera_R)
        self.camera_Ts.append(camera_T)
        return True
    
    def init_pts(self, num, w, h):
        self.pts = np.zeros((num, w, h, 3))
    
    def perspective_proj(self, image, intrinsic, extrinsic, width, height):
        # pts = depth_image_to_point_cloud(depth, cx, cy, fx, fy)
        # self.pts[len] = pts
        # self.cams.append([R,T,cx,cy,fx,fy])
        # self.len += 1

        # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image, intrinsic, extrinsic, depth_scale=1, depth_max=1000)
        pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(image, intrinsic, extrinsic)
        # width = intrinsic[0,2]*2
        # height = intrinsic[1,2]*2
        self.data.append({
            'pcd': pcd,
            'img': image,
            'intrinsic': intrinsic,
            'extrinsic': extrinsic,
            'width': width,
            'height': height,
        })
        self.pcd.append(pcd)


    def perspective_wrap(self, image, intrinsic, extrinsic, width, height):
        pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(image, intrinsic, extrinsic)
        for odata in self.data:
            reproj = pcd.project_to_rgbd_image(width=odata['width'],
                                               height=odata['height'],
                                               intrinsics=odata['intrinsic'],
                                               extrinsics=odata['extrinsic'])
            
            # reproj = odata['pcd'].project_to_rgbd_image(width=width,
            #                                         height=height,
            #                                         intrinsics=intrinsic,
            #                                         extrinsics=extrinsic)
            i=5
        pass