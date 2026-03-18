import torch
import numpy as np
import math
import viser.transforms as vtf
from internal.cameras.cameras import Cameras

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

def pano_to_img_coord(coords, width=2048, height=1024):#, x_offset=0, y_offset=0):
    y, x = coords[..., 0], coords[..., 1]
    # width = 2048
    # height = 1024

    # For playroom
    # x = x + x_offset
    # x[x<-180] += 360
    # x[x>180]  -= 360
    # y = y + y_offset
    # y[y<-90] += 180
    width_tensor = (-(x/360)+.5)*(width-1)
    height_tensor = ((-y/180)+.5)*(height-1)

    # height_tensor = height_tensor + ((y_offset/180)+.5)*(height-1)
    # height_tensor[height_tensor>=height-1] -= height
    # height_tensor[height_tensor<-1] += height
    
    return torch.stack([height_tensor, width_tensor], -1)
    # return torch.stack([((-y/180)+.5)*(height-1), (-(x/360)+.5)*(width-1)], -1)
    # return torch.stack([-y / np.pi + .5, -(x / (2. * np.pi)) + .5], -1)

    # For drjohnson
    # y = y - 45
    # y[y<-90] += 90
    # x = x-offset
    # x[x<-180] += 360
    # return torch.stack([((x/360)+.5)*(height-1), ((y/180)+.5)*(width-1)], -1)

def img_to_pano_coord(coords):
    '''
    :param coords: [n, 2] range of [0, 1]. (row coord, col coord)
    :return: pano coords
    '''
    y, x = coords[..., 0], coords[..., 1]
    return torch.stack([-(y - .5) * np.pi, -(x - .5) * 2. * np.pi], -1)

def img_coord_to_sample_coord(coords):
    return torch.stack([coords[..., 1], coords[..., 0]], -1) * 2. - 1.

def direction_to_img_coord(dirs):
    return pano_to_img_coord(direction_to_pano_coord(dirs))

def img_coord_to_pano_direction(coords):
    return pano_coord_to_direction(img_to_pano_coord(coords))

def get_depth_distort(res=1024):
    spherical_img_depth = torch.ones((res,res))

    cx,cy = res//2,res//2
    x,y = torch.meshgrid(torch.arange(res),torch.arange(res))
    x = x - cx
    y = y - cy
    xy = torch.stack((x,y),axis=2) / cx
    xy_norm = torch.linalg.norm(xy.float(),axis=2)
    
    plain_img_depth = torch.sqrt(spherical_img_depth**2 + xy_norm**2)

    depth_distort = plain_img_depth - spherical_img_depth
    # depth_distort = spherical_img_depth - plain_img_depth
    # Image.fromarray(((depth_distort-depth_distort.min())/(depth_distort.max()-depth_distort.min()+1e-9)*255).astype(np.uint8)).save("test_img2.png")

    return depth_distort

def get_camera_center(R, T):
    world_to_camera = torch.zeros((R.shape[0], 4, 4))
    world_to_camera[:, :3, :3] = R.transpose(1,2)
    world_to_camera[:, :3, 3] = T
    world_to_camera[:, 3, 3] = 1.
    # world_to_camera = torch.transpose(world_to_camera, 1, 2)
    camera_to_world = torch.linalg.inv(world_to_camera)
    return camera_to_world[:, 3, :3]

def camera_position(R, T, trans):
    world_to_camera = torch.zeros((R.shape[0], 4, 4))
    world_to_camera[:, :3, :3] = R.transpose(1,2)
    world_to_camera[:, :3, 3] = T
    world_to_camera[:, 3, 3] = 1.
    # world_to_camera = torch.transpose(world_to_camera, 1, 2)
    camera_to_world = torch.linalg.inv(world_to_camera)
    camera_center = camera_to_world[:, 3, :3] + trans
    camera_to_world[:, 3, :3] = camera_center
    world_to_camera = torch.linalg.inv(camera_to_world)
    # world_to_camera = torch.transpose(world_to_camera, 1, 2)
    return {'R': world_to_camera[:, :3, :3].transpose(1,2), 'T': world_to_camera[:, :3, 3]}

import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelOperator(nn.Module):
    def __init__(self):
        super(SobelOperator, self).__init__()

        self.sobel_x = nn.Conv2d(1,1,kernel_size=3,padding=1,bias=False)
        self.sobel_y = nn.Conv2d(1,1,kernel_size=3,padding=1,bias=False)

        self.sobel_x.weight.data = torch.tensor([[-1,0,1], [-2,0,2], [-1,0,1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y.weight.data = torch.tensor([[-1,-2,-1], [0,0,0], [1,2,1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # self.sobel_x.weight.data = self.sobel_x.weight.data[None,None,...]
        # self.sobel_y.weight.data = self.sobel_y.weight.data[None,None,...]

    def forward(self, input):
        # gradient_x = F.conv2d(x, self.sobel_x)
        # gradient_y = F.conv2d(x, self.sobel_y)

        gradient_x = self.sobel_x(input)
        gradient_y = self.sobel_y(input)

        gradient_magnitude = torch.sqrt(torch.square(gradient_x) + torch.square(gradient_y) + 1e-6)
        gradient_direction = torch.atan2(gradient_y, gradient_x)

        # gradient_magnitude[:,:,0,:] = 0
        # gradient_magnitude[:,:,-1,:] = 0
        # gradient_magnitude[:,:,:,0] = 0
        # gradient_magnitude[:,:,:,-1] = 0

        return gradient_magnitude, gradient_direction


class RGB2Gray(nn.Module):
    '''
    Converts one or more images from RGB to Grayscale.
    Receives a RGB input image with size of (NCHW)
    Outputs a tensor with the size of (N1HW), containing the Grayscale value of the pixels.
    '''
    def __init__(self):
        super(RGB2Gray, self).__init__()

        # rgb weight, shape: (1, 1, 3)
        # https://en.wikipedia.org/wiki/Luma_%28video%29
        rgb_weights = torch.tensor([[[0.2989, 0.5870, 0.1140]]])
        self.register_buffer('rgb_weights', rgb_weights)

    def __call__(self, img):
        '''
        img: RGB input image, shape: NCHW
        '''
        shape = img.shape

        # repeat rgb weights for every image in the mini-batch, shape: (batch, 1, 1, 3)
        rgb_weights = self.rgb_weights.repeat(shape[0], 1, 1)

        img = img.contiguous().view(-1, 3, shape[2] * shape[3])
        gray = torch.bmm(rgb_weights, img)
        gray = gray.view(-1, 1, shape[2], shape[3])
        return gray