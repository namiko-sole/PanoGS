import os
import sys
import cv2
import numpy as np
from . import Perspec2Equirec as P2E
from PIL import Image

def get_xyz(width, height):
    x,y = np.meshgrid(np.linspace(-180, 180,width),np.linspace(90,-90,height))
        
    x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
    y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
    z_map = np.sin(np.radians(y))

    xyz = np.stack((x_map,y_map,z_map),axis=2)
    return xyz

class Perspective:
    def __init__(self, img_array , F_T_P_array ):
        
        assert len(img_array)==len(F_T_P_array)
        
        self.img_array = img_array
        self.F_T_P_array = F_T_P_array
    
    def GetEquirec(self,height,width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #
        channels = self.img_array[0].shape[2]
        merge_image = np.zeros((height,width,channels))
        merge_mask = np.zeros((height,width,channels))

        xyz = get_xyz(width, height)

        for img_dir,[F,T,P] in zip (self.img_array,self.F_T_P_array):
            per = P2E.Perspective(img_dir,F,T,P)        # Load equirectangular image
            img , mask = per.GetEquirec(xyz,height,width)   # Specify parameters(FOV, theta, phi, height, width)
            # NOTE: the old linear-ramp "feather" weights were removed on purpose.
            # With 90-degree faces every equirect pixel is covered by exactly one
            # face, so the ramps cancel out everywhere except at the face boundary
            # lines where they are ~0. Under OpenCV >= 5 cv2.remap evaluates the
            # ramp to exactly 0 there, which triggered the mean-color fill below
            # and produced visible constant-value seam lines in the panorama.
            # `img` is already multiplied by the coverage mask inside
            # Perspec2Equirec.GetEquirec, so accumulate it directly; this is
            # cv2-version independent (verified identical on OpenCV 4.13 and 5.0).
            merge_image += img.astype(np.float32)
            merge_mask += mask.astype(np.float32)
        merge_image[merge_mask==0] = merge_image.mean()

        merge_mask = np.where(merge_mask==0,1,merge_mask)
        merge_image = (np.divide(merge_image,merge_mask))

        return merge_image
        
        
