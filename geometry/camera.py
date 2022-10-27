# inspired by Toyota PackSfM repo
import sys
import os
from tkinter import Y
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from functools import lru_cache
import numpy as np
import torch.nn as nn
from geometry.camera_utils import scale_intrinsics, crop_intrinsics
from geometry.pose import Pose


# from geometry.camera_utils import scale_intrinsics, crop_intrinsics
import math


#subclass of nn.module
class Camera(nn.Module):
    """
    Differentiable camera class implementing reconstruction and projection
    functions for a pinhole model.
    """
    def __init__(self, K, dimensions, Tcw=None):
        """
        Initializes the Camera class

        Parameters
        ----------
        K : torch.Tensor [3,3]
            Camera intrinsics (3D->2D)
        Tcw : Pose
            Camera -> World pose transformation (3D->3D)
        """
        super().__init__() #avoids explicitaly reference to nn.Modulo
        self.K = K 
        self.Tcw = Pose.identity(len(K)) if Tcw is None else Tcw #Pose is the pose of color cam while photos
        self.W, self.H = dimensions #each camera has the dimensions of the photo it takes, new dim new K

    def __len__(self):
        """Batch size of the camera intrinsics"""
        return len(self.K)
    
    def project(self, points_coordinates):
        """
        Projects 3D points onto the image plane

        Parameters
        ----------
        point_coordinates : torch.Tensor [N,3]
            3D points to be projected

        Returns
        -------
        points : torch.Tensor [N,2]
            2D projected points that are within the image boundaries
        """
        # print("self.Tcw: ", self.Tcw)

        N, C = points_coordinates.shape
        homo_ones = np.ones((N,1))
        points_homogeneous_coordinates = np.concatenate([points_coordinates, homo_ones], axis=1) #[xw yw zw 1].T
        pixels_coordinates = (self.K @ (( np.linalg.inv(self.Tcw)@points_homogeneous_coordinates.T))).T #T stands for Tranposal #############

        X = pixels_coordinates[:, 0] #1st colum
        Y = pixels_coordinates[:, 1]
        Z = pixels_coordinates[:, 2]

        #getting pixels normalized
        Xnorm = (X/Z) / self.W
        Ynorm = (Y/Z) / self.H

        Xmask = ((Xnorm >= 1) + (Xnorm < 0))
        Xnorm[Xmask] = -1
        Ymask = ((Ynorm >= 1) + (Ynorm < 0))
        Ynorm[Ymask] = -1
        return np.stack([Xnorm, Ynorm, Z], axis=-1).reshape(N, 3) # Return pixel coordinates

    @property
    def fx(self):
        """Focal length in x"""
        return self.K[:, 0, 0]

    @property
    def fy(self):
        """Focal length in y"""
        return self.K[:, 1, 1]

    @property
    def cx(self):
        """Principal point in x"""
        return self.K[:, 0, 2]

    @property
    def cy(self):
        """Principal point in y"""
        return self.K[:, 1, 2]
    
    @property
    @lru_cache()
    def Twc(self):
        """World -> Camera pose transformation (inverse of Tcw)"""
        return self.Tcw.inverse()

    def scaled(self, x_scale, y_scale=None):
        """
        Returns a scaled version of the camera (changing intrinsics)

        Parameters
        ----------
        x_scale : float
            Resize scale in x
        y_scale : float
            Resize scale in y. If None, use the same as x_scale

        Returns
        -------
        camera : Camera
            Scaled version of the current cmaera
        """
        if y_scale is None: # If single value is provided, use for both dimensions
            y_scale = x_scale

        K = scale_intrinsics(self.K.copy(), x_scale, y_scale) # Scale intrinsics and return new camera with same Pose
        return Camera(K, (int(self.W*x_scale), int(self.H*y_scale)), Tcw=self.Tcw)   
    
    def crop(self, borders):
        """
        Crops the camera intrinsics to a given region

        Parameters
        ----------
        x_min : float
            Minimum x value
        x_max : float
            Maximum x value
        y_min : float
            Minimum y value
        y_max : float
            Maximum y value

        Returns
        -------
        camera : Camera
            Cropped version of the current camera
        """
        K = crop_intrinsics(self.K.copy(), borders)
        left, top, right, bottom = borders
        # print('int(-left+right): ', int(-left+right) )
        return Camera(K, (int(-left+right), int(-top+bottom)), Tcw=self.Tcw)

    def project_on_image_min(self, self_pc, self_colors):
        """you pick the coord(x_w, y_w, z_w) of each pixel that became a point and them see if it is in the image
        
        output:
        image - 
        pc_im_correspondances - 
        """
        proj = self.project(self_pc) #return the 2d pixel whose point is in the img 
        # if corres: print('proj corres',proj)        

        image = np.zeros((self.H, self.W, len(self_colors[0]))) #H = W = 224
        depth_track = np.zeros((224,224,1))
        pc_im_correspondances = {}

        for i in range(len(self_pc)): #2nd for each point in the point cloud of scanet (#1st use could be for each pixels)
            x, y = int((proj[i,1]) * self.H), int((proj[i,0]) * self.W) #'Desnormalizing'
            Z = proj[i, 2]
        
            if proj[i,0] >=0 and proj[i,1] >= 0 and Z>=0:

                if(depth_track[x,y,:]!=0): 
                    depth_track[x,y,:] = min(Z,depth_track[x,y,:])
                image[x,y,:] = self_colors[i]/255
                if(depth_track[x,y,:]==0): depth_track[x,y,:] = Z

                pc_im_correspondances[i] = np.array([x, y])

        
        return image, pc_im_correspondances