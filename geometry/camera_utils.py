import torch
import torch.nn.functional as funct
import numpy as np


def construct_K(fx, fy, cx, cy, dtype=torch.float, device=None):
    """Construct a [3,3] camera intrinsics from pinhole parameters"""

    return torch.tensor([[fx,  0, cx],
                         [0, fy, cy],
                         [0,  0,  1]], dtype=dtype, device=device)


def scale_intrinsics(K, x_scale, y_scale):
    """Scale intrinsics given x_scale and y_scale factors"""
    K[..., 0, 0] *= x_scale
    K[..., 1, 1] *= y_scale
    K[..., 0, 2] *= x_scale
    K[..., 1, 2] *= y_scale
    return K

def crop_intrinsics(intrinsics, borders):
    """
    Crop camera intrinsics matrix

    Parameters
    ----------
    intrinsics : np.array [3,3]
        Original intrinsics matrix
    borders : tuple
        Borders used for cropping (left, top, right, bottom)
    Returns
    -------
    intrinsics : np.array [3,3]
        Cropped intrinsics matrix
    """
    intrinsics = np.copy(intrinsics)
    intrinsics[0, 2] -= borders[0] #cx
    intrinsics[1, 2] -= borders[1] #cy
    return intrinsics