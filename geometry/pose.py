import torch
# from geometry.pose_utils import invert_pose, pose_vec2mat


class Pose:
    """
    Pose class, that encapsulates a [4,4] transformation matrix
    for a specific reference frame
    """
    def __init__(self, mat):
        """
        Initializes a Pose object.

        Parameters
        ----------
        mat : torch.Tensor [B,4,4]
            Transformation matrix
        """
        assert tuple(mat.shape[-2:]) == (4, 4)
        if mat.dim() == 2:
            mat = mat.unsqueeze(0)
        assert mat.dim() == 3
        self.mat = mat

    def __len__(self):
        """Batch size of the transformation matrix"""
        return len(self.mat)

    @classmethod
    def identity(cls, N=1, device=None, dtype=torch.float):
        """Initializes as a [4,4] identity matrix"""
        return cls(torch.eye(4, device=device, dtype=dtype).repeat([N,1,1]))