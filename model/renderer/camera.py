import math
import torch
import numpy as np
import kaolin as kal

class CameraManager:
    def __init__(self,fovy=np.deg2rad(45), iter_res=[512, 512], cam_near_far=[0.1, 1000.0], 
                 cam_radius=3.0, device="cuda:0"):
        """
        Initialize the Camera Manager with default camera parameters.
        
        Args:
            batch_size (int): Number of cameras in the batch.
            fovy (float): Field of view in radians.
            iter_res (list): Resolution of the camera [height, width].
            cam_near_far (list): Near and far clipping planes [near, far].
            cam_radius (float): Radius of the camera from the scene center.
            device (str): Device to run the computation ("cuda" or "cpu").
        """
        self.fovy = fovy
        self.iter_res = iter_res
        self.cam_near_far = cam_near_far
        self.cam_radius = cam_radius
        self.device = device

    def get_random_camera_batch(self, batch_size=1):
        """
        Generate a batch of randomly positioned cameras.
        Args:
            batch_size (int): Number of cameras in the batch. if None, use the default batch size.
        """
        camera_pos = torch.stack(kal.ops.coords.spherical2cartesian(
            *kal.ops.random.sample_spherical_coords((batch_size,), azimuth_low=0., azimuth_high=math.pi * 2,
                                                    elevation_low=-math.pi / 2., elevation_high=math.pi / 2., device=self.device),
            self.cam_radius
        ), dim=-1)

        return kal.render.camera.Camera.from_args(
            eye=camera_pos + torch.rand((batch_size, 1), device=self.device) * 0.5 - 0.25,
            at=torch.zeros(batch_size, 3),
            up=torch.tensor([[0., 1., 0.]], device=self.device),
            fov=self.fovy,
            near=self.cam_near_far[0], far=self.cam_near_far[1],
            height=self.iter_res[0], width=self.iter_res[1],
            device=self.device
        )

    def get_camera_by_rot(self, elev_list):
        """
        Generate a batch of cameras using specific elevation angles.
        Args:
            elev_list (list): List of elevation angles in degrees.
        """
        elevations = torch.tensor([math.radians(angle) for angle in elev_list], device=self.device)
        azimuths = torch.zeros(len(elev_list), device=self.device)

        camera_pos = torch.stack(kal.ops.coords.spherical2cartesian(
            azimuths,
            elevations,
            self.cam_radius
        ), dim=-1)

        return kal.render.camera.Camera.from_args(
            eye=camera_pos,
            at=torch.zeros(len(elev_list), 3),
            up=torch.tensor([[0., 1., 0.]], device=self.device),
            fov=self.fovy,
            near=self.cam_near_far[0], far=self.cam_near_far[1],
            height=self.iter_res[0], width=self.iter_res[1],
            device=self.device
        )

    def get_camera_batch_from_RT(self, R, T):
        """
        Generate a batch of cameras using specified R (rotation) and T (translation) matrices.
        
        Args:
            R (torch.Tensor): Rotation matrix of shape (batch_size, 3, 3) or (3, 3).
            T (torch.Tensor): Translation vector of shape (batch_size, 3) or (3,).
        
        Returns:
            kaolin.render.camera.Camera: A batch of cameras with specified R and T.
        """
        batch_size = R.shape[0] if R.dim() == 3 else 1
        view_matrices = torch.eye(4, device=self.device).repeat(batch_size, 1, 1)

        if R.dim() == 2:
            R = R.unsqueeze(0).expand(batch_size, -1, -1)

        if T.dim() == 1:  
            T = T.unsqueeze(0).expand(batch_size, -1)

        view_matrices[:, :3, :3] = R
        view_matrices[:, :3, 3] = T.unsqueeze(-1) 

        # Return Kaolin camera object
        return kal.render.camera.Camera.from_args(
            view_matrix=view_matrices,
            width=self.iter_res[0],
            height=self.iter_res[1],
            device=self.device
        )

    def sample_camera(self, mode="random", **kwargs):
        """
        Select and sample a camera based on the given mode.
        
        Args:
            mode (str): The camera sampling mode, one of ["random", "rotating", "fixed_rotation", "from_RT"].
            **kwargs: Additional parameters specific to the selected camera mode.

        Returns:
            A batch of sampled camera objects.
        """
        if mode == "random":
            return self.get_random_camera_batch(kwargs.get("batch_size"))
        elif mode == "rotating":
            return self.get_camera_by_rot(kwargs.get("elev_list"))
        elif mode == "from_RT":
            return self.get_camera_batch_from_RT(kwargs.get("R"), kwargs.get("T"))
        else:
            raise ValueError(f"Unknown camera sampling mode: {mode}")
        
    def merged_camera(self, cameras):
        """
        Merge a list of camera objects into a single batch.
        
        Args:
            cameras (list): List of camera objects.
        
        Returns:
            kaolin.render.camera.Camera: A batch of merged cameras.
        """
        return kal.render.camera.Camera.cat(cameras)

