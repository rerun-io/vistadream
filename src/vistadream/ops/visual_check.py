from copy import deepcopy
from pathlib import Path

import imageio
import matplotlib
import numpy as np
import torch

from vistadream.ops.gs.basic import Frame, Gaussian_Scene  # Import only the required classes
from vistadream.ops.trajs import _generate_trajectory
from vistadream.ops.utils import visual_pcd  # Add other required imports explicitly


class Check:
    def __init__(self) -> None:
        pass

    def _visual_pcd(self, scene: Gaussian_Scene):
        xyzs, rgbs = [], []
        for i, gf in enumerate(scene.gaussian_frames):
            xyz = gf.xyz.detach().cpu().numpy()
            rgb = torch.sigmoid(gf.rgb).detach().cpu().numpy()
            opacity = gf.opacity.detach().squeeze().cpu().numpy() > 1e-5
            xyzs.append(xyz[opacity])
            rgbs.append(rgb[opacity])
        xyzs = np.concatenate(xyzs, axis=0)
        rgbs = np.concatenate(rgbs, axis=0)
        visual_pcd(xyzs, color=rgbs, normal=True)

    @torch.no_grad()
    def _render_video(self, scene: Gaussian_Scene, save_dir: Path, colorize: bool = False, nframes: int | None = None):
        """
        Renders a video from a given Gaussian_Scene and saves RGB and depth videos to the specified directory.

        Args:
            scene (Gaussian_Scene): The scene to render, containing frames and rendering methods.
            save_dir (Path): Directory where the output videos will be saved.
            colorize (bool, optional): If True, colorizes the depth video using the 'plasma' colormap.
                                       If False, outputs grayscale depth. Defaults to False.

        Notes:
            - The number of frames in the output video is determined by the number of frames in the scene.
            - The rendered images are resized if their dimensions exceed 512 pixels.
            - Depth values are normalized between the 1st and 99th percentiles before visualization.
            - Two videos are saved: 'video_rgb.mp4' for RGB frames and 'video_dpt.mp4' for depth frames.
        """
        # render 5times frames
        nframes = len(scene.frames) * 25 if nframes is None else nframes
        video_trajs = _generate_trajectory(None, scene, nframes=nframes)
        H, W, intrinsic = scene.frames[0].H, scene.frames[0].W, deepcopy(scene.frames[0].intrinsic)
        # render
        rgbs, dpts = [], []
        print(f"[INFO] rendering final video with {nframes} frames...")
        for pose in video_trajs:
            frame = Frame(H=H, W=W, intrinsic=intrinsic, cam_T_world=np.linalg.inv(pose))
            rgb, dpt, alpha = scene._render_RGBD(frame)
            rgb = rgb.detach().float().cpu().numpy()
            dpt = dpt.detach().float().cpu().numpy()
            dpts.append(dpt)
            rgbs.append((rgb * 255).astype(np.uint8))
        rgbs = np.stack(rgbs, axis=0)
        dpts = np.stack(dpts, axis=0)
        valid_dpts = dpts[dpts > 0.0]
        _min = np.percentile(valid_dpts, 1)
        _max = np.percentile(valid_dpts, 99)
        dpts = (dpts - _min) / (_max - _min)
        dpts = dpts.clip(0, 1)

        if colorize:
            cm = matplotlib.colormaps["plasma"]
            dpts_color = cm(dpts, bytes=False)[..., 0:3]
            dpts_color = (dpts_color * 255).astype(np.uint8)
            dpts = dpts_color
        else:
            dpts = dpts[..., None].repeat(3, axis=-1)
            dpts = (dpts * 255).astype(np.uint8)

        imageio.mimwrite(f"{save_dir}/video_rgb.mp4", rgbs, fps=20)
        imageio.mimwrite(f"{save_dir}/video_dpt.mp4", dpts, fps=20)
