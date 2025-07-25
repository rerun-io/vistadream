from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
from einops import rearrange
from icecream import ic
from jaxtyping import Bool, Float, UInt8, UInt16
from monopriors.depth_utils import depth_edges_mask, depth_to_points
from monopriors.relative_depth_models import (
    RelativeDepthPrediction,
    get_relative_predictor,
)
from monopriors.relative_depth_models.base_relative_depth import BaseRelativePredictor
from PIL import Image
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole

from vistadream.ops.flux import FluxInpainting, FluxInpaintingConfig
from vistadream.ops.gs.basic import Frame, Gaussian_Scene
from vistadream.ops.gs.train import GS_Train_Tool
from vistadream.ops.trajs import _generate_trajectory
from vistadream.ops.utils import save_ply
from vistadream.ops.visual_check import Check
from vistadream.resize_utils import add_border_and_mask, process_image


def log_frame(parent_log_path: Path, frame: Frame, cam_params: PinholeParameters):
    cam_log_path: Path = parent_log_path / cam_params.name
    pinhole_log_path: Path = cam_log_path / "pinhole"
    # extract values from frame
    rgb_hw3: UInt8[np.ndarray, "H W 3"] = (deepcopy(frame.rgb) * 255).astype(np.uint8)
    depth_hw: Float[np.ndarray, "H W"] = deepcopy(frame.dpt)
    depth_mm_uint16: UInt16[np.ndarray, "H W"] = np.clip((depth_hw * 1000).round(), 0, 2**16).astype(np.uint16)
    inpaint_mask: Bool[np.ndarray, "H W"] = deepcopy(frame.inpaint)
    inpaint_wo_edge_mask: Bool[np.ndarray, "H W"] = deepcopy(frame.inpaint_wo_edge)

    rr.log(f"{pinhole_log_path}/rgb", rr.Image(rgb_hw3, color_model=rr.ColorModel.RGB).compress())
    rr.log(f"{pinhole_log_path}/depth", rr.DepthImage(depth_mm_uint16, meter=1000.0))
    rr.log(
        f"{pinhole_log_path}/inpaint_mask",
        rr.Image(inpaint_mask.astype(np.uint8) * 255, color_model=rr.ColorModel.L),
    )
    rr.log(
        f"{pinhole_log_path}/inpaint_wo_edges_mask",
        rr.Image(inpaint_wo_edge_mask.astype(np.uint8) * 255, color_model=rr.ColorModel.L),
    )
    log_pinhole(camera=cam_params, cam_log_path=cam_log_path, image_plane_distance=0.05)

    edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(depth_hw, threshold=0.05)
    masked_depth_hw: Float[np.ndarray, "h w"] = depth_hw * ~edges_mask
    depth_1hw: Float[np.ndarray, "1 h w"] = rearrange(masked_depth_hw, "h w -> 1 h w").astype(np.float32)
    pts_3d: Float[np.ndarray, "h w 3"] = depth_to_points(depth_1hw, cam_params.intrinsics.k_matrix.astype(np.float32))
    rr.log(
        f"{parent_log_path}/{cam_params.name}_point_cloud",
        rr.Points3D(
            positions=pts_3d.reshape(-1, 3),
            colors=rgb_hw3.reshape(-1, 3),
        ),
    )


@dataclass
class SingleImageConfig:
    """
    Configuration for Single Image Processing.
    """

    rr_config: RerunTyroConfig
    image_path: Path
    offload: bool = True
    num_steps: int = 25
    guidance: float = 30.0
    expansion_percent: float = 0.15
    n_frames: int = 5


class SingleImagePipeline:
    """
    Pipeline for Flux Outpainting using VistaDream.
    """

    def __init__(self, config: SingleImageConfig):
        self.config: SingleImageConfig = config
        self.scene: Gaussian_Scene = Gaussian_Scene()
        self.flux_inpainter: FluxInpainting = FluxInpainting(FluxInpaintingConfig())
        self.predictor: BaseRelativePredictor = get_relative_predictor("MogeV1Predictor")(device="cuda")
        self.checkor: Check = Check()
        # Initialize rerun with the provided configuration
        self.shared_intrinsics: Intrinsics | None = None
        self.image_plane_distance: float = 0.01
        self.logged_cam_idx_list: list[int] = [0, 1]

        ic("Pipeline initialized with configuration:", self.config)

    def __call__(self):
        # intialize rerun with the provided configuration
        self.setup_rerun()
        # outpaint -> depth prediction -> scene generation
        self._initialize()
        # generate the coarse scene
        save_dir: Path = Path("data/test_dir/")
        gf_path: Path = save_dir / "gf.ply"
        save_dir.mkdir(exist_ok=True, parents=True)
        self._render_splats()
        save_ply(self.scene, gf_path)

    def _render_splats(self):
        # render 5times frames
        nframes: int = len(self.scene.frames) * 25 if len(self.scene.frames) > 2 else 200
        cam_T_world_traj: Float[np.ndarray, "n_frames 4 4"] = _generate_trajectory(None, self.scene, nframes=nframes)
        # render
        print(f"[INFO] rendering final video with {nframes} frames...")
        for i, cam_T_world in enumerate(cam_T_world_traj, start=0):
            rr.set_time("time", sequence=i)
            frame = Frame(
                H=self.shared_intrinsics.height,
                W=self.shared_intrinsics.width,
                intrinsic=self.shared_intrinsics.k_matrix,
                cam_T_world=cam_T_world,
            )
            rgb, dpt, alpha = self.scene._render_RGBD(frame)
            rgb: Float[np.ndarray, "H W 3"] = rgb.detach().float().cpu().numpy()
            dpt: Float[np.ndarray, "H W"] = dpt.detach().float().cpu().numpy()

            rgb = (rgb * 255).astype(np.uint8)

            rr.set_time("time", sequence=i)
            cam_log_path: Path = self.final_log_path / "camera"
            pinhole_log_path: Path = cam_log_path / "pinhole"
            pinhole_param = PinholeParameters(
                name="camera",
                intrinsics=self.shared_intrinsics,
                extrinsics=Extrinsics(
                    cam_R_world=cam_T_world[:3, :3],
                    cam_t_world=cam_T_world[:3, 3],
                ),
            )
            rr.log(f"{pinhole_log_path}/rgb", rr.Image(rgb, color_model=rr.ColorModel.RGB).compress())
            log_pinhole(
                camera=pinhole_param, cam_log_path=cam_log_path, image_plane_distance=self.image_plane_distance * 10
            )
            depth_mm_uint16: UInt16[np.ndarray, "H W"] = np.clip((dpt * 1000).round(), 0, 2**16).astype(np.uint16)
            rr.log(f"{pinhole_log_path}/depth", rr.DepthImage(depth_mm_uint16, meter=1000.0))

            edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(dpt, threshold=0.01)
            masked_depth_hw: Float[np.ndarray, "h w"] = dpt * ~edges_mask
            depth_1hw: Float[np.ndarray, "1 h w"] = rearrange(masked_depth_hw, "h w -> 1 h w").astype(np.float32)
            pts_3d: Float[np.ndarray, "h w 3"] = depth_to_points(
                depth_1hw, pinhole_param.intrinsics.k_matrix.astype(np.float32)
            )

            # Downscale point cloud using Open3D
            # Flatten points and colors
            pts_flat = pts_3d.reshape(-1, 3)
            rgb_flat = rgb.reshape(-1, 3) / 255.0  # Open3D expects [0,1] floats
            # Remove invalid points (NaN/Inf)
            valid_mask = np.isfinite(pts_flat).all(axis=1)
            pts_valid = pts_flat[valid_mask]
            rgb_valid = rgb_flat[valid_mask]
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_valid)
            pcd.colors = o3d.utility.Vector3dVector(rgb_valid)
            # Downsample
            pcd_down = pcd.voxel_down_sample(voxel_size=0.01)
            down_pts = np.asarray(pcd_down.points)
            down_rgb = np.asarray(pcd_down.colors)
            # Convert colors back to uint8 for rerun
            down_rgb_uint8 = (down_rgb * 255).clip(0, 255).astype(np.uint8)
            rr.log(
                f"{cam_log_path}/point_cloud",
                rr.Points3D(
                    positions=down_pts,
                    colors=down_rgb_uint8,
                ),
            )

        print(f"[INFO] DONE rendering {nframes} frames.")

    def _initialize(self):
        rr.set_time("time", sequence=0)

        input_image: Image.Image = Image.open(self.config.image_path).convert("RGB")
        # ensures image is correctly sized and processed
        input_image: Image.Image = process_image(input_image)

        # Auto-generate outpainting setup: user-controlled border expansion
        border_percent: float = (
            self.config.expansion_percent / 2.0
        )  # Convert to fraction per side (divide by 2 for each side)
        border_output: tuple[Image.Image, Image.Image] = add_border_and_mask(
            input_image,
            zoom_all=1.0,
            zoom_left=border_percent,
            zoom_right=border_percent,
            zoom_up=border_percent,
            zoom_down=border_percent,
            overlap=0,
        )
        outpaint_img: Image.Image = border_output[0]
        outpaint_mask: Image.Image = border_output[1]
        outpaint_img: Image.Image = self.flux_inpainter(rgb_hw3=np.array(outpaint_img), mask=np.array(outpaint_mask))

        outpaint_rgb_hw3: UInt8[np.ndarray, "H W 3"] = np.array(outpaint_img.convert("RGB"))
        outpaint_rel_depth: RelativeDepthPrediction = self.predictor.__call__(rgb=outpaint_rgb_hw3, K_33=None)
        outpaint_depth_hw: Float[np.ndarray, "H W"] = outpaint_rel_depth.depth
        # remove any nans or infs and set them to 0
        outpaint_depth_hw[np.isnan(outpaint_depth_hw) | np.isinf(outpaint_depth_hw)] = 0
        # mask showing where outpainting (inpainting) is applied
        outpaint_mask: Bool[np.ndarray, "H W"] = np.array(outpaint_mask).astype(np.bool_)
        # depth edges, True near edges, False otherwise
        outpaint_edges_mask: Bool[np.ndarray, "H W"] = depth_edges_mask(outpaint_depth_hw, threshold=0.1)
        # inpaint/outpaint mask without edges (True where inpainting is applied, False near edges and where no inpainting)
        outpaint_wo_edges: Bool[np.ndarray, "H W"] = outpaint_mask & ~outpaint_edges_mask

        outpaint_intri: Intrinsics = Intrinsics(
            camera_conventions="RDF",
            fl_x=outpaint_rel_depth.K_33[0, 0].item(),
            fl_y=outpaint_rel_depth.K_33[1, 1].item(),
            cx=outpaint_rel_depth.K_33[0, 2].item(),
            cy=outpaint_rel_depth.K_33[1, 2].item(),
            width=outpaint_rgb_hw3.shape[1],
            height=outpaint_rgb_hw3.shape[0],
        )
        outpaint_extri = Extrinsics(
            world_R_cam=np.eye(3, dtype=np.float32),
            world_t_cam=np.zeros(3, dtype=np.float32),
        )
        outpaint_pinhole: PinholeParameters = PinholeParameters(
            name="camera_1",
            intrinsics=outpaint_intri,
            extrinsics=outpaint_extri,
        )

        outpaint_frame: Frame = Frame(
            H=outpaint_rgb_hw3.shape[0],
            W=outpaint_rgb_hw3.shape[1],
            rgb=outpaint_rgb_hw3.astype(np.float32) / 255.0,  # Convert to [0,1] range
            dpt=outpaint_depth_hw,
            intrinsic=outpaint_intri.k_matrix,
            cam_T_world=outpaint_extri.world_T_cam,  # Identity matrix for world coordinates
            inpaint=outpaint_mask,
            inpaint_wo_edge=outpaint_wo_edges,
        )

        log_frame(parent_log_path=self.parent_log_path, frame=outpaint_frame, cam_params=outpaint_pinhole)

        input_rgb_hw3: UInt8[np.ndarray, "H W 3"] = np.array(input_image.convert("RGB"))
        # get input depth from outpaint depth, where outpaint mask is False.
        # This allows for getting the depth of the original image and only having to run the depth model once
        input_area = ~outpaint_mask
        input_depth_hw: Float[np.ndarray, "H W"] = outpaint_depth_hw[input_area].reshape(
            input_image.height, input_image.width
        )
        """
        ### Why Input Frame `inpaint` is Set to All `True`

        1.  **The Conceptual Issue**
            At first glance, this seems wrong because the input frame contains the original image data, and original pixels shouldn't need "inpainting". We'd expect `inpaint=False` for original content.

        2.  **The Training Logic Explanation**
            The `inpaint` mask in training determines which pixels should be supervised during optimization.
            - When `inpaint=True`: "Supervise this pixel - ensure the rendered result matches the target."
            - When `inpaint=False`: "Don't supervise this pixel - ignore it during training."

        3.  **Why All `True` Makes Sense for the Input Frame**
            For the input frame, setting `inpaint=True` everywhere means: "Train the Gaussians to perfectly reproduce the original image." All original pixels become supervision targets, forcing the 3D representation to learn to render the input view accurately.

        4.  **The Two-Frame Training Strategy**
            - **Input Frame**: Learns to reproduce original content. `input_frame.inpaint` is all `True`, so ALL pixels are supervised.
            - **Outpaint Frame**: Learns to reproduce only new content. `outpaint_frame.inpaint` is `True` only for outpainted areas.

        5.  **What This Achieves**
            - **Input Frame Training**: Every original pixel has `inpaint=True`, so the model learns to render them perfectly.
            - **Outpaint Frame Training**: Original pixels have `inpaint=False` (they are already learned, so we skip them), while new outpainted pixels have `inpaint=True` so the model learns to render the new content.

        6.  **The Alternative Would Be Problematic**
            If we set `input_frame.inpaint = False` everywhere, there would be no supervision on the original content. The Gaussians wouldn't learn to render the input view, leading to poor reconstruction quality for the reference image. Training would only learn the outpainted content.
        """
        input_mask: Bool[np.ndarray, "H W"] = np.full_like(input_depth_hw, True, dtype=np.bool_)
        input_edges_mask: Bool[np.ndarray, "H W"] = outpaint_wo_edges[input_area].reshape(
            input_image.height, input_image.width
        )
        input_mask_wo_edges: Bool[np.ndarray, "H W"] = ~input_edges_mask
        input_k33: Float[np.ndarray, "3 3"] = outpaint_rel_depth.K_33.copy()
        # focal stays the same, but principal point is adjusted to center of input image
        input_k33[0, 2] = input_rgb_hw3.shape[1] / 2.0
        input_k33[1, 2] = input_rgb_hw3.shape[0] / 2.0

        input_intri: Intrinsics = Intrinsics(
            camera_conventions="RDF",
            fl_x=input_k33[0, 0].item(),
            fl_y=input_k33[1, 1].item(),
            cx=input_k33[0, 2].item(),
            cy=input_k33[1, 2].item(),
            width=input_rgb_hw3.shape[1],
            height=input_rgb_hw3.shape[0],
        )
        input_extri = Extrinsics(
            world_R_cam=np.eye(3, dtype=np.float32),
            world_t_cam=np.zeros(3, dtype=np.float32),
        )
        input_pinhole: PinholeParameters = PinholeParameters(
            name="camera_0",
            intrinsics=input_intri,
            extrinsics=input_extri,
        )
        self.shared_intrinsics = input_intri

        input_frame: Frame = Frame(
            H=input_rgb_hw3.shape[0],
            W=input_rgb_hw3.shape[1],
            rgb=input_rgb_hw3.astype(np.float32) / 255.0,  # Convert to [0,1] range
            dpt=input_depth_hw,
            intrinsic=input_k33,
            cam_T_world=input_extri.world_T_cam,  # Identity matrix for world coordinates
            inpaint=input_mask,
            inpaint_wo_edge=input_mask_wo_edges,
        )

        log_frame(parent_log_path=self.parent_log_path, frame=input_frame, cam_params=input_pinhole)

        self.scene._add_trainable_frame(input_frame, require_grad=True)
        self.scene._add_trainable_frame(outpaint_frame, require_grad=True)
        self.scene = GS_Train_Tool(self.scene, iters=100)(self.scene.frames)

    def setup_rerun(self):
        self.parent_log_path: Path = Path("/world")
        self.final_log_path: Path = Path("/final")

        rr.send_blueprint(blueprint=self._create_blueprint())
        rr.log("/", rr.ViewCoordinates.RDF, static=True)

    def _create_blueprint(self):
        """
        Create a rerun blueprint for the pipeline.
        """
        # only show at most 5 cameras, pick them distributed
        if len(self.logged_cam_idx_list) > 5:
            # Sample 5 indices evenly from the list (including endpoints)
            idxs = np.linspace(0, len(self.logged_cam_idx_list) - 1, 5, dtype=int)
            view_cam_list: list[int] = [self.logged_cam_idx_list[i] for i in idxs]
        else:
            view_cam_list: list[int] = self.logged_cam_idx_list
        grid_view = rrb.Horizontal(
            contents=[
                rrb.Vertical(
                    contents=[
                        rrb.Spatial2DView(
                            origin=f"{self.parent_log_path}/camera_{i}/pinhole/mask",
                            contents=[
                                "+ $origin/**",
                            ],
                            name=f"Camera {i} Mask",
                        ),
                        rrb.Spatial2DView(
                            origin=f"{self.parent_log_path}/camera_{i}/pinhole/rgb",
                            contents=[
                                "+ $origin/**",
                            ],
                            name=f"Camera {i} RGB",
                        ),
                        rrb.Spatial2DView(
                            origin=f"{self.parent_log_path}/camera_{i}/pinhole/inpainted_rgb",
                            contents=[
                                "+ $origin/**",
                            ],
                            name=f"Inpainted Camera {i} Pinhole Content",
                        ),
                        rrb.Spatial2DView(
                            origin=f"{self.parent_log_path}/camera_{i}/pinhole/depth",
                            contents=[
                                "+ $origin/**",
                            ],
                            name=f"Camera {i} Depth",
                        ),
                    ]
                )
                for i in view_cam_list
            ]
        )

        content_3d = [
            "+ $origin/**",
            *[f"- {self.parent_log_path}/camera_{i}/pinhole/depth" for i in self.logged_cam_idx_list],
            *[f"- {self.parent_log_path}/camera_{i}/pinhole/inpaint_mask" for i in self.logged_cam_idx_list],
            *[f"- {self.parent_log_path}/camera_{i}/pinhole/inpaint_wo_edges_mask" for i in self.logged_cam_idx_list],
        ]

        blueprint = rrb.Blueprint(
            rrb.Tabs(
                rrb.Horizontal(
                    rrb.Spatial3DView(origin=self.parent_log_path, contents=content_3d),
                    grid_view,
                    column_shares=[5, 2],
                    name="Initial Coarse Scene",
                ),
                rrb.Horizontal(
                    rrb.Spatial3DView(
                        origin=self.final_log_path,
                        contents=[
                            "+ $origin/**",
                            f"- {self.final_log_path}/camera/pinhole/depth",
                        ],
                    ),
                    rrb.Vertical(
                        contents=[
                            rrb.Spatial2DView(
                                origin=f"{self.final_log_path}/camera/pinhole/rgb",
                                contents=["+ $origin/**"],
                                name="Final RGB",
                            ),
                            rrb.Spatial2DView(
                                origin=f"{self.final_log_path}/camera/pinhole/depth",
                                contents=["+ $origin/**"],
                                name="Final Depth",
                            ),
                        ]
                    ),
                    column_shares=[5, 2],
                    name="Final Scene",
                ),
            ),
            collapse_panels=True,
        )
        return blueprint


def main(config: SingleImageConfig) -> None:
    """
    Main function to run the Single Image Processing Outpainting/Depth/Splat.
    """
    vd_pipeline = SingleImagePipeline(config)
    vd_pipeline()
