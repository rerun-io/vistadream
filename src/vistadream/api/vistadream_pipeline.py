from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from einops import rearrange
from icecream import ic
from jaxtyping import Bool, Float, UInt8
from monopriors.depth_utils import depth_edges_mask, depth_to_points
from monopriors.relative_depth_models import (
    RelativeDepthPrediction,
    get_relative_predictor,
)
from monopriors.relative_depth_models.base_relative_depth import BaseRelativePredictor
from PIL import Image
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from tqdm import tqdm

from vistadream.ops.connect import Smooth_Connect_Tool
from vistadream.ops.flux import FluxInpainting, FluxInpaintingConfig
from vistadream.ops.gs.basic import Frame, Gaussian_Scene
from vistadream.ops.gs.train import GS_Train_Tool
from vistadream.ops.trajs import _generate_trajectory
from vistadream.ops.utils import save_ply
from vistadream.ops.visual_check import Check
from vistadream.resize_utils import add_border_and_mask, process_image


@dataclass
class VistaDreamConfig:
    """
    Configuration for Flux Outpainting.
    """

    rr_config: RerunTyroConfig
    image_path: Path
    offload: bool = True
    num_steps: int = 25
    guidance: float = 30.0
    expansion_percent: float = 0.2
    n_frames: int = 5


class VistaDreamPipeline:
    """
    Pipeline for Flux Outpainting using VistaDream.
    """

    def __init__(self, config: VistaDreamConfig):
        self.config: VistaDreamConfig = config
        self.scene: Gaussian_Scene = Gaussian_Scene()
        self.flux_inpainter: FluxInpainting = FluxInpainting(FluxInpaintingConfig())
        self.predictor: BaseRelativePredictor = get_relative_predictor("MogeV1Predictor")(device="cuda")
        self.checkor: Check = Check()
        self.connector = Smooth_Connect_Tool()
        # Initialize rerun with the provided configuration
        self.shared_intrinsics: Intrinsics | None = None
        self.image_plane_distance: float = 0.01
        self.logged_cam_idx_list: list[int] = [0]

        ic("Pipeline initialized with configuration:", self.config)

    def __call__(self):
        # intialize rerun with the provided configuration
        self.setup_rerun()
        # outpaint -> depth prediction -> scene generation
        self._initialize()
        cam_T_world_traj: Float[np.ndarray, "n_frames 4 4"] = _generate_trajectory(
            None, self.scene, nframes=self.config.n_frames * 6
        )
        # log trajectory
        for i, cam_T_world in tqdm(
            enumerate(cam_T_world_traj, start=1), total=len(cam_T_world_traj), desc="Rendering trajectory"
        ):
            rr.set_time("time", sequence=i)
            cam_log_path: Path = self.parent_log_path / f"camera_{i}"
            pinhole_log_path: Path = cam_log_path / "pinhole"
            pinhole_param = PinholeParameters(
                name=f"camera_{i}",
                intrinsics=self.shared_intrinsics,
                extrinsics=Extrinsics(
                    cam_R_world=cam_T_world[:3, :3],
                    cam_t_world=cam_T_world[:3, 3],
                ),
            )
            frame = Frame(
                H=self.shared_intrinsics.height,
                W=self.shared_intrinsics.width,
                intrinsic=self.shared_intrinsics.k_matrix,
                cam_T_world=pinhole_param.extrinsics.cam_T_world,
            )
            frame: Frame = self.scene._render_for_inpaint(frame)
            rr.log(f"{pinhole_log_path}/rgb", rr.Image(frame.rgb, color_model=rr.ColorModel.RGB))
            rr.log(
                f"{pinhole_log_path}/mask", rr.Image(frame.inpaint.astype(np.uint8) * 255, color_model=rr.ColorModel.L)
            )
            log_pinhole(camera=pinhole_param, cam_log_path=cam_log_path, image_plane_distance=self.image_plane_distance)
            # inpaint every 5th frame TODO use vistadream metric for choosing instead of hardcoded 5
            if i % 5 == 0:
                # generate inpainted frame
                # first get frame convert to uint8 255, right now its float32 [0,1]
                self.logged_cam_idx_list.append(i)
                rr.send_blueprint(blueprint=self._create_blueprint())
                input_frame = (frame.rgb * 255).astype(np.uint8)
                mask = frame.inpaint.astype(np.uint8) * 255
                outpainted_image: Image.Image = self.flux_inpainter(rgb_hw3=input_frame, mask=mask)
                rr.log(f"{pinhole_log_path}/inpainted_rgb", rr.Image(outpainted_image, color_model=rr.ColorModel.RGB))
                # run depth prediction
                rgb_hw3: UInt8[np.ndarray, "H W 3"] = np.array(outpainted_image.convert("RGB"))
                rel_depth_pred: RelativeDepthPrediction = self.predictor.__call__(
                    rgb=rgb_hw3, K_33=self.shared_intrinsics.k_matrix
                )
                depth_hw: Float[np.ndarray, "h w"] = rel_depth_pred.depth
                prev_frame: Frame = self.scene.frames[-1] if self.scene.frames else None
                metric_dpt_connect = self.connector._affine_dpt_to_GS(
                    prev_frame.dpt, depth_hw, ~prev_frame.inpaint
                ).astype(np.float32)

                edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(metric_dpt_connect, threshold=0.1)
                masked_depth_hw: Float[np.ndarray, "h w"] = metric_dpt_connect * ~edges_mask

                rr.log(f"{pinhole_log_path}/depth", rr.DepthImage(masked_depth_hw, meter=1.0))
                # update frame with inpainted image and depth
                frame.dpt = masked_depth_hw
                frame.rgb = (np.array(outpainted_image.convert("RGB")) / 255.0).astype(np.float32)
                frame.inpaint = np.array(mask).astype(np.bool)
                # Create mask without edges for the inpainted frame
                mask_wo_edges: Bool[np.ndarray, "h w"] = np.array(mask).astype(np.bool_) & ~edges_mask
                frame.inpaint_wo_edge = mask_wo_edges
                self.scene._add_trainable_frame(frame)
                self.scene = GS_Train_Tool(self.scene, iters=512)(self.scene.frames)

        # generate the coarse scene
        save_dir: Path = Path("data/test_dir/")
        gf_path: Path = save_dir / "gf.ply"
        save_dir.mkdir(exist_ok=True, parents=True)
        self.checkor._render_video(self.scene, save_dir=save_dir)
        self._render_splats()
        save_ply(self.scene, gf_path)

    def _render_splats(self):
        # render 5times frames
        nframes: int = len(self.scene.frames) * 25
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

            # valid_dpt = dpt[dpt > 0.0]
            # _min = np.percentile(valid_dpt, 1)
            # _max = np.percentile(valid_dpt, 99)
            # dpt = (dpt - _min) / (_max - _min)
            # dpt = dpt.clip(0, 1)

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
            rr.log(f"{pinhole_log_path}/rgb", rr.Image(rgb, color_model=rr.ColorModel.RGB))
            log_pinhole(camera=pinhole_param, cam_log_path=cam_log_path, image_plane_distance=self.image_plane_distance)
            rr.log(f"{pinhole_log_path}/depth", rr.DepthImage(dpt))

    def _initialize(self):
        rr.set_time("time", sequence=0)
        cam_log_path: Path = self.parent_log_path / "camera_0"
        pinhole_path: Path = cam_log_path / "pinhole"
        input_image: Image.Image = Image.open(self.config.image_path).convert("RGB")
        # ensures image is correctly sized and processed
        input_image: Image.Image = process_image(input_image)

        # Auto-generate outpainting setup: user-controlled border expansion
        border_percent: float = (
            self.config.expansion_percent / 2.0
        )  # Convert to fraction per side (divide by 2 for each side)
        input_image, mask = add_border_and_mask(
            input_image,
            zoom_all=1.0,
            zoom_left=border_percent,
            zoom_right=border_percent,
            zoom_up=border_percent,
            zoom_down=border_percent,
            overlap=0,
        )
        # generate intrinsic for input image to be used for the rest of the pipeline
        input_rgb_hw3: UInt8[np.ndarray, "H W 3"] = np.array(input_image.convert("RGB"))
        input_rel_depth: RelativeDepthPrediction = self.predictor.__call__(rgb=input_rgb_hw3, K_33=None)
        self.shared_intrinsics: Intrinsics = Intrinsics(
            camera_conventions="RDF",
            fl_x=input_rel_depth.K_33[0, 0].item(),
            fl_y=input_rel_depth.K_33[1, 1].item(),
            cx=input_rel_depth.K_33[0, 2].item(),
            cy=input_rel_depth.K_33[1, 2].item(),
            width=input_rgb_hw3.shape[1],
            height=input_rgb_hw3.shape[0],
        )

        # extri = Extrinsics(
        #     world_R_cam=np.eye(3, dtype=np.float32),
        #     world_t_cam=np.zeros(3, dtype=np.float32),
        # )

        # input_depth_hw: Float[np.ndarray, "h w"] = input_rel_depth.depth
        # edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(input_depth_hw, threshold=0.1)
        # input_masked_depth_hw: Float[np.ndarray, "h w"] = input_depth_hw * ~edges_mask
        # mask_wo_edges: Bool[np.ndarray, "h w"] = np.array(mask).astype(np.bool_) & ~edges_mask

        # input_frame: Frame = Frame(
        #     H=input_rgb_hw3.shape[0],
        #     W=input_rgb_hw3.shape[1],
        #     rgb=input_rgb_hw3.astype(np.float32) / 255.0,  # Convert to [0,1] range
        #     dpt=input_masked_depth_hw,
        #     intrinsic=input_rel_depth.K_33,
        #     extrinsic=extri.world_T_cam,  # Identity matrix for world coordinates
        #     inpaint=np.ones_like(mask).astype(np.bool_),
        #     inpaint_wo_edge=np.ones_like(mask_wo_edges).astype(np.bool_),
        # )

        # input_frame.keep = True
        # self.scene._add_trainable_frame(input_frame, require_grad=True)

        rr.log(f"{pinhole_path}/rgb", rr.Image(input_image, color_model=rr.ColorModel.RGB))
        rr.log(f"{pinhole_path}/mask", rr.Image(mask, color_model=rr.ColorModel.L))

        outpainted_image: Image.Image = self.flux_inpainter(rgb_hw3=np.array(input_image), mask=np.array(mask))
        rr.log(f"{pinhole_path}/inpainted_rgb", rr.Image(outpainted_image, color_model=rr.ColorModel.RGB))
        rgb_hw3 = np.array(outpainted_image.convert("RGB"))
        rel_depth_pred: RelativeDepthPrediction = self.predictor.__call__(rgb=rgb_hw3, K_33=None)

        intri = Intrinsics(
            camera_conventions="RDF",
            fl_x=rel_depth_pred.K_33[0, 0].item(),
            fl_y=rel_depth_pred.K_33[1, 1].item(),
            cx=rel_depth_pred.K_33[0, 2].item(),
            cy=rel_depth_pred.K_33[1, 2].item(),
            width=rgb_hw3.shape[1],
            height=rgb_hw3.shape[0],
        )
        extri = Extrinsics(
            world_R_cam=np.eye(3, dtype=np.float32),
            world_t_cam=np.zeros(3, dtype=np.float32),
        )
        pinhole_params = PinholeParameters(name="pinhole", intrinsics=intri, extrinsics=extri)
        log_pinhole(camera=pinhole_params, cam_log_path=cam_log_path, image_plane_distance=self.image_plane_distance)

        depth_hw: Float[np.ndarray, "h w"] = rel_depth_pred.depth
        edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(depth_hw, threshold=0.1)
        masked_depth_hw: Float[np.ndarray, "h w"] = depth_hw * ~edges_mask

        mask_wo_edges: Bool[np.ndarray, "h w"] = np.array(mask).astype(np.bool_) & ~edges_mask

        rr.log(f"{pinhole_path}/depth", rr.DepthImage(masked_depth_hw, meter=1.0))
        rr.log(
            f"{pinhole_path}/mask_wo_edges",
            rr.Image(mask_wo_edges.astype(np.uint8) * 255, color_model=rr.ColorModel.L),
        )

        depth_1hw: Float[np.ndarray, "1 h w"] = rearrange(masked_depth_hw, "h w -> 1 h w")
        pts_3d: Float[np.ndarray, "h w 3"] = depth_to_points(depth_1hw, rel_depth_pred.K_33)

        rr.log(
            f"{self.parent_log_path}/point_cloud",
            rr.Points3D(
                positions=pts_3d.reshape(-1, 3),
                colors=rgb_hw3.reshape(-1, 3),
            ),
        )
        outpaint_frame: Frame = Frame(
            H=rgb_hw3.shape[0],
            W=rgb_hw3.shape[1],
            rgb=rgb_hw3.astype(np.float32) / 255.0,  # Convert to [0,1] range
            dpt=masked_depth_hw,
            intrinsic=rel_depth_pred.K_33,
            cam_T_world=extri.cam_T_world,  # Identity matrix for world coordinates
            inpaint=np.ones_like(mask).astype(np.bool_),
            inpaint_wo_edge=np.ones_like(mask_wo_edges).astype(np.bool_),
        )
        outpaint_frame.keep = True
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
            # *[f"- {self.parent_log_path}/camera_{i}/pinhole/depth" for i in self.logged_cam_idx_list],
            *[f"- {self.parent_log_path}/camera_{i}/pinhole/mask" for i in self.logged_cam_idx_list],
            *[f"- {self.parent_log_path}/camera_{i}/pinhole/rgb" for i in self.logged_cam_idx_list],
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
                        contents=["+ $origin/**", f"- {self.parent_log_path}/camera/pinhole/depth"],
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
                    name="Final Scene",
                ),
            ),
            collapse_panels=True,
        )
        return blueprint


def main(config: VistaDreamConfig) -> None:
    """
    Main function to run the Flux Outpainting process.
    """
    vd_pipeline = VistaDreamPipeline(config)
    vd_pipeline()
