import numpy as np
import pandas as pd
from pathlib import Path
from happypose.pose_estimators.megapose.config import LOCAL_DATA_DIR
from happypose.pose_estimators.megapose.inference.pose_estimator import PoseEstimator
from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.inference.utils import load_pose_models
from happypose.toolbox.inference.types import DetectionsType, ObservationTensor



# MEGAPOSE MODELS
NAMED_MODELS = {
    "megapose-1.0-RGB": {
        "coarse_run_id": "coarse-rgb-906902141",
        "refiner_run_id": "refiner-rgb-653307694",
        "requires_depth": False,
        "inference_parameters": {
            "n_refiner_iterations": 5,
            "n_pose_hypotheses": 1,
        },
    },
    "megapose-1.0-RGB-icp": {
        "coarse_run_id": "coarse-rgb-906902141",
        "refiner_run_id": "refiner-rgb-653307694",
        "requires_depth": True,
        "depth_refiner": "icp",
        "inference_parameters": {
            "n_refiner_iterations": 5,
            "n_pose_hypotheses": 1,
            "run_depth_refiner": False,
        },
    },
    "megapose-1.0-RGB-teaserpp": {
        "coarse_run_id": "coarse-rgb-906902141",
        "refiner_run_id": "refiner-rgbd-288182519",
        "requires_depth": True,
        "depth_refiner": "teaserpp",
        "inference_parameters": {
            "n_refiner_iterations": 3,
            "n_pose_hypotheses": 1,
            "run_depth_refiner": False,
        },
    },
    "megapose-1.0-RGB-multi-hypothesis-teaserpp": {
        "coarse_run_id": "coarse-rgb-906902141",
        "refiner_run_id": "refiner-rgb-653307694",
        "requires_depth": True,
        "depth_refiner": "teaserpp",
        "inference_parameters": {
            "n_refiner_iterations": 5,
            "n_pose_hypotheses": 3,
            "run_depth_refiner": False,
        },
    },
    "megapose-1.0-RGB-multi-hypothesis": {
        "coarse_run_id": "coarse-rgb-906902141",
        "refiner_run_id": "refiner-rgb-653307694",
        "requires_depth": False,
        "inference_parameters": {
            "n_refiner_iterations":5,
            "n_pose_hypotheses": 5,
        },
    },
    "megapose-1.0-RGB-multi-hypothesis-bis": {
        "coarse_run_id": "coarse-rgb-906902141",
        "refiner_run_id": "refiner-rgb-653307694",
        "requires_depth": False,
        "inference_parameters": {
            "n_refiner_iterations": 10,
            "n_pose_hypotheses": 5,
        },
    },
    "megapose-1.0-RGB-multi-hypothesis-icp": {
        "coarse_run_id": "coarse-rgb-906902141",
        "refiner_run_id": "refiner-rgb-653307694",
        "requires_depth": True,
        "depth_refiner": "icp",
        "inference_parameters": {
            "n_refiner_iterations": 5,
            "n_pose_hypotheses": 5,
            "run_depth_refiner": False,
        },
    },
}


def load_named_model(
    model_name: str,
    object_dataset: RigidObjectDataset,
    n_workers: int = 4,
    bsz_images: int = 128,
    models_root: Path = LOCAL_DATA_DIR / "megapose-models"
) -> PoseEstimator:
    model = NAMED_MODELS[model_name]

    renderer_kwargs = {
        "preload_cache": True,
        "split_objects": False,
        "n_workers": n_workers,
        "use_antialiasing": False,
    }

    coarse_model, refiner_model, mesh_db = load_pose_models(
        coarse_run_id=model["coarse_run_id"],
        refiner_run_id=model["refiner_run_id"],
        object_dataset=object_dataset,
        force_panda3d_renderer=True,
        renderer_kwargs=renderer_kwargs,
        models_root=models_root,
    )

    depth_refiner = None
    if model.get("depth_refiner", None) == "icp":
        from happypose.pose_estimators.megapose.inference.icp_refiner import ICPRefiner
        depth_refiner = ICPRefiner(
            mesh_db,
            refiner_model.renderer,
        )
    if model.get("depth_refiner", None) == "teaserpp":
        from happypose.pose_estimators.megapose.inference.teaserpp_refiner import TeaserppRefiner
        depth_refiner = TeaserppRefiner(
            mesh_db,
            refiner_model.renderer,
            use_farthest_point_sampling=False
        )

    pose_estimator = PoseEstimator(
        refiner_model=refiner_model,
        coarse_model=coarse_model,
        detector_model=None,
        depth_refiner=depth_refiner,
        bsz_objects=8,
        bsz_images=bsz_images,
    )
    return pose_estimator


def dets_2_happydets(bboxes, scores, labels) -> DetectionsType:
    N_dets = len(labels)
    infos = pd.DataFrame({
        "label": labels,
        "batch_im_id": [0 for _ in range(N_dets)],
        "instance_id": np.arange(N_dets),
        "score": scores.cpu(),
        # "scene_id": scores,
        # "view_id": [idx_sensor for _ in range(N_dets)],
    })
    return DetectionsType(infos=infos, bboxes=bboxes)


def create_pose_estimator_pylone(
        model_config: str, 
        object_label: str,
        mesh_path: str | Path, 
        device: str,
        SO3_grid_size_scale_down: int = 1,
        mesh_path_icp: str | None = None,
        object_label_icp: str | None = None,
    ) -> tuple[PoseEstimator, dict]:
    """_summary_

    Args:
        model_config (str): Name of the megapose config used 
        object_label (str): Label of the object mesh used by megapose. 
        mesh_path (str | Path): Path of the object mesh used by megapose.
        device (str): device used to run megapose
        SO3_grid_size_scale_down (int, optional): Factor scaling down the number of coarse renders (the higher, the faster but the less precise).. Defaults to 1.
        mesh_path_icp (str | None, optional): Path of the object mesh used by the icp step (slightly different, doesn't need texture). If None, no additional mesh added to the renderer collection.
        object_label_icp (str | None, optional): Label of the object used by the icp step, used when calling the renderer API. If None, ignored.

    Returns:
        tuple[PoseEstimator, dict]: _description_
    """

    ########
    # Init MEGAPOSE
    objects = [
        RigidObject(
            label=object_label, mesh_path=mesh_path, mesh_units="m"
        )
    ]
    if mesh_path_icp is not None:
        objects.append(
            RigidObject(
                label=object_label_icp, mesh_path=mesh_path_icp, mesh_units="m"
            )
        )
    object_dataset = RigidObjectDataset(objects=objects)

    pose_model_info = NAMED_MODELS[model_config]
    pose_model_info["bsz_objects"] = 32  # How many parallel refiners to run
    pose_model_info["bsz_images"] = 576  # How many images to push through coarse model
    n_workers = 16  # number of renderer workers8
    pose_estimator = load_named_model(model_config, object_dataset, n_workers)
    pose_estimator = pose_estimator.to(device)
    pose_estimator._SO3_grid = pose_estimator._SO3_grid[::SO3_grid_size_scale_down]

    return pose_estimator, pose_model_info

