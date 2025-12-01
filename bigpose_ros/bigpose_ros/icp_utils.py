from dataclasses import dataclass

import numpy as np
import trimesh
import open3d as o3d


ICP_METHODS = {
    "point2point": o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    "point2plane": o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    "generalized": o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(epsilon=0.001),
    "colored": o3d.pipelines.registration.TransformationEstimationForColoredICP(),
}
        
@dataclass
class ICPConvergeCriteria:
    max_iteration: int = 100
    relative_fitness: float = 1e-6
    relative_rmse: float =  1e-6


def icp_registration_o3d(
        pcd_src: o3d.geometry.PointCloud,
        pcd_tgt: o3d.geometry.PointCloud,
        T_tgt_src_init: np.ndarray,
        th_corresp: float,
        method_name: str,
        params: ICPConvergeCriteria
    ):
    """
    pcd_src: source point cloud
    pcd_tgt: target point cloud, requires normals estimates
    T_tgt_src_init: initial guess for the T_tgt_src transformation
    th_corresp: coresponds to max_correspondence_distance
    params: object storing all ICP input params

    return: refined transformation T_tgt_src
    """
    assert method_name in ICP_METHODS, f"{method_name} not in {ICP_METHODS}"
    
    icp_method = ICP_METHODS[method_name]
    icp_convergence_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=params.max_iteration,
        relative_rmse=params.relative_rmse,
        relative_fitness=params.relative_fitness,
    )

    # PointToPlane ICP requires normals for the target point cloud 
    if method_name == "point2plane":
        assert len(pcd_tgt.normals) > 0

    icp_sol = o3d.pipelines.registration.registration_icp(
        pcd_src, 
        pcd_tgt, 
        th_corresp, 
        T_tgt_src_init,
        icp_method, 
        icp_convergence_criteria
    )
    # icp_sol.transformation: T_tgt_src
    # icp_sol.fitness, which measures the overlapping area (# of inlier correspondences / # of points in target). The higher the better.
    # icp_sol.inlier_rmse, which measures the RMSE of all inlier correspondences. The lower the better.
    # icp_sol.correspondence_set
    return icp_sol


def create_o3d_poincloud_from_depth(
        depth: np.ndarray, 
        K: np.ndarray, 
        mask: np.ndarray | None = None,
        rgb: np.ndarray | None = None, 
        depth_scale: float = 1.0, 
        depth_trunc: float = 1e6,
        convert_rgb_to_intensity: bool = True,
    ) -> o3d.geometry.PointCloud:
    """
    Create an open3d PointCloud from depth, intrinsics and optionally rgb and mask.

    Contrary to open3d, applies depth_scale whatever the input detph dtype. 
    By default, do not rescale and do not truncate
    
    depth: (h,w) depth image, should be of types np.uint16, or np.float32
    K: (3,3) intrinsics matrix
    depth_scale: factor by which the depth image values should be divided to get a pointcloud correct units (e.g. meters)
    rgb: (h,w,3) if passed, the point cloud will be "colored"
    mask: (h,w) depth pixels that should be kept 
    """

    # convert depth map to a dtype supported by open3d
    if depth.dtype not in [np.uint16, np.float32]:
        depth = depth.astype(np.float32)
    
    h, w = depth.shape[:2]
    fx, fy, cx, cy = K_to_fxfycxcy(K)
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    
    if mask is not None:
        depth = np.where(mask > 0, depth, 0)
    
    if rgb is None:
        # create_from_depth_image ignores depth_scale if passed depth image is np.float32 type
        # -> fix this inconsistency
        if depth.dtype == np.float32 and depth_scale != 1.0:
            depth = depth / depth_scale  # copy and rescale
        depth_o3d = o3d.geometry.Image(depth) 
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, o3d_intr, depth_scale=depth_scale, depth_trunc=depth_trunc)
    else:
        assert rgb.shape[:2] == depth.shape[:2], "rgb and depth images should be aligned and have the same resolution."
        # depth images are rescaled by create_from_color_and_depth whatever their type
        depth_o3d = o3d.geometry.Image(depth) 
        rgb_o3d = o3d.geometry.Image(rgb) 
        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, depth_scale=depth_scale, depth_trunc=depth_trunc, convert_rgb_to_intensity=convert_rgb_to_intensity)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, o3d_intr)
    return pcd


def K_to_fxfycxcy(K: np.ndarray) -> np.ndarray:
    assert K.shape == (3,3)
    return K[0][0], K[1][1], K[0][2], K[1][2]


def orient_normals_toward_camera(pcd, camera_direction=np.array([0, 0, 1])):
    """Orient normals to face the camera"""
    normals = np.asarray(pcd.normals)
    dots = np.sum(normals * camera_direction, axis=1)
    normals[dots > 0] *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def crop_pcd_sphere(
        pcd: o3d.geometry.PointCloud, 
        center: np.ndarray, 
        radius: float, 
        margin: float
    ) -> o3d.geometry.PointCloud:
    """
    Crop the point cloud using a square axis aligned bounding box.
    """
    min_bound = center - margin*radius
    max_bound = center + margin*radius
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return pcd.crop(aabb)


def trimesh2open3d(trimesh_mesh: trimesh.Trimesh, sample_count: int, colored=True):
    obj_pcd = o3d.geometry.PointCloud()
    samples = trimesh.sample.sample_surface(trimesh_mesh, sample_count, sample_color=colored)
    sampled_points = samples[0]
    obj_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    if colored:
        assert len(samples) == 3
        sampled_colors = samples[2]
        obj_pcd.colors = o3d.utility.Vector3dVector(sampled_colors[:, :3] / 255)

    return obj_pcd

