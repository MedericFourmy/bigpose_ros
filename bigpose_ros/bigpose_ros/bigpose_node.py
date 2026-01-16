#!/usr/bin/env python3

import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import quaternion
import open3d as o3d
import trimesh  
import pinocchio as pin

from s2m2.s2m2 import load_model
from s2m2.config import S2M2_PRETRAINED_WEIGHTS_PATH

from happypose.toolbox.inference.types import ObservationTensor

import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from geometry_msgs.msg import TransformStamped, Transform, Point
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration
from ament_index_python.packages import get_package_share_directory

import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener, TransformBroadcaster, LookupException, ConnectivityException, ExtrapolationException
from cv_bridge import CvBridge

from bigpose_ros.s2m2_utils import get_disparity_map, depth_to_pointcloud
from bigpose_ros.detector import ZeroShotObjectDetector, pad_boxes
from bigpose_ros.megapose_utils import create_pose_estimator_pylone, dets_2_happydets
from bigpose_ros.icp_utils import create_o3d_poincloud_from_depth, orient_normals_toward_camera, crop_pcd_sphere, ICPConvergeCriteria, icp_registration_o3d
from bigpose_ros.render_utils import extract_np_from_renderings, DEFAULT_RENDER_PARAMS, render_ts, get_panda3d_ambient
from bigpose_msgs.srv import GetTransformStamped

# Automatically generated file
from bigpose_ros.bigpose_ros_parameters import bigpose_ros  # noqa: E402


class BigPoseNode(Node):
    def __init__(self):
        super().__init__('bigpose_node')

        try:
            self._param_listener = bigpose_ros.ParamListener(self)
            self._params = self._param_listener.get_params()
        except Exception as e:
            self.get_logger().error(str(e))
            raise e


        # Load detector model
        # ----------------------------
        self.detector = ZeroShotObjectDetector(self._params.detector_model_id, self._params.device)

        # ----------------------------
        # Load Megapose
        # ----------------------------
        package_share_directory = get_package_share_directory('bigpose_ros')
        self.model_path_obj = os.path.join(package_share_directory, self._params.megapose.mesh_megapose_relative_path)
        params_pose_est = {
            "model_config": self._params.megapose.megapose_model_config,
            "object_label": self._params.object_frame_id,
            "mesh_path": self.model_path_obj,
            "device": self._params.device,
            "SO3_grid_size_scale_down": self._params.megapose.SO3_grid_size_scale_down,
        }
        self.run_icp_with_different_mesh = False
        if self._params.icp.use_different_mesh_for_icp and len(self._params.icp.mesh_icp_relative_path) > 0:
            self.object_label_icp = self._params.object_frame_id+"_icp"
            self.mesh_path_icp = os.path.join(package_share_directory, self._params.icp.mesh_icp_relative_path)
            params_pose_est["mesh_path_icp"] = self.mesh_path_icp
            params_pose_est["object_label_icp"] = self.object_label_icp
            self.run_icp_with_different_mesh = True

        self.pose_estimator, self.pose_model_info = create_pose_estimator_pylone(**params_pose_est)
        self.renderer = self.pose_estimator.refiner_model.renderer
        self.light_datas = [get_panda3d_ambient()]

        # ----------------------------
        # Load s2m2 stereo depth model
        # ----------------------------
        assert Path(S2M2_PRETRAINED_WEIGHTS_PATH).exists(), f"S2M2 pretrained weights not found at {S2M2_PRETRAINED_WEIGHTS_PATH}, set S2M2_PRETRAINED_WEIGHTS_PATH env var to the correct path."
        self.model_s2m2 = load_model(
            S2M2_PRETRAINED_WEIGHTS_PATH,
            self._params.s2m2.model_type,
            self._params.s2m2.allow_negative,
            self._params.s2m2.num_refine,
        ).to(self._params.device).eval()
        if self._params.s2m2.torch_compile:
            # TODO: explore disk caching options
            # most of the compilation time happens 
            # when the model is first called, which can take > 10s
            # then, each call is 3-5x faster than non compiled version
            self.get_logger().info(f"Compiling s2m2 model...")
            self.model_s2m2.compile()
        self.get_logger().info(f"Loaded s2m2 model '{self._params.s2m2.model_type}' on device {self._params.device}")

        # ---------------------
        # Subscribers Realsense
        # ---------------------
        self.rgb_image_sub = Subscriber(self, Image, "color/image_raw")
        self.rgb_info_sub = Subscriber(self, CameraInfo, "color/camera_info")
        self.infra1_img_sub = Subscriber(self, Image, "infra1/image_rect_raw")
        self.infra1_info_sub = Subscriber(self, CameraInfo, "infra1/camera_info")
        self.infra2_img_sub = Subscriber(self, Image, "infra2/image_rect_raw")
        self.infra2_info_sub = Subscriber(self, CameraInfo, "infra2/camera_info")
        self.rgb_img_msg: Image | None = None
        self.rgb_info_msg: CameraInfo | None = None
        self.infra1_img_msg: Image | None = None
        self.infra1_info_msg: CameraInfo | None = None
        self.infra2_img_msg: Image | None = None
        self.infra2_info_msg: CameraInfo | None = None
        self.bridge = CvBridge()

        # -----------------------------------
        # Sync callack for rgb and infra imgs
        # -----------------------------------
        sync_rgb_subs = [self.rgb_image_sub, self.rgb_info_sub]
        self.ts = ApproximateTimeSynchronizer(sync_rgb_subs, queue_size=1, slop=0.05)
        self.ts.registerCallback(self.sync_rgb_callback)
        sync_infra_subs = [self.infra1_img_sub, self.infra1_info_sub, self.infra2_img_sub, self.infra2_info_sub]
        self.ts = ApproximateTimeSynchronizer(sync_infra_subs, queue_size=1, slop=0.05)
        self.ts.registerCallback(self.sync_infra_callback)

        # ---------------------------------
        # TF needed for the camera baseline
        # ---------------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.baseline = None

        # ---------------------------------
        # Publishers S2M2 depth (for debug)
        # ---------------------------------
        # TODO: make these absolute topics relative
        topic_depth_info = '/camera/camera/depth_s2m2/camera_info'
        topic_depth_image = '/camera/camera/depth_s2m2/depth'
        self.depth_info_pub = self.create_publisher(CameraInfo, topic_depth_info, 1)
        self.depth_pub = self.create_publisher(Image, topic_depth_image, 1)
        self.topic_s2m2_points = '/camera/camera/depth_s2m2/points'
        self.topic_pcd_icp = '/camera/camera/depth_s2m2/points_icp'
        self.pc_s2m2_pub = self.create_publisher(PointCloud2, self.topic_s2m2_points, 1)
        self.pc_icp_pub = self.create_publisher(PointCloud2, self.topic_pcd_icp, 1)
        self.timer_s2m2_debug = self.create_timer(0.5, self.publish_object_s2m2_debug)  # 10 Hz
        self.new_s2m2_depth_pub = True
        self.depth_s2m2 = None
        self.pcd_ct = None
        self.T_wc_s2m2 = None

        # ----------------------------------
        # Detect/Refine object pose services
        # ----------------------------------
        self.detect_object_srv = self.create_service(Trigger, 'bigpose_ros/detect', self.detect_object_callback)
        self.refine_object_srv = self.create_service(GetTransformStamped, 'bigpose_ros/refine', self.refine_object_callback)
        self.timer_obj_tf = self.create_timer(0.1, self.publish_object_tf_callback)  # 10 Hz
        self.tf_world_object: TransformStamped | None = None

        # --------------------
        # reading object model
        # --------------------
        mesh = trimesh.load(self.model_path_obj)
        self.mesh_radius = mesh.bounding_sphere.primitive.radius

        self.marker_pub = self.create_publisher(Marker, 'bigpose_ros/object_detection_marker', 10)
        self.timer_obj_marker = self.create_timer(0.5, self.publish_object_marker_callback)

        self.get_logger().info(f"Bigpose is ready!")

    def sync_infra_callback(self,
            infra1_img_msg: Image,
            infra1_info_msg: CameraInfo,
            infra2_img_msg: Image,
            infra2_info_msg: CameraInfo
        ):
        self.get_logger().info("INFRAS received")
        assert np.allclose(infra1_info_msg.k, infra2_info_msg.k), "Left and right images should have identical intrinsics"
        
        self.infra1_img_msg = infra1_img_msg
        self.infra1_info_msg = infra1_info_msg
        self.infra2_img_msg = infra2_img_msg
        self.infra2_info_msg = infra2_info_msg

    def sync_rgb_callback(self,
            rgb_img_msg: Image,
            rgb_info_msg: CameraInfo
        ):
        self.get_logger().info("RGB received")
        self.rgb_img_msg = rgb_img_msg
        self.rgb_info_msg = rgb_info_msg
        
    def detect_object_callback(self, request: Trigger.Request, response: Trigger.Response):
        self.get_logger().warn("detect_object_callback called")

        if self.rgb_img_msg is None:
            msg = "Did not receive first RGB frame, return."
            self.get_logger().warn(msg)
            response.message = msg
            response.success = False
            return response
        
        header_rgb = self.rgb_img_msg.header
        try:
            rgb_np = self.bridge.imgmsg_to_cv2(self.rgb_img_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return
        K_rgb = (self.rgb_info_msg.k).reshape((3,3))
        self.get_logger().warn(f"detect_object_callback {self._params.world_frame_id} -> {header_rgb.frame_id}")

        tf_stamped_wc = self.get_tf(self._params.world_frame_id, header_rgb.frame_id, header_rgb.stamp)
        if tf_stamped_wc is None: return
        self.get_logger().warn(f"detect_object_callback tf_stamped_wc:\n{tf_stamped_wc.transform}")

        # 2D detection
        PROMPT = "a black and gray object with holes"
        text_labels = [[PROMPT]]
        self.get_logger().warn("detect_object_callback running 2D detector...")
        results = self.detector.predict(rgb_np, text_labels)
        self.get_logger().warn("detect_object_callback detected!")
        nb_dets = len(results[0]["scores"])
        scores = results[0]["scores"]
        boxes = results[0]["boxes"]
        labels = [self._params.object_frame_id for _ in range(nb_dets)]

        # 3D pose estimation
        h, w, _ = rgb_np.shape
        boxes_pad = pad_boxes(boxes, w, h, pixpad=5)
        detections = dets_2_happydets(boxes_pad, scores, labels)
        obs = ObservationTensor.from_numpy(rgb_np, None, K_rgb).to(self._params.device)
        self.get_logger().warn("detect_object_callback running Megapose...")
        data_TCO_final, extra_data = self.pose_estimator.run_inference_pipeline(
            obs, detections=detections,
            n_refiner_iterations=self.pose_model_info["inference_parameters"]["n_refiner_iterations"],
            n_pose_hypotheses=self.pose_model_info["inference_parameters"]["n_pose_hypotheses"],
        )
        self.get_logger().warn("detect_object_callback Megapose done!")

        # TODO: add scoring + reject step
        # select the best pose
        T_co = data_TCO_final.poses[0].cpu().numpy()
        T_wc = transform_to_np_mat(tf_stamped_wc.transform)
        
        # compose forward kinematics and object estimation to get object in "world" frame
        T_wo = T_wc @ T_co

        tf_world_object = TransformStamped()
        tf_world_object.header = Header()
        tf_world_object.header.stamp = self.rgb_img_msg.header.stamp
        tf_world_object.header.frame_id = self._params.world_frame_id
        tf_world_object.child_frame_id = self._params.object_frame_id
        tf_world_object.transform = np_mat_to_transform(T_wo)
        self.tf_world_object = tf_world_object  # update attribute -> will be published on tf

        response.message = "Object detection successful!"
        response.success = True
        return response

    def refine_object_callback(self, request: GetTransformStamped.Request, response: GetTransformStamped.Response):
        self.get_logger().warn("refine_object_callback")

        if self.tf_world_object is None:
            self.get_logger().error("Object pose was not initialized, cannot refine!!")
            response.success = False
            response.message = "Object pose was not initialized, cannot refine!!"
            return response  

        infra1_frame = self.infra1_img_msg.header.frame_id
        infra2_frame = self.infra2_img_msg.header.frame_id
        img_time = self.infra1_img_msg.header.stamp
        if self.baseline is None:
            self.baseline = self.get_stereo_baseline(infra1_frame, infra2_frame, self.infra1_img_msg.header.stamp)
        if self.baseline is None:
            self.get_logger().error("Couldn't read stereo baseline, returning")
            response.success = False
            response.message = "Couldn't read stereo baseline, returning"
            return response

        # get current T_camera_object transform
        # -------------------------------------
        tf_world_infra1 = self.get_tf(self._params.world_frame_id, infra1_frame, img_time)
        if tf_world_infra1 is None:
            self.get_logger().error("Couldn't read tf_world_infra1 transform")
            response.success = False
            response.message = "Couldn't read tf_world_infra1 transform, returning"
            return response

        T_wo_init = transform_to_np_mat(self.tf_world_object.transform) 
        T_wc = transform_to_np_mat(tf_world_infra1.transform)
        T_co_init = inverse_se3(T_wc) @ T_wo_init

        # S2M2 depth prediction
        # ---------------------
        infra1 = self.bridge.imgmsg_to_cv2(self.infra1_img_msg, desired_encoding='mono8')
        infra2 = self.bridge.imgmsg_to_cv2(self.infra2_img_msg, desired_encoding='mono8')
        assert infra1.shape == infra2.shape
        h, w = infra1.shape 
        disp = get_disparity_map(self.model_s2m2, infra1, infra2, self._params.device)  # (H,W), f32
        fx = self.infra1_info_msg.k[0]
        depth = self.baseline * fx / disp  # metric depth
        depth = depth.cpu().numpy()  # fp32

        # Open3D depth refinement
        # -----------------------
        voxel_size = self._params.icp.voxel_size
        dist_threshold = voxel_size*self._params.icp.dist_thresh_factor

        # create point cloud from measured depth
        K_infra1 = self.infra1_info_msg.k.reshape((3,3))
        pcd_ct = create_o3d_poincloud_from_depth(depth, K_infra1)
        pcd_ct = crop_pcd_sphere(pcd_ct, center=T_co_init[:3,3], radius=self.mesh_radius, margin=self._params.icp.margin_sphere_crop)
        pcd_ct.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(3.0*voxel_size, 40))

        # create point cloud from object model (render then backproject)
        if self.run_icp_with_different_mesh:
            mesh_id = self.object_label_icp
        else:
            mesh_id = self._params.object_frame_id
        renderings = self.renderer.render([mesh_id], render_ts(T_co_init), render_ts(K_infra1), [self.light_datas], (h, w), **DEFAULT_RENDER_PARAMS)
        ren = extract_np_from_renderings(renderings, 0)
        pcd_cp = create_o3d_poincloud_from_depth(ren["depth"], K_infra1)
        pcd_cp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(3.0*voxel_size, 40))
        pcd_cp = orient_normals_toward_camera(pcd_cp)

        # ICP refinement
        pcd_cp = pcd_cp.voxel_down_sample(voxel_size=voxel_size)
        pcd_ct = pcd_ct.voxel_down_sample(voxel_size=voxel_size)
        icp_res_ct_cp = icp_registration_o3d(pcd_cp, pcd_ct, np.eye(4), dist_threshold, self._params.icp.icp_method, ICPConvergeCriteria())
        T_ct_cp_icp = icp_res_ct_cp.transformation

        # ICP outlier rejection
        dt_norm = np.linalg.norm(T_ct_cp_icp[:3,3])
        dr_norm_deg = np.rad2deg(np.linalg.norm(pin.log3(T_ct_cp_icp[:3,:3])))
        if dt_norm > self._params.icp.reject.translation_th:
            msg = f"ICP refinement exceeded translation threshold {dt_norm} > {self._params.icp.reject.translation_th}"
            self.get_logger().error(msg)
            response.success = False
            response.message = msg
            return response
        if dr_norm_deg > self._params.icp.reject.rotation_th:
            msg = f"ICP refinement exceeded rotation threshold {dr_norm_deg} > {self._params.icp.reject.rotation_th}"
            self.get_logger().error(msg)
            response.success = False
            response.message = msg
            return response


        T_co_ref = T_ct_cp_icp @ T_co_init

        # update tf object pose and set the service response
        # --------------------------------------------------
        T_wo_refined = T_wc @ T_co_ref

        tf_world_object = TransformStamped()
        tf_world_object.header = Header()
        tf_world_object.header.stamp = self.infra1_img_msg.header.stamp
        tf_world_object.header.frame_id = self._params.world_frame_id
        tf_world_object.child_frame_id = self._params.object_frame_id
        tf_world_object.transform = np_mat_to_transform(T_wo_refined)

        self.tf_world_object = tf_world_object  # update attribute -> will be published on tf
        response.transform = tf_world_object  # also placed in the response
        response.success = True
        response.message = "Refinement successful!"

        self.depth_s2m2 = depth  # store for debug publishing.
        self.pcd_ct = pcd_ct  # store for debug publishing.
        self.T_wc_s2m2 = T_wc  # store for debug publishing.
        self.new_s2m2_depth_pub = True  # flag to recompute pointcloud for debug publishing

        return response

    def get_stereo_baseline(self, left_frame: str, right_frame: str, img_time: rclpy.time.Time) -> float:
        """
        Returns the baseline (meters) as the absolute translation along x
        between left and right camera frames.
        """
        tf = self.get_tf(left_frame, right_frame, img_time)
        if tf is None:
            return None
        baseline = abs(tf.transform.translation.x)
        return baseline

    def get_tf(self, target_frame: str, infra2_frame: str, img_time: rclpy.time.Time) -> TransformStamped:
        """
        Returns the tf transform from target_frame to infra2_frame.
        """
        try:
            # lookup the transform from left → right
            return self.tf_buffer.lookup_transform(
                target_frame=target_frame,
                source_frame=infra2_frame,
                time=img_time,
                timeout=rclpy.duration.Duration(seconds=0.2)
            )
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Failed to lookup transform {target_frame} → {infra2_frame}: {e}")
            return None
        
    def publish_object_tf_callback(self):
        if self.tf_world_object is not None:
            # update timestamp of object transform using
            # latest received RGB frame timestamp
            self.tf_world_object.header.stamp = self.infra1_img_msg.header.stamp
            self.tf_broadcaster.sendTransform(self.tf_world_object)

    def publish_object_s2m2_debug(self):
        if self.depth_s2m2 is None or self.pcd_ct is None: return
        if self.pc_s2m2_pub.get_subscription_count() == 0: return

        # ----------------------------------------
        # Publish point cloud from S2M2 depth
        # ----------------------------------------
        if self.new_s2m2_depth_pub:
            self.get_logger().info("Recomputing pointcloud for S2M2 depth debug publishing")
                
            k = self.infra1_info_msg.k  # stored in row-major order
            fx, fy, cx, cy = k[0], k[4], k[2], k[5]
            points = depth_to_pointcloud(self.depth_s2m2, fx, fy, cx, cy)
            # Remove invalid points (inf or NaN or zero depth)
            points = points.reshape(-1,3)
            valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
            points = points[valid]   # (N, 3)

            # Transform points to world frame
            R_wc = self.T_wc_s2m2[:3, :3]
            t_wc = self.T_wc_s2m2[:3, 3]
            points_s2m2_world = (R_wc @ points.T).T + t_wc

            depth_s2m2_header = self.infra1_info_msg.header
            depth_s2m2_header.frame_id = self._params.world_frame_id
            self.pc_s2m2_msg = pc2.create_cloud_xyz32(depth_s2m2_header, points_s2m2_world)
            
            # Also create point cloud from ICP refined depth
            self.pcd_ct.transform(self.T_wc_s2m2)  # to world frame
            self.pc_icp_msg = pc2.create_cloud_xyz32(depth_s2m2_header, np.asarray(self.pcd_ct.points))

        # publish point cloud every time
        self.new_s2m2_depth_pub = False
        self.pc_s2m2_pub.publish(self.pc_s2m2_msg)
        self.pc_icp_pub.publish(self.pc_icp_msg)

    def publish_object_marker_callback(self):
        if self.tf_world_object is not None:
            if self.run_icp_with_different_mesh:
                mesh_path_marker = "file://"+self.mesh_path_icp
            else:
                mesh_path_marker = "file://"+self.model_path_obj

            marker_object_mesh = make_mesh_marker(mesh_path_marker, self.tf_world_object)
            self.marker_pub.publish(marker_object_mesh)


def transform_to_np_mat(transform: Transform) -> np.ndarray:
    """
    Convert a ROS 2 Transform to a 4x4 NumPy matrix.

    Args:
        transform: The ROS 2 Transform message.

    Returns:
        A 4x4 NumPy matrix representing the transform.
    """
    t = transform.translation
    q = transform.rotation

    # Create a 4x4 matrix from translation and quaternion
    mat = np.eye(4)
    mat[:3, 3] = [t.x, t.y, t.z]
    # Convert quaternion to rotation matrix
    mat[:3,:3] = quaternion.as_rotation_matrix(np.quaternion(q.w, q.x, q.y, q.z))
    return mat


def inverse_se3(T: np.ndarray):
    T_inv = np.eye(4)
    T_inv[:3,:3] = T[:3,:3].T
    T_inv[:3,3] = -T[:3,:3].T @ T[:3,3]
    return T_inv


def np_mat_to_transform(mat: np.ndarray) -> Transform:
    """
    Convert a 4x4 NumPy matrix to a ROS 2 Transform.

    Args:
        mat: The 4x4 NumPy matrix.
        header_frame: The header.frame_id for the Transform.
        child_frame: The child_frame_id for the Transform.
        stamp: Optional timestamp for the header.

    Returns:
        A ROS 2 Transform message.
    """
    transform = Transform()

    # Extract translation
    transform.translation.x = mat[0, 3]
    transform.translation.y = mat[1, 3]
    transform.translation.z = mat[2, 3]

    # Extract rotation as quaternion
    R = mat[:3,:3]
    q = quaternion.from_rotation_matrix(R)
    transform.rotation.x = q.x
    transform.rotation.y = q.y
    transform.rotation.z = q.z
    transform.rotation.w = q.w

    return transform


def make_mesh_marker(mesh_path: str, tf: TransformStamped):
    marker = Marker()
    marker.header.stamp = tf.header.stamp
    marker.header.frame_id = tf.header.frame_id

    marker.ns = "object_mesh"
    marker.id = 0
    marker.type = Marker.MESH_RESOURCE
    marker.mesh_resource = mesh_path     # e.g. "package://my_pkg/meshes/object.stl"
    marker.mesh_use_embedded_materials = True

    # Pose from TransformStamped
    marker.pose.position = Point(
        x=tf.transform.translation.x,
        y=tf.transform.translation.y,
        z=tf.transform.translation.z
    )
    marker.pose.orientation = tf.transform.rotation

    # Scale (must be non-zero)
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0

    # Lifetime (0 = forever)
    marker.lifetime = Duration(sec=0, nanosec=0)

    # Color (ignored if using embedded materials)
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 1.0
    marker.color.a = 0.8

    marker.action = Marker.ADD
    return marker


def main(args=None):
    rclpy.init(args=args)
    node = BigPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



# [bigpose_node-1] [INFO] [1768571180.371182504] [bigpose_node]: T_wc: 
# [bigpose_node-1] [[-0.11663425 -0.59962013  0.79173995 -0.31597204]
# [bigpose_node-1]  [-0.05234108 -0.79236194 -0.60780175  0.50925493]
# [bigpose_node-1]  [ 0.99179477 -0.11233103  0.06103178  0.83184457]
# [bigpose_node-1]  [ 0.          0.          0.          1.        ]]
# [bigpose_node-1] [INFO] [1768571180.371415796] [bigpose_node]: T_wo: 
# [bigpose_node-1] [[ 0.99603363 -0.08540442 -0.02496201  0.48358051]
# [bigpose_node-1]  [ 0.0847467   0.99605492 -0.02631702 -0.05082718]
# [bigpose_node-1]  [ 0.02711112  0.02409719  0.99934194  0.73766623]
# [bigpose_node-1]  [ 0.          0.          0.          1.        ]]
# [bigpose_node-1] [INFO] [1768571180.371654881] [bigpose_node]: T_co_init_TF: 
# [bigpose_node-1] [[-0.0937187  -0.01827404  0.99543099 -0.1573455 ]
# [bigpose_node-1]  [-0.66743729 -0.74073267 -0.07643678 -0.02506091]
# [bigpose_node-1]  [ 0.73874506 -0.67155132  0.05722373  0.96770871]
# [bigpose_node-1]  [ 0.          0.          0.          1.        ]]
# [bigpose_node-1] [INFO] [1768571180.371878541] [bigpose_node]: T_co_init: 
# [bigpose_node-1] [[-0.11617164  0.00447016 -0.02475719 -0.40389481]
# [bigpose_node-1]  [-0.05081583 -0.78923601  0.00295622 -0.015629  ]
# [bigpose_node-1]  [ 0.02146496 -0.01464631  0.06099162  0.37541662]
# [bigpose_node-1]  [ 0.          0.          0.          1.        ]]
# [bigpose_node-1] [INFO] [1768571180.372099856] [bigpose_node]: DIFF: 
# [bigpose_node-1] [[ 0.06066097  0.51552671  0.04540438 -0.42074178]
# [bigpose_node-1]  [ 0.02534905  0.59436696 -0.04269635  0.39527347]
# [bigpose_node-1]  [-0.11052835  0.06393828 -0.02137987 -0.28003693]
# [bigpose_node-1]  [ 0.          0.          0.          1.        ]]