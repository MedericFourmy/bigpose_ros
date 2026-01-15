#!/usr/bin/env python3

import os
import math
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import quaternion
import open3d as o3d
import trimesh  

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
            "model_config": self._params.megaposREADME.mdig,
            "object_label": self._params.object_frame_id,
            "mesh_path": self.model_path_obj,
            "device": self._params.device,
            "SO3_grid_size_scale_down": self._params.megapose.SO3_grid_size_scale_down,
        }
        self.run_icp_with_different_mesh = False
        if self._params.icp.use_different_mesh_for_icp and len(self._params.icp.mesh_icp_relative_path) > 0:
            self.object_label_icp = self._params.object_frame_id+"_icp"
            params_pose_est["mesh_path_icp"] = os.path.join(package_share_directory, self._params.icp.mesh_icp_relative_path)
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

        # ---------------------------------
        # Publishers S2M2 depth (for debug)
        # ---------------------------------
        # TODO: deal with these hardcoded topics
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
        self.tf_stamped_wo: TransformStamped | None = None

        # --------------------
        # reading object model
        # --------------------
        mesh = trimesh.load(self.model_path_obj)
        self.mesh_radius = mesh.bounding_sphere.primitive.radius

        self.marker_pub = self.create_publisher(Marker, 'megapose_detection_marker', 10)
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

        tf_stamped_wo = TransformStamped()
        tf_stamped_wo.header = Header()
        tf_stamped_wo.header.stamp = self.rgb_img_msg.header.stamp
        tf_stamped_wo.header.frame_id = self._params.world_frame_id
        tf_stamped_wo.child_frame_id = self._params.object_frame_id
        tf_stamped_wo.transform = np_mat_to_transform(T_wo)
        self.tf_stamped_wo = tf_stamped_wo  # update attribute -> will be published on tf

        response.message = "Object detection successful!"
        response.success = True
        return response

    def refine_object_callback(self, request: GetTransformStamped.Request, response: GetTransformStamped.Response):
        self.get_logger().warn("refine_object_callback")

        if self.tf_stamped_wo is None:
            self.get_logger().error("Object pose was not initialized, cannot refine!!")
            response.success = False
            response.message = "Object pose was not initialized, cannot refine!!"
            return response  

        infra1_frame = self.infra1_img_msg.header.frame_id
        infra2_frame = self.infra2_img_msg.header.frame_id
        img_time = self.infra1_img_msg.header.stamp
        baseline = self.get_stereo_baseline(infra1_frame, infra2_frame, self.infra1_img_msg.header.stamp)
        if baseline is None:
            self.get_logger().error("Couldn't read stereo baseline, returning")
            response.success = False
            response.message = "Couldn't read stereo baseline, returning"
            return response

        # get current T_camera_object transform
        # -------------------------------------
        tf_infra1_object = self.get_tf(infra1_frame, self._params.object_frame_id, img_time)
        tf_world_infra1 = self.get_tf(self._params.world_frame_id, infra1_frame, img_time)
        if tf_infra1_object is None:
            self.get_logger().error("Couldn't read tf_infra1_object transform")
        if tf_world_infra1 is None:
            self.get_logger().error("Couldn't read tf_world_infra1 transform")
        if tf_infra1_object is None or tf_world_infra1 is None:
            response.success = False
            response.message = "Couldn't read required transforms, returning"
            return response

        T_co_init = transform_to_np_mat(tf_infra1_object.transform)
        T_wc = transform_to_np_mat(tf_world_infra1.transform)
        # T_wo_init = transform_to_np_mat(self.tf_stamped_wo.transform)  # for debug?

        # S2M2 depth prediction
        # ---------------------
        infra1 = self.bridge.imgmsg_to_cv2(self.infra1_img_msg, desired_encoding='mono8')
        infra2 = self.bridge.imgmsg_to_cv2(self.infra2_img_msg, desired_encoding='mono8')
        assert infra1.shape == infra2.shape
        h, w = infra1.shape 
        disp = get_disparity_map(self.model_s2m2, infra1, infra2, self._params.device)  # (H,W), f32
        fx = self.infra1_info_msg.k[0]
        depth = baseline * fx / disp  # metric depth
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
        T_co_ref = T_ct_cp_icp @ T_co_init

        # update tf object pose and set the service response
        # --------------------------------------------------
        T_wo_refined = T_wc @ T_co_ref

        tf_stamped_wo = TransformStamped()
        tf_stamped_wo.header = Header()
        tf_stamped_wo.header.stamp = self.infra1_img_msg.header.stamp
        tf_stamped_wo.header.frame_id = self._params.world_frame_id
        tf_stamped_wo.child_frame_id = self._params.object_frame_id
        tf_stamped_wo.transform = np_mat_to_transform(T_wo_refined)

        self.tf_stamped_wo = tf_stamped_wo  # update attribute -> will be published on tf
        response.transform = tf_stamped_wo  # also placed in the response
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
        if self.tf_stamped_wo is not None:
            # update timestamp of object transform using
            # latest received RGB frame timestamp
            self.tf_stamped_wo.header.stamp = self.infra1_img_msg.header.stamp
            self.tf_broadcaster.sendTransform(self.tf_stamped_wo)

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
        if self.tf_stamped_wo is not None:
            marker_object_mesh = make_mesh_marker("file://"+self.model_path_obj, self.tf_stamped_wo)
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
    marker.color.a = 1.0

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
