#!/usr/bin/env python3

import os
import math
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import torch
import numpy as np

# S2M2
from s2m2.s2m2 import load_model
from s2m2.config import S2M2_PRETRAINED_WEIGHTS_PATH

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_srvs.srv import Trigger
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener

from bigpose_ros.s2m2_utils import image_crop, image_pad, depth_to_pointcloud
from bigpose_msgs.srv import GetPose



class BigPoseNode(Node):
    def __init__(self):
        super().__init__('bigpose_node')
        # ----------------------------
        # Load s2m2 stereo depth model
        # ----------------------------
        model_type = "S"  # select model type: S,M,L,XL
        allow_negative = False  # TODO: figure out what this is
        num_refine = 3
        self.device = "cuda"
        # most of the compilation time happens 
        # when the model is first called, which can take > 10s
        # then, each call is 3-5x faster than non compiled version
        # TODO: explore disk caching options
        torch_compile = False 
        self.model_s2m2 = load_model(
            S2M2_PRETRAINED_WEIGHTS_PATH,
            model_type,
            allow_negative,
            num_refine,
        ).to(self.device).eval()
        if torch_compile:
            self.get_logger().info(f"Compiling s2m2 model...")
            self.model_s2m2.compile()
        # self.model_s2m2 = torch.compile(self.model_s2m2)
        self.get_logger().info(f"Loaded s2m2 model '{model_type}' on device {self.device}")

        self.bridge = CvBridge()

        # -----------------------------
        # Subscribers Realsense (message_filters)
        # -----------------------------
        topic_left_image_rect = '/camera/camera/infra1/image_rect_raw'
        topic_left_info = '/camera/camera/infra1/camera_info'
        topic_right_image_rect = '/camera/camera/infra2/image_rect_raw'
        topic_right_info = '/camera/camera/infra2/camera_info'
        self.infra1_img_sub = Subscriber(self, Image, topic_left_image_rect)
        self.infra1_info_sub = Subscriber(self, CameraInfo, topic_left_info)
        self.infra2_img_sub = Subscriber(self, Image, topic_right_image_rect)
        self.infra2_info_sub = Subscriber(self, CameraInfo, topic_right_info)
        # stored in place
        self.infra1_img_msg = None
        self.infra1_info_msg = None
        self.infra2_img_msg = None
        self.infra2_info_msg = None

        # tf needed to get the baseline
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.baseline = None

        # -----------------------------
        # Approximate Time Sync
        # -----------------------------
        sync_topics = [self.infra1_img_sub, self.infra1_info_sub, self.infra2_img_sub, self.infra2_info_sub]
        self.ts = ApproximateTimeSynchronizer(sync_topics, queue_size=1, slop=0.05)
        self.ts.registerCallback(self.sync_callback)

        # ---------------------------------
        # Publishers S2M2 depth (for debug)
        # ---------------------------------
        topic_depth_info = '/camera/camera/depth_s2m2/camera_info'
        topic_depth_image = '/camera/camera/depth_s2m2/depth'
        self.depth_info_pub = self.create_publisher(CameraInfo, topic_depth_info, 1)
        self.depth_pub = self.create_publisher(Image, topic_depth_image, 1)
        self.topic_s2m2_points = '/camera/camera/depth_s2m2/points'
        self.pc_pub = self.create_publisher(PointCloud2, self.topic_s2m2_points, 1)

        # ----------------------------------
        # Detect/Refine object pose services
        # ----------------------------------
        self.detect_object_srv = self.create_service(Trigger, 'detect', self.detect_object_callback)

    def sync_callback(self,
            infra1_img_msg: Image,
            infra1_info_msg: CameraInfo,
            infra2_img_msg: Image,
            infra2_info_msg: CameraInfo
        ):
        assert np.allclose(infra1_info_msg.k, infra2_info_msg.k), "Left and right images should have identical intrinsics"
        
        left_frame = infra1_img_msg.header.frame_id
        right_frame = infra2_img_msg.header.frame_id

        baseline = self.get_stereo_baseline(left_frame, right_frame, infra1_img_msg.header.stamp)
        if baseline is None:
            self.get_logger().warn("Baseline unavailable — skipping this frame")
            return
        
    def detect_object_callback(self, request: Trigger.Request, response: Trigger.Response):
        self.get_logger().warn("detect_object_callback")
        pass
        # TODO:

    def refine_object_callback(self, request: GetPose.Request, response: GetPose.Response):
        self.get_logger().warn("refine_object_callback")
        pass
        # TODO:

def main(args=None):
    rclpy.init(args=args)
    node = BigPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
