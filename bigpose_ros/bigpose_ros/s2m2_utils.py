#!/usr/bin/env python3

import os
import math
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import torch
import torch.nn.functional as F
import numpy as np


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import torch._dynamo
torch._dynamo.config.verbose=True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException




def image_pad(img: torch.Tensor, factor: int):
    """
    Pad img so that its dimensions are a multiple of `factor`.

    img: torch.Tensor (B,C,H,W), img needing to be padded
    factor: factor by which the new img dimensions should be divisible

    Out: torch.Tensor (B,C,H_new,W_new), padded image with H_new and W_new divisible by factor 
    """
    with torch.no_grad():
        B, C, H, W = img.shape
        assert H > factor, W > factor

        H_new = math.ceil(H / factor) * factor
        W_new = math.ceil(W / factor) * factor
        
        # pad the original img
        pad_h = H_new - H
        pad_w = W_new - W
        pad = (
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2
        )
        padded = F.pad(img, pad, "constant", 0)

        # smooth the whole padded img
        down = F.adaptive_avg_pool2d(padded, (H // factor, W // factor))
        img_smooth_pad = F.interpolate(down, (H_new, W_new),
                            mode="bilinear", align_corners=False)

        # put original img back in non-padded area
        h_s, w_s = pad_h // 2, pad_w // 2
        img_smooth_pad[..., h_s:h_s+H, w_s:w_s+W] = img
        return img_smooth_pad


def image_crop(img: torch.Tensor, H_new: int, W_new: int) -> torch.Tensor:
    """
    Center crop the image to a new dimension
    """
    with torch.no_grad():
        H, W = img.shape[-2], img.shape[-1]
        if H_new > H or W_new > W:
            raise ValueError(f"Crop size {(H_new, W_new)} exceeds image size {(H, W)}")
        top  = (H - H_new) // 2
        left = (W - W_new) // 2
        return img[..., top:top + H_new, left:left + W_new]


def depth_to_pointcloud(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    """
    depth_mm: (H, W) depth in meters
    returns Nx3 float32 array (X,Y,Z) in meters
    """
    h, w = depth.shape

    # pixel coordinate grid
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))

    # back project to 3D points
    points = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=depth.dtype)
    points[:,:,0] = (xs - cx) * depth / fx  # X
    points[:,:,1] = (ys - cy) * depth / fy  # Y
    points[:,:,2] = depth  # Z
    return points


