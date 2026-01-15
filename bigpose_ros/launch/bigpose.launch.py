
import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("bigpose_ros")

    params_file = os.path.join(
        pkg_share,
        "config",
        "params.yaml"
    )

    rviz_config_file = PathJoinSubstitution(
        [
            FindPackageShare("bigpose_ros"),
            "rviz",
            "debug.rviz",
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "camera_namespace",
            default_value="camera",
            description="Namespace for the camera"
        ),
        DeclareLaunchArgument(
            "camera_name",
            default_value="camera",
            description="Name of the camera"
        ),
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="false",
            description="Launch rviz2 with debug configuration"
        ),
        DeclareLaunchArgument(
            "rviz_config_path",
            default_value=rviz_config_file,
            description="Path to a file containing RViz view configuration.",
        ),
        Node(
            package="bigpose_ros",
            executable="bigpose_node",
            name="bigpose_node",
            output="screen",
            parameters=[params_file],
            namespace=LaunchConfiguration("camera_namespace"),
            remappings=[
                ("color/image_raw", [LaunchConfiguration("camera_name"), "/color/image_raw"]),
                ("color/camera_info", [LaunchConfiguration("camera_name"), "/color/camera_info"]),
                ("infra1/image_rect_raw", [LaunchConfiguration("camera_name"), "/infra1/image_rect_raw"]),
                ("infra1/camera_info", [LaunchConfiguration("camera_name"), "/infra1/camera_info"]),
                ("infra2/image_rect_raw", [LaunchConfiguration("camera_name"), "/infra2/image_rect_raw"]),
                ("infra2/camera_info", [LaunchConfiguration("camera_name"), "/infra2/camera_info"]),
            ],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config_file],
            condition=IfCondition(LaunchConfiguration("launch_rviz")),
        )
    ])