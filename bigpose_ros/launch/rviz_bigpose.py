from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the path to the package share directory
    pkg_share = get_package_share_directory('bigpose_ros')
    rviz_config = os.path.join(pkg_share, 'rviz', 'debug.rviz')

    return LaunchDescription([
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
        )
    ])
