

`colcon build --packages-up-to bigpose_ros --symlink-install`


`ros2 run bigpose_ros bigpose_node`

`ros2 service call /detect std_srvs/srv/Trigger {}`


`rviz2 --display-config src/bigpose_ros/bigpose_ros/rviz/debug.rviz`