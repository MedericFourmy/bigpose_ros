# Bigpose ROS 

## Setup
Follow installation and setup described in:
- [happypose](https://github.com/agimus-project/happypose)
- [s2m2](https://github.com/MedericFourmy/s2m2)

Install additional dependencies in `requirements.txt`.
Then:  
`colcon build --packages-up-to bigpose_ros --symlink-install`

Download [mesh assets](https://drive.google.com/drive/folders/1GEZm8vE_1ecMzEAHaR_6Z8YHAfaff1hl?usp=sharing) and place them in an `bigpose_ros/assets` folder.


## Run
Main node:  
`ros2 run bigpose_ros bigpose_node`

bigpose_ros exposes 2 main services:
- `/detect` runs 2D detection and 3D pose estimation. The pose is stored in the node and published on \tf as "\pylone_est":   
`ros2 service call /detect std_srvs/srv/Trigger {}`

- `/refine` refines the pose using stereo depth estimation and ICP  
`ros2 service call /refine bigpose_msgs/srv/GetTransformStamped`

Rviz for debug:  
`ros2 launch bigpose_ros rviz_bigpose.py`