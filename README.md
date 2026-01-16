# Bigpose ROS 

## Setup
Follow installation and setup described in:
- [happypose](https://github.com/agimus-project/happypose)
- [s2m2](https://github.com/MedericFourmy/s2m2)

Install additional dependencies in `requirements.txt`.  
Download the open set 2D detector you want to use, e.g.:
```
hf download openmmlab-community/mm_grounding_dino_base_all
```

Then:  
```bash
colcon build --symlink-install --packages-up-to bigpose_ros
```
Download [mesh assets](https://drive.google.com/drive/folders/1GEZm8vE_1ecMzEAHaR_6Z8YHAfaff1hl?usp=sharing) and place them in an `bigpose_ros/assets` folder. Your project should look like:


```bash
.
├── bigpose_msgs
│   ├── CMakeLists.txt
│   ├── package.xml
│   └── srv
│       └── GetTransformStamped.srv
├── bigpose_ros
│   ├── assets
│   │   └── meshes
│   │       ├── pylone_but_better.mtl
│   │       ├── pylone_but_better.obj
│   │       ├── pylone_but_better_with_textures.mtl
│   │       ├── pylone_icp_metal_2parts.mtl
│   │       ├── pylone_icp_metal_2parts.obj
│   │       └── textures_pylone_but_better
│   │           ├── black_body_ka.png
│   │           ├── black_body_kd.png
│   │           ├── black_body_ks.png
│   │           ├── gray_parts_ka.png
│   │           ├── gray_parts_kd.png
│   │           └── gray_parts_ks.png
```


## Run
```bash
ros2 launch bigpose_ros bigpose.launch.py launch_rviz:=true
```

bigpose_ros exposes 2 main services:
- `/bigpose_ros/detect` runs 2D detection and 3D pose estimation. The pose is stored in the node and published on \tf (see `object_frame_id` parameter):
```bash
ros2 service call /bigpose_ros/detect std_srvs/srv/Trigger {} 
```

- `/bigpose_ros/refine` refines the pose using stereo depth estimation and ICP  
```bash
ros2 service call /bigpose_ros/refine bigpose_msgs/srv/GetTransformStamped 
```
