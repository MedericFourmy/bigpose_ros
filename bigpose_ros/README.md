# bigpose ROS


$ ros2 node info /bigpose_node
/bigpose_node
  Subscribers:
    /camera/camera/color/camera_info: sensor_msgs/msg/CameraInfo
    /camera/camera/color/image_raw: sensor_msgs/msg/Image
    /camera/camera/infra1/camera_info: sensor_msgs/msg/CameraInfo
    /camera/camera/infra1/image_rect_raw: sensor_msgs/msg/Image
    /camera/camera/infra2/camera_info: sensor_msgs/msg/CameraInfo
    /camera/camera/infra2/image_rect_raw: sensor_msgs/msg/Image
    /tf: tf2_msgs/msg/TFMessage
    /tf_static: tf2_msgs/msg/TFMessage
  Publishers:
    /parameter_events: rcl_interfaces/msg/ParameterEvent
    /rosout: rcl_interfaces/msg/Log
    /tf: tf2_msgs/msg/TFMessage
  Service Servers:
    /bigpose_node/describe_parameters: rcl_interfaces/srv/DescribeParameters
    /bigpose_node/get_parameter_types: rcl_interfaces/srv/GetParameterTypes
    /bigpose_node/get_parameters: rcl_interfaces/srv/GetParameters
    /bigpose_node/list_parameters: rcl_interfaces/srv/ListParameters
    /bigpose_node/set_parameters: rcl_interfaces/srv/SetParameters
    /bigpose_node/set_parameters_atomically: rcl_interfaces/srv/SetParametersAtomically
    /detect: std_srvs/srv/Trigger
    /refine: std_srvs/srv/Trigger
  Service Clients:

  Action Servers:

  Action Clients:




  [WARN] [1764603948.787982873] [bigpose_node]: detect_object_callback tf_stamped_wc:
geometry_msgs.msg.Transform(
    translation=geometry_msgs.msg.Vector3(x=-0.31461541301023366, y=0.5104838376301892, z=0.8170933207922224), 
    rotation=geometry_msgs.msg.Quaternion(x=0.6347767709735721, y=-0.2634210104106203, z=0.7012567168184068, w=0.18949100090421686))