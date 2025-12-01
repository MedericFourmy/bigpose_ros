cmake_minimum_required(VERSION 3.10)
project(bigpose_msgs)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetTransformStamped.srv"
  DEPENDENCIES geometry_msgs
)

# Install Python modules
# ament_python_install_package(${PROJECT_NAME})
install(FILES package.xml DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(rosidl_default_runtime)
ament_package()