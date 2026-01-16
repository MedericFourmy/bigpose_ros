import os
from pathlib import Path
from glob import glob
from setuptools import find_packages, setup

from generate_parameter_library_py.setup_helper import generate_parameter_module

package_name = "bigpose_ros"

module_name = "bigpose_ros_parameters"
yaml_file = "bigpose_ros/bigpose_ros_parameters.yaml"
generate_parameter_module(module_name, yaml_file)


setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            f"share/{package_name}/assets/meshes",
            [f for f in glob("assets/meshes/*") if not os.path.isdir(f)],
        ),
        (f"share/{package_name}/assets/meshes/textures_pylone_but_better", glob("assets/meshes/textures_pylone_but_better/*")),
        (f"share/{package_name}/launch", glob("launch/*")),
        (f"share/{package_name}/config", glob("config/*")),
        (f"share/{package_name}/rviz", glob("rviz/*")),
    ],
    install_requires=[
        "setuptools",
        "transformers",
    ],
    zip_safe=True,
    maintainer="Mederic Fourmy",
    maintainer_email="mederic.fourmy@gmail.com",
    description="2 step pose estimation for big objects",
    license="TODO",
    entry_points={
        "console_scripts": ["bigpose_node = bigpose_ros.bigpose_node:main"],
    },
)
