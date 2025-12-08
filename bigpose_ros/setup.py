import os
from glob import glob
from setuptools import find_packages, setup

package_name = "bigpose_ros"

setup(
    name=package_name,
    version="0.0.2",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/bigpose_ros"]),
        (
            "share/bigpose_ros/assets/meshes",
            [f for f in glob("assets/meshes/*") if not os.path.isdir(f)],
        ),
        ("share/bigpose_ros/assets/meshes/textures_pylone_but_better", glob("assets/meshes/textures_pylone_but_better/*")),
        ("share/bigpose_ros/launch", glob("launch/*")),
        ("share/bigpose_ros/rviz", glob("rviz/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mederic Fourmy",
    maintainer_email="mederic.fourmy@gmail.com",
    description="2 step pose estimation for big objects",
    license="TODO",
    entry_points={
        "console_scripts": ["bigpose_node = bigpose_ros.bigpose_node:main"],
    },
)


# ValueError: string is not a file: `/home/ros/sandbox_mf/ws_pylone/install/bigpose_ros/share/bigpose_ros/assets/meshes/pylone_but_better.obj`