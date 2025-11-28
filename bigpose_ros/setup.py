from setuptools import find_packages, setup

package_name = "bigpose_ros"

setup(
    name=package_name,
    version="0.0.2",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/bigpose_ros"]),
        ("share/bigpose_ros", ["package.xml"]),
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