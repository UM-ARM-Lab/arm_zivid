import os
from setuptools import setup

package_name = 'arm_zivid'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
    ],
    # This is important as well
    install_requires=[
        'setuptools',
        'opencv-python',
        'h5py',
        'zarr<3.0.0',  # Ensure compatibility with zarr v2
    ],
    zip_safe=True,
    author='ROS 2 Developer',
    author_email='ros2@ros.com',
    maintainer='ROS 2 Developer',
    maintainer_email='ros2@ros.com',
    keywords=['foo', 'bar'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: TODO',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='arm_zivid',
    license='MIT',
    # Like the CMakeLists add_executable macro, you can add your python
    # scripts here.
    entry_points={
        'console_scripts': [
            'arm_zivid_ros_node.py = arm_zivid.arm_zivid_ros_node:main',
            'arm_zivid_ros_node_local.py = arm_zivid.arm_zivid_ros_node_local:main',
            'zivid_ds_to_vid.py = arm_zivid.zivid_ds_to_vid:main',
        ],
    },
)