# perception_pkg/setup.py
from setuptools import setup
import os
from glob import glob

package_name = 'perception_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ZH',
    maintainer_email='howard40729@gmail.com',
    description='Perception node with Mediapipe for ROS 2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = perception_pkg.perception_node:main',
        ],
    },
)
