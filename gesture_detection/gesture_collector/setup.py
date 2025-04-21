from setuptools import find_packages, setup

package_name = 'gesture_collector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zhangzihao@ad.ufl.edu',
    maintainer_email='howard40729@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hand_gesture_collector = gesture_collector.hand_gesture_collector:main'
        ],
    },
)
