# My_ROV_WS Project Guide

My_ROV_WS is a ROS 2 setup that demonstrates how to integrate MediaPipe with 
camera, perception, and visualization nodes. Follow the steps below to install
dependencies, build the workspace, and launch the application.

---

## Install MediaPipe

Install the MediaPipe library:

```bash
cd ~/my_rov_ws
python -m pip install mediapipe

---

## Build the Workspace

```bash
cd ~/my_rov_ws
colcon build

---# My_ROV_WS Project Guide

My_ROV_WS is a ROS 2 setup that demonstrates how to integrate MediaPipe with 
camera, perception, and visualization nodes. Follow the steps below to install
dependencies, build the workspace, and launch the application.

---

## Install MediaPipe

Install the MediaPipe library:

```bash
cd ~/my_rov_ws
python -m pip install mediapipe
```

---

## Build the Workspace

```bash
cd ~/my_rov_ws
colcon build
```

---

## 1. Source the Workspace

Before launching, source the workspace:

```bash
cd ~/my_rov_ws
source install/setup.bash
```

---

## 2. Launch the Nodes

Run the launch file to start all three nodes (camera_pkg, perception_pkg, and visualization_pkg):

```bash
ros2 launch camera_pkg camera_perception_viz.launch.py
```

---

## Check Topics

List the active ROS 2 topics to confirm data is being published:

```bash
ros2 topic list
```

You should see topics like:

```
/camera/image_raw
/perception/output
/visualization/output
```

---

## Visualize Camera Feed

Use `rqt_image_view` to visualize the camera feed:

```bash
ros2 run rqt_image_view rqt_image_view


