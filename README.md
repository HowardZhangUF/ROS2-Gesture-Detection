My_ROV_WS Project Guide
Install MediaPipe
Install the MediaPipe library:



cd ~/my_rov_ws
python -m pip install mediapipe
Build the Workspace
Ensure the workspace is properly set up and build it:


cd ~/my_rov_ws
colcon build
Launch the Application
Source the Workspace
Before launching, source the workspace:


cd ~/my_rov_ws
source install/setup.bash
Launch the Nodes
Run the launch file to start all three nodes (camera_pkg, perception_pkg, visualization_pkg):

ros2 launch camera_pkg camera_perception_viz.launch.py
Check Logs
Monitor the output to ensure all nodes are running:

camera_node should indicate the camera is publishing images.
perception_node should process the images.
visualization_node should display the results.
Check Topics
List the active ROS2 topics to confirm data is being published:

ros2 topic list
You should see topics like:

/camera/image_raw
/perception/output
/visualization/output
Visualize Camera Feed
Use rqt_image_view to visualize the camera feed:


ros2 run rqt_image_view rqt_image_view
Select /camera/image_raw from the dropdown.

