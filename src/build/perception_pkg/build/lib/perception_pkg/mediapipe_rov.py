import cv2
import os
import math
import time
import numpy as np
import mediapipe as mp

# Mediapipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

###############################################################################
# 1) ENUMS AND DATA CLASSES
###############################################################################
from enum import Enum

class Keypoint(Enum):
    nose = 0
    left_eye_inner = 1
    left_eye = 2
    left_eye_outer = 3
    right_eye_inner = 4
    right_eye = 5
    right_eye_outer = 6
    left_ear = 7
    right_ear = 8
    mouth_left = 9
    mouth_right = 10
    left_shoulder = 11
    right_shoulder = 12
    left_elbow = 13
    right_elbow = 14
    left_wrist = 15
    right_wrist = 16
    left_pinky = 17
    right_pinky = 18
    left_index = 19
    right_index = 20
    left_thumb = 21
    right_thumb = 22
    left_hip = 23
    right_hip = 24
    left_knee = 25
    right_knee = 26
    left_ankle = 27
    right_ankle = 28
    left_heel = 29
    right_heel = 30
    left_foot_index = 31
    right_foot_index = 32

class HandGesture(Enum):
    Unknown = 0
    Thumb_Up = 1
    Thumb_Down = 2
    Point_Up = 3
    Fist = 4
    Victory = 5
    ILoveYou = 6

class Human:
    def __init__(self, human_id):
        self.ID = human_id
        self.is_buddy = True
        self.pose = None
        self.hand_gesture = None

    # Optional body-fixed frame reference
    def get_bff(self):
        pass

    # Example face bounding box function (not used in main loop but included)
    def get_fbb(self, rov):
        """
        Get the face bounding box based on detected face keypoints.
        """
        keypoint_dict = get_pose_landmarks(rov)

        # Extract keypoints for the face (keypoints 0-10)
        face_keypoints = [
            keypoint_dict[Keypoint(ii).name]
            for ii in range(11)
            if Keypoint(ii).name in keypoint_dict
        ]

        if not face_keypoints:
            print("[INFO] No face keypoints detected.")
            return None

        # Calculate the bounding box
        x_coords = [point[0] for point in face_keypoints]
        y_coords = [point[1] for point in face_keypoints]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        return {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'center_x': center_x,
            'center_y': center_y,
            'width': width,
            'height': height
        }

###############################################################################
# 2) STUB / UTILITY FUNCTIONS (angle calculation, distance estimate, etc.)
###############################################################################

def calculate_angle(v1, v2):
    """
    Calculate the angle (in degrees) between two 2D vectors v1 and v2.
    """
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    # Avoid division by zero:
    cos_angle = dot_product / (norm1 * norm2 + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # numerical stability
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def mocap_human_distance(bbox):
    """
    Stub function for distance estimation; in practice, you'd compute or 
    return a meaningful depth/distance from the bounding box or pose info.
    """
    # For now, just return a fixed or dummy distance:
    return 1.0

###############################################################################
# 3) PERCEPTION LOGIC: OBJECT DETECTION, POSE, HAND GESTURE
###############################################################################

def get_human_bbox(rov):
    """
    Find the bounding box of the largest 'person' detection.
    """
    image = rov.image_buffer
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    detector_result = rov.obj_detector.detect(mp_image)

    num_human = len(detector_result.detections)
    if num_human > 0:
        # find the largest bounding box
        bbox_size = []
        for ii in range(num_human):
            bbox = detector_result.detections[ii].bounding_box
            bbox_size.append(bbox.width + bbox.height)
        max_size = max(bbox_size)
        max_index = bbox_size.index(max_size)
        return detector_result.detections[max_index].bounding_box
    else:
        return None

def get_pose_landmarks(rov):
    """
    Run pose estimation and return a dictionary of keypoint_name -> (x, y).
    """
    image = rov.image_buffer
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    

    estimator_result = rov.pose_estimator.detect(mp_image)
    if len(estimator_result.pose_landmarks) == 0:
        return {}

    landmark_pixels = estimator_result.pose_landmarks[0]
    # (Optional) 3D landmarks if needed:
    # landmark_meters = estimator_result.pose_world[0]

    keypoint_dict = {}
    for ii in range(33):
        kp_name = Keypoint(ii).name
        kp_loc = np.array([
            landmark_pixels[ii].x * rov.FRAME_WIDTH,
            landmark_pixels[ii].y * rov.FRAME_HEIGHT
        ])
        keypoint_dict[kp_name] = kp_loc
    return keypoint_dict

def get_hand_gesture(rov):
    """
    Use Mediapipe GestureRecognizer to detect which gesture each hand is making.
    """
    image = rov.image_buffer
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    recognizer_result = rov.hand_recognizer.recognize(mp_image)

    hand_dict = {"Left": "Unknown", "Right": "Unknown"}
    num_hands = len(recognizer_result.gestures)
    if num_hands > 0:
        for i in range(num_hands):
            # Each hand has 'handedness' and recognized 'gestures'
            handedness_str = recognizer_result.handedness[i][0].category_name
            gesture_str = recognizer_result.gestures[i][0].category_name
            if handedness_str == 'Left':
                hand_dict["Left"] = gesture_str
            if handedness_str == 'Right':
                hand_dict["Right"] = gesture_str
    
    return hand_dict

###############################################################################
# 4) ACTION RECOGNITION / LOGIC (stubs for pointing, buddy-up, etc.)
###############################################################################

def is_pointing(rov, action_duration_sec=2.0):
    """
    Example function to detect if the right arm is straight for at least 50% of frames in the time window.
    """
    start_time = time.time()
    pointing_frames = 0
    total_frames = 0

    while time.time() - start_time < action_duration_sec:
        pose_result = get_pose_landmarks(rov)
        if len(pose_result) == 0:
            continue

        right_shoulder = pose_result.get('right_shoulder', np.array([0,0]))
        right_elbow = pose_result.get('right_elbow', np.array([0,0]))
        right_wrist = pose_result.get('right_wrist', np.array([0,0]))

        v_upper_arm = right_elbow - right_shoulder
        v_forearm   = right_wrist - right_elbow
        angle = calculate_angle(v_upper_arm, v_forearm)
        
        if angle < 30:  # some threshold
            pointing_frames += 1
        total_frames += 1

    if total_frames == 0:
        return False
    pointing_ratio = pointing_frames / total_frames
    return pointing_ratio > 0.5

def is_buddy_up(rov, action_duration_sec=2.0):
    """
    Example function: two hands are 'point_up' for >50% of frames in given time.
    NOTE: For quick testing, you can adapt the recognized gesture strings to match actual recognized names.
    """
    start_time = time.time()
    buddy_up_frames = 0
    total_frames = 0

    while time.time() - start_time < action_duration_sec:
        hand_results = get_hand_gesture(rov)
        # Suppose 'point_up' is recognized as "Point_Up" from the enumerations
        if (hand_results['Left'] == 'Thumb_Up' and 
            hand_results['Right'] == 'Thumb_Up'):
            buddy_up_frames += 1
        total_frames += 1

    if total_frames == 0:
        return False
    buddy_up_ratio = buddy_up_frames / total_frames
    return buddy_up_ratio > 0.5

def get_arm_direction(rov):
    """
    Return a normalized direction vector from right_shoulder to right_wrist.
    """
    pose_result = get_pose_landmarks(rov)
    if len(pose_result) == 0:
        return np.array([0, 0])

    right_shoulder = pose_result['right_shoulder']
    right_wrist = pose_result['right_wrist']
    arm_vector = right_wrist - right_shoulder
    norm = np.linalg.norm(arm_vector)
    if norm < 1e-9:
        return np.array([0, 0])
    return arm_vector / norm

def get_body_frame(rov):
    """
    Return bounding box and center for the torso (shoulders + hips).
    """
    pose_result = get_pose_landmarks(rov)
    if len(pose_result) == 0:
        return None

    left_shoulder = pose_result['left_shoulder']
    right_shoulder = pose_result['right_shoulder']
    left_hip = pose_result['left_hip']
    right_hip = pose_result['right_hip']

    x_min = min(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0])
    y_min = min(left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1])
    x_max = max(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0])
    y_max = max(left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1])
    bbox = (x_min, y_min, x_max, y_max)

    center_of_mass = np.mean([left_shoulder, right_shoulder, left_hip, right_hip], axis=0)
    return {'bbox': bbox, 'origin': center_of_mass}

def find_human(rov):
    """
    Combined function: get the person's bounding box plus any distance estimate.
    """
    bbox = get_human_bbox(rov)
    if bbox is None:
        return None
    pose_result = get_pose_landmarks(rov)

    human_info = {
        'bbox': {
            'x': bbox.origin_x,
            'y': bbox.origin_y,
            'width': bbox.width,
            'height': bbox.height
        },
        'distance_estimate': mocap_human_distance(bbox)
    }
    return human_info

# Example rule-based gesture classifier (unused in main loop, included for reference)
def recognize_gesture_rules(mp_hands, hand_landmarks):
    pass  # left as a placeholder

###############################################################################
# 5) ACTION CLASS (STUB): You can expand logic if needed
###############################################################################
class ROVAction:
    """
    Minimal stubs for your ROVAction calls so code doesn't break.
    Adjust or expand as desired for your own logic.
    """
    @staticmethod
    def follow_diver_action(bbox, frame_width, frame_height):
        """
        Follow action based on the bounding box of the detected object.

        Args:
            bbox: A dictionary containing the bounding box with keys: `x_min`, `y_min`, `x_max`, `y_max`.
            frame_width: The width of the frame.
            frame_height: The height of the frame.

        Returns:
            A tuple of (dx, dy, mean_x, mean_y), representing the offsets and the center of the bounding box.
        """
        # Ensure bbox is valid
        if not bbox or not all(k in bbox for k in ['x_min', 'y_min', 'x_max', 'y_max']):
            print("[FOLLOW] No valid bounding box found.")
            return None

        # Calculate the center of the bounding box
        mean_x = (bbox['x_min'] + bbox['x_max']) / 2
        mean_y = (bbox['y_min'] + bbox['y_max']) / 2

        # Calculate offsets from the frame center
        dx = mean_x - (frame_width / 2)
        dy = mean_y - (frame_height / 2)

        # Print debugging information
        print("[FOLLOW] Bounding box detected. Moving to center.")
        print(f"[FOLLOW] Bounding box center: ({mean_x:.2f}, {mean_y:.2f}), Offsets: (dx={dx:.2f}, dy={dy:.2f})")

        # Return offsets and the center of the bounding box for visualization
        return dx, dy, mean_x, mean_y


    @staticmethod
    def turn_action(rov):
        """
        Turn Controller:
        Adjust the ROV's movement (up, down, left, or right) based on detected hand gestures.
        
        Args:
            rov: The ROV object containing perception modules and control capabilities.
        """
        # Detect hand gestures
        hand_gestures = get_hand_gesture(rov)

        # Check if hand_gestures is None or missing keys
        if not hand_gestures or "Left" not in hand_gestures or "Right" not in hand_gestures:
            return "[TURN] No valid hand gestures detected. Stopping movement."

        # Extract gestures for left and right hands
        left_hand_gesture = hand_gestures["Left"]
        right_hand_gesture = hand_gestures["Right"]

        # Control logic for turning
        if left_hand_gesture == "Closed_Fist" and right_hand_gesture == "Open_Palm":
            return "[TURN] Gesture detected: Left Fist and Right Palm. Turning left. Yaw + 90."
        
        elif left_hand_gesture == "Open_Palm" and right_hand_gesture == "Closed_Fist":
            return "[TURN] Gesture detected: Left Palm and Right Fist. Turning right. Yaw - 90."
        
        elif left_hand_gesture == "Thumb_Up" or right_hand_gesture == "Thumb_Up":
            return "[TURN] Gesture detected: Thumb Up. Moving up."
        
        elif left_hand_gesture == "Thumb_Down" or right_hand_gesture == "Thumb_Down":
            return "[TURN] Gesture detected: Thumb Down. Moving down."
        
        elif left_hand_gesture == "Open_Palm" and right_hand_gesture == "Unknown" or right_hand_gesture == "Open_Palm" and left_hand_gesture == "Unknown":
            return "[TURN] Gesture detected: Palm. Stop"
        
        else:
            return "[TURN] No valid gesture detected. Stopping movement."


                

    def circle_action(self, rov, increment=10):
        # If we haven't set a direction, default to +1
        if not hasattr(rov, 'yaw_direction'):
            rov.yaw_direction = 1

        rov.yaw += increment * rov.yaw_direction

        # Flip direction
        if rov.yaw >= 360:
            rov.yaw = 360
            rov.yaw_direction = -1
        elif rov.yaw <= -360:
            rov.yaw = -360
            rov.yaw_direction = 1

        print(f"[CIRCLE] Yaw updated to {rov.yaw} (direction={rov.yaw_direction})")
        return f"[CIRCLE] Yaw is now {rov.yaw}"


    
        
        
        
    

###############################################################################
# 6) ROV CLASS: ties everything together (Camera + Perception + Visualization)
###############################################################################
class rov:
    MAX_SPEED = 1           # [m/s]
    MAX_TURN_RATE = 0.2     # [rad/s]
    CTRL_FREQ = 10          # [Hz]
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30                # might get updated from camera
    CTRL_INT = 10

    DEFAULT_CONTROL = 'follow'
    DEFAULT_PERCEPTION = 'obj_detection'
    THRES_BBOX_RATIO = 0.6  # ratio threshold for switching to hand_recognition

    def __init__(self, camera_id=0, action=None):
        self.launch_time = time.time()
        self.position = np.array([0, 0, 0])
        self.yaw = 0
        self.camera_id = camera_id
        self.cap = None
        self.events = [(0.0,'start')]
        
        # modes
        self.control_mode = self.DEFAULT_CONTROL
        self.perception_mode = self.DEFAULT_PERCEPTION
        
        # frame counters
        self.frame_counter = 0
        self.perception_counter = 0
        self.image_buffer = None
        
        # results
        self.obj_result = None
        self.pose_result = None
        self.hand_result = None
        
        self.action = action
        
        # Initialize models
        base_options = python.BaseOptions(model_asset_path='models/efficientdet_lite0.tflite')
        detector_options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            category_allowlist=["person"],
            score_threshold=0.5
        )
        self.obj_detector = vision.ObjectDetector.create_from_options(detector_options)

        base_options = python.BaseOptions(model_asset_path='models/pose_landmarker_full.task')
        pose_options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5
        )
        self.pose_estimator = vision.PoseLandmarker.create_from_options(pose_options)

        base_options = python.BaseOptions(model_asset_path='models/gesture_recognizer.task')
        gesture_options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5
        )
        self.hand_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)

    def turn_on_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Check camera ID or device.")
        self.FRAME_WIDTH = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.FRAME_HEIGHT = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.FPS = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.CTRL_INT = max(1, int(self.FPS / self.CTRL_FREQ))

    def get_camera_image(self):
        """
        Grab a frame from the webcam. If successful, store in self.image_buffer.
        """
        if self.frame_counter % self.CTRL_INT == 0:
            self.perception_counter += 1
        self.frame_counter += 1

        success, frame = self.cap.read()
        if not success:
            print("[INFO] Frame is not successfully captured.")
        else:
            self.image_buffer = frame.copy()

    def get_perception(self):
        """
        Depending on the frame, run object detection or other tasks in a round-robin or logic-driven manner.
        """
        # Only run the "heavy" tasks every CTRL_INT frames
        if (self.frame_counter - 1) % self.CTRL_INT == 0:
            # This toggles between object detection and other tasks
            if self.perception_counter % 2 == 1:
                self.obj_result = get_human_bbox(self)
            else:
                if self.perception_mode == 'pose_estimation':
                    self.pose_result = get_pose_landmarks(self)
                    # example: calls in ROVAction
                    

                elif self.perception_mode == 'hand_recognition':
                    self.hand_result = get_hand_gesture(self)

                else:
                    self.obj_result = get_human_bbox(self)

    def visualize_perception(self):
        """
        Overlay bounding boxes, gestures, or pose keypoints on the current frame.
        """
        if self.image_buffer is None:
            return None
        frame = self.image_buffer.copy()

        # Some text annotations
        font = cv2.FONT_HERSHEY_PLAIN
        margin = 5
        font_scale = 1
        font_thickness = 1
        red = (0, 0, 255)
        _, line_height = cv2.getTextSize("Text", font, font_scale, font_thickness)[0]
        origin_x = margin
        origin_y = margin + line_height

        # top-left: time
        session_time = time.time() - self.launch_time
        time_text = f"Time: {session_time:.2f}"
        cv2.putText(frame, time_text, (origin_x, origin_y), font, font_scale, red, font_thickness)
        origin_y += margin + line_height

        # control & perception modes
        control_text = 'Control mode: ' + self.control_mode
        cv2.putText(frame, control_text, (origin_x, origin_y), font, font_scale, red, font_thickness)
        origin_y += margin + line_height

        perception_text = 'Perception mode: ' + self.perception_mode
        cv2.putText(frame, perception_text, (origin_x, origin_y), font, font_scale, red, font_thickness)
        origin_y += margin + line_height

        # 1) Show object detection bounding box
        if self.obj_result:
            bbox_pt1 = (int(self.obj_result.origin_x), int(self.obj_result.origin_y))
            bbox_pt2 = (
                int(self.obj_result.origin_x + self.obj_result.width),
                int(self.obj_result.origin_y + self.obj_result.height)
            )
            cv2.rectangle(frame, bbox_pt1, bbox_pt2, red, thickness=2)

        # 2) If in hand_recognition mode, print recognized gestures
        if self.perception_mode == 'hand_recognition' and self.hand_result is not None:
            left_text = 'Left hand: ' + str(self.hand_result['Left'])
            right_text = 'Right hand: ' + str(self.hand_result['Right'])
            cv2.putText(frame, left_text, (origin_x, origin_y), font, font_scale, red, font_thickness)
            origin_y += margin + line_height
            cv2.putText(frame, right_text, (origin_x, origin_y), font, font_scale, red, font_thickness)
            origin_y += margin + line_height

        # 3) If in pose_estimation mode, display a couple keypoints
        elif self.perception_mode == 'pose_estimation' and self.pose_result is not None:
            # draw shoulders, for example
            if 'right_shoulder' in self.pose_result:
                r_shoulder = self.pose_result['right_shoulder']
                cv2.circle(frame, (int(r_shoulder[0]), int(r_shoulder[1])), 5, red, -1)
            if 'left_shoulder' in self.pose_result:
                l_shoulder = self.pose_result['left_shoulder']
                cv2.circle(frame, (int(l_shoulder[0]), int(l_shoulder[1])), 5, red, -1)

        if self.control_mode == 'follow' and hasattr(self, 'follow_center'):
            if self.follow_center is not None:
                cx, cy = self.follow_center
                # Draw a green circle at (cx, cy)
                green = (0, 255, 0)
                cv2.circle(frame, (int(cx), int(cy)), 6, green, thickness=-1)
        return frame

    def update_task(self):
        """
        Logic: if the detected person's bounding box is big, assume they're close 
        enough to do hand recognition; otherwise, do object detection. 
        Returns an action string (or None).
        """
        action_str = None  # We'll store the result text here

        if self.obj_result:
            # 1) Check bounding box size
            if (self.obj_result.width < (self.THRES_BBOX_RATIO * self.FRAME_WIDTH) or
                self.obj_result.height < (self.THRES_BBOX_RATIO * self.FRAME_HEIGHT)):
                # If bounding box is small => object detection mode
                self.perception_mode = 'obj_detection'
                self.control_mode = 'follow'

                # Trigger "follow" action
                if self.action:
                    bbox = {
                        'x_min': self.obj_result.origin_x,
                        'y_min': self.obj_result.origin_y,
                        'x_max': self.obj_result.origin_x + self.obj_result.width,
                        'y_max': self.obj_result.origin_y + self.obj_result.height
                    }
                    result = self.action.follow_diver_action(bbox, self.FRAME_WIDTH, self.FRAME_HEIGHT)
                    print(result)            # Debug print
                    action_str = str(result)  # Capture the return string/tuple
            else:
                # If bounding box is large => switch to hand recognition mode
                self.perception_mode = 'hand_recognition'
                self.control_mode = 'turn'
                
                # Trigger "turn" action
                if self.action:
                    result = self.action.turn_action(self)
                    print(result)
                    action_str = str(result)
        else:
            self.perception_mode = 'obj_detection'
            self.control_mode = 'search'
            if self.action:
                result = self.action.circle_action(self, increment=10)
                print(result)
                action_str = str(result)

        return action_str