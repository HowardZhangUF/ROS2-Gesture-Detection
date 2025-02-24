import rclpy
from rclpy.node import Node
import os
import numpy as np
import cv2
import time
import mediapipe as mp
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
class HandGestureCollector(Node):
    def __init__(self):
        super().__init__('hand_gesture_collector')
        self.get_logger().info("Initializing Hand Gesture Collector Node")

        # Define the base data path
        self.DATA_PATH = os.path.join('MP_Data')

        # Define the actions
        self.actions = ['ASCEND', 'DESCEND', 'ME', 'STOP', 'ToRight', 'BUDDY UP', 'FOLLOW ME', 'OK', 'ToLeft', 'YOU', 'STAY']
        self.action_index = 0

        # Define parameters
        self.sequence_length = 30
        self.no_sequences = 60
        self.start_folder = 30
        self.cam_sub = self.create_subscription(msg_type=Image,topic="image_raw",callback=self.image_callback,qos_profile=1)
        self.collect_timer = self.create_timer(timer_period_sec=1/10,callback=self.collect_data)
        self.br = CvBridge()
        
        # Ensure MP_Data directory exists
        os.makedirs(self.DATA_PATH, exist_ok=True)

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        self.frame =None 
        self.ret = False
        self.frame_num = 0
        self.sequence_num = 0
        
        self.setup_directories()

    def image_callback(self,msg):
            self.frame = self.br.imgmsg_to_cv2(msg,"bgr8")
            self.ret = True

    def setup_directories(self):
        for action in self.actions:
            action_path = os.path.join(self.DATA_PATH, action)
            os.makedirs(action_path, exist_ok=True)
            try:
                dirmax = np.max(np.array([int(f) for f in os.listdir(action_path) if f.isdigit()]))
            except ValueError:
                dirmax = 0
            for sequence in range(1, self.no_sequences + 1):
                os.makedirs(os.path.join(action_path, str(dirmax + sequence)), exist_ok=True)
        self.get_logger().info("✅ All directories created successfully!")

    def extract_keypoints(self, results):
        keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                keypoints.extend([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        else:
            keypoints = [[0, 0, 0]] * 21  # Default when no hands detected
        return np.array(keypoints).flatten()

    def collect_data(self):
        if self.frame is None:
            return
        action = self.actions[self.action_index]

        # for sequence in range(self.start_folder, self.start_folder + self.no_sequences):
        sequence = self.sequence_num
        # self.cap.grab()
        # ret, frame = self.cap.retrieve()
        if not self.ret:
            return
        self.ret = False
        rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        # rgb_frame = self.frame
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(self.frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        if self.frame_num == 0:
            self.display_message("STARTING COLLECTION", duration=0.5, frame=self.frame)
        
        cv2.putText(self.frame, f"Collecting {action} - Video {sequence}", (15, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', self.frame)

        keypoints = self.extract_keypoints(results)
        npy_path = os.path.join(self.DATA_PATH, action, str(sequence), str(self.frame_num))
        np.save(npy_path, keypoints)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            return
        if self.frame_num == self.sequence_length-1:
            self.display_message("FINISHED COLLECTION", duration=3.0)
            time.sleep(1)
            self.sequence_num +=1
            self.frame_num = 0
        
        if self.sequence_num == (self.start_folder+self.no_sequences) -1:
            self.action_index+=1
            self.sequence_num = 0
            self.display_message(f"Switching to {action}", duration=5.0)
        if self.action_index == len(self.actions):
            self.display_message("✅ All Actions Collected!", duration=5.0)
            cv2.destroyAllWindows()
            self.get_logger().info("Data collection complete.")

    def display_message(self, message, duration=1.0, frame=None):
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, message, (100, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', frame)
        cv2.waitKey(int(duration * 1000))


def main(args=None):
    rclpy.init(args=args)
    node = HandGestureCollector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
