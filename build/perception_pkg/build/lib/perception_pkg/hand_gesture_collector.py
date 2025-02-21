import rclpy
from rclpy.node import Node
import os
import numpy as np
import cv2
import time
import mediapipe as mp

class HandGestureCollector(Node):
    def __init__(self):
        super().__init__('hand_gesture_collector')
        self.get_logger().info("Initializing Hand Gesture Collector Node")

        # Define the base data path
        self.DATA_PATH = os.path.join('MP_Data')

        # Define the actions
        self.actions = np.array(['ASCEND', 'DESCEND', 'ME', 'STOP', 'ToRight', 'BUDDY UP', 'FOLLOW ME', 'OK', 'ToLeft', 'YOU', 'STAY'])

        # Define parameters
        self.sequence_length = 30
        self.no_sequences = 60
        self.start_folder = 30
        
        # Ensure MP_Data directory exists
        os.makedirs(self.DATA_PATH, exist_ok=True)

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        # Open webcam
        self.cap = cv2.VideoCapture(0)
        
        self.setup_directories()
        self.collect_data()

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
        for action in self.actions:
            self.display_message(f"Switching to {action}", duration=1.5)

            for sequence in range(self.start_folder, self.start_folder + self.no_sequences):
                for frame_num in range(self.sequence_length):
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(rgb_frame)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    if frame_num == 0:
                        self.display_message("STARTING COLLECTION", duration=0.5, frame=frame)
                    
                    cv2.putText(frame, f"Collecting {action} - Video {sequence}", (15, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', frame)

                    keypoints = self.extract_keypoints(results)
                    npy_path = os.path.join(self.DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        return

                self.display_message("FINISHED COLLECTION", duration=1.0)
                time.sleep(0.5)

        self.display_message("✅ All Actions Collected!", duration=2.0)
        self.cap.release()
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
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
