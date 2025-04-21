import rclpy
from rclpy.node import Node
import os
import numpy as np
import cv2
import time
import mediapipe as mp
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8
from cv_bridge import CvBridge
from datetime import datetime
class HandGestureCollector(Node):
    def __init__(self):
        super().__init__('hand_gesture_collector')
        self.get_logger().info("Initializing Hand Gesture Collector Node")

        # Define the base data path
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.DATA_PATH = os.path.join(f'{timestamp}_MP_Data')

        # Define the actions
        self.actions = [ 'OK','ASCEND', 'DESCEND', 'YOU', 'ME', 'ToRight','ToLeft','BUDDY_UP', 'FOLLOW_ME', 'STAY', 'STOP']
        self.action_index = 0

        # Define parameters
        self.sequence_length = 30
        self.no_sequences = 30 # 60
        self.start_folder = 1
        self.cam_sub = self.create_subscription(msg_type=Image,topic="image_raw",callback=self.image_callback,qos_profile=1)
        self.collect_timer = self.create_timer(timer_period_sec=1/30,callback=self.collect_data)
        self.light_pwm_pub = self.create_publisher(
            msg_type = UInt8,
            topic= f"rov/lights_b",
            qos_profile = 1,
        )
        self.br = CvBridge()
        
        # Ensure MP_Data directory exists
        os.makedirs(self.DATA_PATH, exist_ok=True)

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        self.frame = None 
        self.ret = False    
        self.frame_num = 1
        self.sequence_num = 1
        
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
        light_msg = UInt8()
        light_msg.data = int(10)
        self.light_pwm_pub.publish(light_msg)
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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', self.frame)

        keypoints = self.extract_keypoints(results)
        npy_path = os.path.join(self.DATA_PATH, action, str(sequence), str(self.frame_num))
        np.save(npy_path, keypoints)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            return
        self.frame_num+=1
        if self.frame_num == self.sequence_length:
            light_msg = UInt8()
            light_msg.data = int(0)
            self.light_pwm_pub.publish(light_msg)
            self.display_message("FINISHED COLLECTION", duration=1)
            time.sleep(0.5)
            self.sequence_num +=1
            self.frame_num = 1
        if self.sequence_num == (self.start_folder+self.no_sequences) -1:
            self.action_index+=1
            self.sequence_num = self.start_folder
            light_msg = UInt8()
            light_msg.data = int(0)
            self.light_pwm_pub.publish(light_msg)
            self.display_message(f"Switching to {self.actions[self.action_index]}", duration=7.0)
        if self.action_index == len(self.actions):
            self.display_message("✅ All Actions Collected!", duration=5.0)
            cv2.destroyAllWindows()
            self.get_logger().info("Data collection complete.")

    def display_message(self, message, duration=1.0, frame=None):
        if frame is None:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(frame, message, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', frame)
        cv2.waitKey(int(duration * 1000))


def main(args=None):
    rclpy.init(args=args)
    node = HandGestureCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    
   
    

if __name__ == '__main__':
    main()
