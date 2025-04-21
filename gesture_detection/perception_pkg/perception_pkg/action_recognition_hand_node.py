#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import mediapipe as mp
import torch.nn as nn
import torch.nn.functional as F
from ament_index_python.packages import get_package_share_directory
import os

# --------------------------------------
# 1) MODEL + ACTION LABELS + COLORS
# --------------------------------------
package_share = get_package_share_directory("perception_pkg")
MODEL_PATH = os.path.join(package_share, "models", "0306transformer_action_recognition_hand3030_office.pth")

actions = [
    'ASCEND', 'DESCEND', 'ME', 'STOP', 'RIGHT', 'BUDDY_UP',
    'FOLLOW_ME', 'OKAY', 'LEFT', 'YOU', 'LEVEL'
]
NUM_CLASSES = len(actions)

colors = [
    (245, 117, 16),
    (117, 245, 16),
    (16, 117, 245),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 0, 128),
    (128, 128, 0),
    (0, 128, 128),
    (50, 100, 50),
    (100, 50, 150)
]

def prob_viz(probs, actions, input_frame, colors):
    """
    Draw a horizontal bar for each action's probability.
    """
    output_frame = input_frame.copy()
    limit = min(len(probs), len(actions), len(colors))
    for i in range(limit):
        action = actions[i]
        c = colors[i]
        prob = probs[i]
        prob_width = int(prob*320*0.8)
        prob_percent = int(prob * 100)
        start_point = (0, 30 + i * 20)
        end_point = (prob_width, 45 + i * 20)
        # cv2.rectangle(output_frame, start_point, end_point, c, -1)
        label_text = f"{action} {prob_percent}%"
        # cv2.putText(output_frame, label_text, (0, 43 + i * 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# --------------------------------------
# 2) DEFINE THE TRANSFORMER MODEL
# --------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, d_model, 2).float() * (np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)

class ActionTransformer(nn.Module):
    def __init__(self, feature_dim, num_classes, embed_dim=64, num_heads=4, ff_dim=128, dropout=0.1):
        super(ActionTransformer, self).__init__()
        self.input_proj = nn.Linear(feature_dim, embed_dim)
        self.pos_encoding = PositionalEncoding(d_model=embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)  # Global average pooling
        x = self.dropout(x)
        logits = self.fc_out(x)
        return logits

# --------------------------------------
# 3) EXTRACT LANDMARKS
# --------------------------------------
def extract_landmarks(results):
    """
    Extract left-hand and right-hand (21x3 each) landmarks from mediapipe Hands.
    If a hand is not detected, returns an array of zeros for that hand.
    """
    left_hand = np.zeros((21, 3))
    right_hand = np.zeros((21, 3))
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            if label == "Left":
                left_hand = landmarks
            elif label == "Right":
                right_hand = landmarks
    return np.concatenate([left_hand.flatten(), right_hand.flatten()])

# --------------------------------------
# 4) ROS2 NODE: ACTION RECOGNITION
# --------------------------------------
class ActionRecognitionNode(Node):
    def __init__(self):
        super().__init__('action_recognition_node')
        self.bridge = CvBridge()
        # Subscribe to the "/image_raw" topic.
        self.subscription = self.create_subscription(
            Image,
            "/image_raw",  # Changed topic to match your running camera topic
            self.image_callback,
            10
        )

        self.gesture_publisher = self.create_publisher(UInt8, "/gesture", 10)

        # Rolling window and prediction storage
        self.sequence = []   # Rolling window of 60 frames
        self.sentence = []   # Store recent predicted actions
        self.threshold = 0.6

        # Initialize and load the Transformer model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActionTransformer(
            feature_dim=126,   # 21 left-hand x 3 + 21 right-hand x 3
            num_classes=NUM_CLASSES,
            embed_dim=64,
            num_heads=4,
            ff_dim=128,
            dropout=0.1
        )
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.get_logger().info("✅ Model Loaded Successfully!")

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error("Failed to convert image: " + str(e))
            return

        # Process frame with MediaPipe Hands
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw hand landmarks for visualization
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Extract landmarks and update sequence
        keypoints = extract_landmarks(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-60:]  # Keep only the last 60 frames

        # When the sequence is complete, run inference
        if len(self.sequence) == 60:
            input_data = torch.tensor([self.sequence], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                output = self.model(input_data)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]
                pred_label = int(np.argmax(probs))
                confidence = probs[pred_label]

            # Visualize probabilities on the image
            image = prob_viz(probs, actions, image, colors)

            # Update sentence with high-confidence prediction
            action_enum = 11
            if confidence > self.threshold:
                predicted_action = actions[pred_label]
                action_enum = self.action_to_enum(predicted_action) if predicted_action != "" else 11 
                if not self.sentence or (predicted_action != self.sentence[-1]):
                    self.sentence.append(predicted_action)
                self.gesture_publisher.publish(UInt8(data=action_enum))
            if len(self.sentence) > 5:
                self.sentence = self.sentence[-5:]
            # Display the sentence on the image
            # cv2.rectangle(image, (0, 0), (320, 25), (245, 117, 16), -1)
            # Uncomment the following lines to display text on the image:
            # cv2.putText(image, " | ".join(self.sentence), (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # For debugging/visualization: show the image window
        cv2.imshow("Action Recognition", image)
        cv2.waitKey(1)

    def action_to_enum(self, action):
        match action:
            case "ASCEND":
                return 0
            case "DESCEND":
                return 1
            case "ME":
                return 2
            case "STOP":
                return 3
            case "RIGHT":
                return 4
            case "BUDDY_UP":
                return 5
            case "FOLLOW_ME":
                return 6
            case "OKAY":
                return 7
            case "LEFT":
                return 8
            case "YOU":
                return 9
            case "LEVEL":
                return 10
            case _:
                return 11

    def destroy_node(self):
        # Clean up MediaPipe and OpenCV windows when shutting down
        self.hands.close()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ActionRecognitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
