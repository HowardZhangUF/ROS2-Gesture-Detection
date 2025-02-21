import os
import cv2
import mediapipe as mp

# Force TensorFlow to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define folder path
video_folder = "/home/bakerherrin/Desktop/Gestures"

# List of files to EXCLUDE
excluded_files = {}

# Get all videos in the folder, filtering out excluded ones
all_videos = [
    f for f in os.listdir(video_folder)
    if f.lower().endswith((".mp4", ".MP4")) and f not in excluded_files
] 

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open {video_path}")
        return

    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30  # Default FPS if invalid

    frame_count = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
        static_image_mode=False,  # Use False for smoother tracking
        model_complexity=1,  # Default complexity
        enable_segmentation=False
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"❌ No more frames to read. Processed {frame_count} frames.")
                break

            frame = cv2.resize(frame, (640, 480))  # Resize for stability

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            frames.append(image)
            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        if len(frames) > 0:
            save_processed_video(frames, video_path, 640, 480, fps)
        else:
            print(f"⚠️ No frames were processed for {video_path}. Skipping save.")

def save_processed_video(frames, video_path, frame_width, frame_height, fps):
    output_path = video_path.replace(".mp4", "_processed.mp4")
    
    # ✅ Ensure codec is compatible with OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Most compatible for MP4

    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"✅ Processed video saved as {output_path}")

for video in all_videos:
    print(f"📌 Processing: {video}")
    process_video(os.path.join(video_folder, video))

print("🎉 All videos processed successfully!")
