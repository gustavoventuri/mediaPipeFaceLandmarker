from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image



import cv2
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose


mp_drawing = mp.solutions.drawing_utils 
mp_pose = mp.solutions.pose 

pose = mp_pose.Pose(
    static_image_mode=False,  
    model_complexity=2,  
    enable_segmentation=True, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5  
)

cap = cv2.VideoCapture(1)  

while cap.isOpened():
    s, image = cap.read()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)
    #print(results.pose_landmarks)
    #image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Pose', image)


    if cv2.waitKey(50) == 27:
        break

cap.release()
cv2.destroyAllWindows()
