import cv2
import time
import math as m
import mediapipe as mp

#função para encontrar a distancia entre os pontos do ragdoll

def distEuclidiana(x1,y1, x2,y2):
    dist = m.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist

#funcao para calcular o angulo entre os pontos
def angulo(x1,y1, x2,y2):
    i = m.acos((y2-y1)*(-y1) / (m.sqrt( (x2-x1)**2 + (y2-y1)**2 ) * y1) )
    ang = int(180/m.pi)*i 
    return ang

# Initialize frame counters.
good_frames = 0
bad_frames  = 0
 
# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX
 
# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(1)
 
while True:
    s, image = cap.read()
 
    fps = int(cap.get(cv2.CAP_PROP_FPS)) #para medir o tempo em que fica em má posica
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    # Get fps.
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Get height and width of the frame.
    h, w = image.shape[:2]
    
    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image.
    keypoints = pose.process(image)
    
    # Convert the image back to BGR.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #norm_coordinate  = pose.process(image).pose_landmark.landmark[mp.solutions.pose.PoseLandmark.Neck].coordinate

        # Use lm and lmPose as representative of the following methods.
    lm = keypoints.pose_landmarks
    lmPose  = mp_pose.PoseLandmark
    # ombro esquedo
    l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
    l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
    
    # ombro direito.
    r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
    r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
    
    # orelha esquerda.
    l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
    l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
    
    # Left hip.
    l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
    l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

    # Calculate distance between left shoulder and right shoulder points.
    offset = distEuclidiana(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
    
    # Assist to align the camera to point at the side view of the person.
    # Offset threshold 30 is based on results obtained from analysis over 100 samples.
    if offset < 100:
        cv2.putText(image, str(int(offset)) + ' Alinhado', (w - 150, 30), font, 0.9, green, 2)
    else:
        cv2.putText(image, str(int(offset)) + ' Desalinhado', (w - 150, 30), font, 0.9, red, 2)

    # Calculate angles.
    neck_inclination = angulo(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
    torso_inclination = angulo(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
    
    # Draw landmarks.
    cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
    cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)
    
    # Let's take y - coordinate of P3 100px above x1,  for display elegance.
    # Although we are taking y = 0 while calculating angle between P1,P2,P3.
    cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
    cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
    cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
    
    # Similarly, here we are taking y - coordinate 100px above x1. Note that
    # you can take any value for y, not necessarily 100 or 200 pixels.
    cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)
    
    # Put text, Posture and angle inclination.
    # Text string for display.
    angle_text_string = 'Pescoco : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

    if neck_inclination < 40 and torso_inclination < 10:
        bad_frames = 0
        good_frames += 1
        
        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
        cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
        cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)
    
        # Join landmarks.
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)
 
    else:
        good_frames = 0
        bad_frames += 1
    
        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
        cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
        cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)
    
        # Join landmarks.
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)
    
        # Calculate the time of remaining in a particular posture.
        good_time = (1 / fps) * good_frames
        bad_time =  (1 / fps) * bad_frames
    
    # Pose time.
    if good_time > 0:
        time_string_good = 'Boa Postura : ' + str(round(good_time, 1)) + 's'
        cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
    else:
        time_string_bad = 'Postura Ruim : ' + str(round(bad_time, 1)) + 's'
        cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)



    cv2.imshow('Pose', image)

    if cv2.waitKey(5)  == 27:
      break
cap.release()