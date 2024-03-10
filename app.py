# app.py
from flask import Flask, render_template, Response
from camera import VideoCamera
import cv2


#----------------------------------------------------------------------------------------------------

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import math as m
import statistics

#modelo pre treinado para montar a pose
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

#funçaõ para medir a distancia
def distEuclidiana(x1,y1, x2,y2):
    dist = m.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist

#funcao para calcular o angulo entre os pontos
def angulo(x1,y1, x2,y2):
    i = m.acos((y2-y1)*(-y1) / (m.sqrt( (x2-x1)**2 + (y2-y1)**2 ) * y1) )
    ang = int(180/m.pi)*i 
    return ang

def ang_triangulo(a, b):
    a = abs(a)
    b = abs(b)
    return abs(int(180 - (a + b)))

#função para desenhar a pose na imagem
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

font = cv2.FONT_HERSHEY_SIMPLEX
dist_maos = 0
mao_direita_x, mao_direita_y, mao_esquerda_x, mao_esquerda_y = 0,0,0,0
stt_maos = ''
stt_braco_dir = ''
stt_braco_esq = ''
#carrega video
video = cv2.VideoCapture(0)

#grava video....
gravacao = 1
if gravacao == 1:
    if (video.isOpened() == False):  
        print("Error reading video file")

    frame_width = int(video.get(3)) 
    frame_height = int(video.get(4)) 
    size = (frame_width, frame_height) 
    gravado = cv2.VideoWriter('video_exemplo.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size)

    
    def pose_worker():
        with mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as pose:
            while True:
                s, image = video.read()
                #crop video
                #x,y,h,w = 500,270,250,500
                #img = image[y:y+h, x:x+w]
                h, w = image.shape[:2]
                img = image
                

                #trata imagem
                img.flags.writeable = False
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                #chama mediapipe
                results = pose.process(img)

                #obtem mao direita
                #pontos
                # 16 - right wrist
                # 18 - right pinky
                # 20 - right index
                # 22 - right thumb
                keypoints = pose.process(image)
            
                # Convert the image back to BGR.
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                #norm_coordinate  = pose.process(image).pose_landmark.landmark[mp.solutions.pose.PoseLandmark.Neck].coordinate

                    # Use lm and lmPose as representative of the following methods.
                lm = results.pose_landmarks
                lmPose  = mp_pose.PoseLandmark
                
                try:
                    mao_direita_x = [(0 + lm.landmark[lmPose.RIGHT_WRIST].x * w),
                                    (0 + lm.landmark[lmPose.RIGHT_PINKY].x * w),
                                    (0 + lm.landmark[lmPose.RIGHT_INDEX].x * w),
                                    (0 + lm.landmark[lmPose.RIGHT_THUMB].x * w)]
                    mao_direita_y = [(0 + lm.landmark[lmPose.RIGHT_WRIST].y * h),
                                    (0 + lm.landmark[lmPose.RIGHT_PINKY].y * h),
                                    (0 + lm.landmark[lmPose.RIGHT_INDEX].y * h),
                                    (0 + lm.landmark[lmPose.RIGHT_THUMB].y * h)] 

                    mao_direita_x = int(statistics.mean(mao_direita_x))
                    mao_direita_y = int(statistics.mean(mao_direita_y))           
                    

                    mao_esquerda_x = [(0 + lm.landmark[lmPose.LEFT_WRIST].x * w),
                                    (0 + lm.landmark[lmPose.LEFT_PINKY].x * w),
                                    (0 + lm.landmark[lmPose.LEFT_INDEX].x * w),
                                    (0 + lm.landmark[lmPose.LEFT_THUMB].x * w)]
                    mao_esquerda_y = [(0 + lm.landmark[lmPose.LEFT_WRIST].y * h),
                                    (0 + lm.landmark[lmPose.LEFT_PINKY].y * h),
                                    (0 + lm.landmark[lmPose.LEFT_INDEX].y * h),
                                    (0 + lm.landmark[lmPose.LEFT_THUMB].y * h)] 

                    cotovelo_direito_x = int(0 + lm.landmark[lmPose.RIGHT_ELBOW].x * w)
                    cotovelo_direito_y = int(0 + lm.landmark[lmPose.RIGHT_ELBOW].y * h)

                    ombro_direito_x = int(0 + lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
                    ombro_direito_y = int(0 + lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

                    cotovelo_esquerdo_x = int(0 + lm.landmark[lmPose.LEFT_ELBOW].x * w)
                    cotovelo_esquerdo_y = int(0 + lm.landmark[lmPose.LEFT_ELBOW].y * h)
                    
                    ombro_esquerdo_x = int(0 + lm.landmark[lmPose.LEFT_SHOULDER].x * w)
                    ombro_esquerdo_y = int(0 + lm.landmark[lmPose.LEFT_SHOULDER].y * h)



                    mao_esquerda_x = int(statistics.mean(mao_esquerda_x))
                    mao_esquerda_y = int(statistics.mean(mao_esquerda_y))

                    try:
                        cv2.line(img, (mao_direita_x, mao_direita_y), (cotovelo_direito_x, cotovelo_direito_y), (255,0,0), 2)
                        cv2.line(img, (mao_direita_x, mao_direita_y), (ombro_direito_x, ombro_direito_y), (255,0,0), 2)
                        cv2.line(img, (ombro_direito_x, ombro_direito_y), (cotovelo_direito_x, cotovelo_direito_y), (255,0,0), 2)

                        cv2.line(img, (mao_esquerda_x, mao_esquerda_y), (cotovelo_esquerdo_x, cotovelo_esquerdo_y), (255,0,255), 2)
                        cv2.line(img, (mao_esquerda_x, mao_esquerda_y), (ombro_esquerdo_x, ombro_esquerdo_y), (255,0,255), 2)
                        cv2.line(img, (ombro_esquerdo_x, ombro_esquerdo_y), (cotovelo_esquerdo_x, cotovelo_esquerdo_y), (255,0,255), 2)
                    except:
                        pass

                    dist_ombros = int(distEuclidiana(ombro_direito_x,ombro_direito_y, ombro_esquerdo_x, ombro_esquerdo_y))            
                    dist_maos = int(distEuclidiana(mao_direita_x,mao_direita_y, mao_esquerda_x, mao_esquerda_y))
                    dist_mao_ombro_dir = int(distEuclidiana(mao_direita_x,mao_direita_y, ombro_direito_x,ombro_direito_y))
                    dist_mao_ombro_esq = int(distEuclidiana(mao_esquerda_x, mao_esquerda_y, ombro_esquerdo_x, ombro_esquerdo_y))
                    dist_mao_cotov_dir = int(distEuclidiana(mao_direita_x,mao_direita_y, cotovelo_direito_x, cotovelo_direito_y))
                    dist_mao_cotov_esq = int(distEuclidiana(mao_esquerda_x, mao_esquerda_y, cotovelo_esquerdo_x, cotovelo_esquerdo_y))

                    if dist_maos >= dist_ombros:
                        stt_maos = "Afastadas"
                    else:
                        stt_maos = "Juntas"
                    
                    try:
                        ang_cotovelo_dir = int(angulo(cotovelo_direito_x, cotovelo_direito_y, mao_direita_x, mao_direita_y))
                        ang_ombro_dir = int(angulo(cotovelo_direito_x, cotovelo_direito_y, ombro_direito_x, ombro_direito_y))
                        ang_bc_dir = ang_triangulo(ang_cotovelo_dir, ang_ombro_dir)
                        
                        cv2.putText(img,f'ang cot dir: {str(ang_bc_dir)}', (10,60), font, 0.75, (127,127,0), 2 )

                        if dist_mao_ombro_dir > dist_mao_cotov_dir and ang_bc_dir < 45:
                            stt_braco_dir = "Esticado"
                        else:
                            stt_braco_dir = "Dobrado"


                    except:
                        pass
                    
                    try:            
                        ang_cotovelo_esq = int(angulo(cotovelo_esquerdo_x, cotovelo_esquerdo_y, mao_esquerda_x, mao_esquerda_y))
                        ang_ombro_esq = int(angulo(cotovelo_esquerdo_x, cotovelo_esquerdo_y, ombro_esquerdo_x, ombro_esquerdo_y))
                        ang_bc_esq = ang_triangulo(ang_cotovelo_esq, ang_ombro_esq)
                        
                        cv2.putText(img,f'ang cot esq: {str(ang_bc_esq)}', (10,90), font, 0.75, (127,127,0), 2 )

                        if dist_mao_ombro_esq > dist_mao_cotov_esq and ang_bc_esq < 45:
                            stt_braco_esq = "Esticado"
                        else:
                            stt_braco_esq = "Dobrado"

                    except:
                        pass
                    

                except:
                    pass
                
                cv2.circle(img,(mao_direita_x,mao_direita_y), 20, (255,0,0),3)
                cv2.circle(img,(mao_esquerda_x,mao_esquerda_y), 20, (0,0,255),3)
                cv2.putText(img,f'Dist maos: {str(dist_maos)}', (10,30), font, 0.75, (100,100,0), 2 )
                cv2.putText(img,f'Maos: {stt_maos}', (220,30), font, 0.75, (100,100,0), 2 )
                cv2.putText(img,f'Braco dir: {stt_braco_dir}', (220,60), font, 0.75, (100,100,0), 2 )
                cv2.putText(img,f'Braco esq: {stt_braco_esq}', (220,90), font, 0.75, (100,100,0), 2 )


                # mp_drawing.draw_landmarks(
                #      img,
                #      results.pose_landmarks,
                #      mp_pose.POSE_CONNECTIONS,
                #      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                #mostra imagem
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                s, jpeg = cv2.imencode('.jpg', img)
                return jpeg.tobytes()

def gen_cam(imagem):
    while True:
        frame = pose_worker()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#----------------------------------------------------------------------------------------------------



app = Flask(__name__)
video_stream = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_cam(video_stream), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, port=5000)
