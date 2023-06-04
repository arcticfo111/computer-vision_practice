import cv2
import mediapipe as mp
import numpy as np
from cv2 import VideoCapture
from cv2 import waitKey

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose #포즈추정모델을 가져온다. 

# calculate angles 각도 계산 함수 만들기
def calculate_angle(a,b,c):
    # a,b,c 각 landmarks[mp.pose.PoseLandmark.관절이름.value]의 x,y,z
    a = np.array(a)
    b = np.array(b) 
    c = np.array(c) 
    radians = np.arctan2(c[1]-b[1],c[0]-b[0])  - np.arctan2(a[1]-b[1],a[0]-b[0])    # 특정 관절에 대한 라디안 계산 후 0~180으로 변환
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

# video feed
cap = VideoCapture(0)  # 비디오 캡처 장치 설정, 매개변수로 비디오 캡처 장치 수 사용
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:   # 미디어 파이프 피드의 새 인스턴스 설정, min_detection과 min_tracking_confidence는 각 원하는 탐지, 추적 신뢰도이다. 이 코드라인을 pose 변수로 사용하기
    while cap.isOpened():   # 비디오 캡처 장치가 열려있는 동안 실행 
        ret, frame = cap.read() # 비디오 장치로 얻은 캡처 읽기, frame에는 얻은 캡처 이미지가 잇다. 
        # 이미지 다시 그리기 -> 이미지를 미디어파이프에 전달하기 위해 , 이미지피드는 bgr형식이므로 -> rgb형식으로
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # 이미지 쓰기 가능여부를 불가로 설정
        
        # make detection
        results = pose.process(image)   # 위에서 정의한 pose 모델 사용으로 포즈도트가 이미지를 처리 -> 감지해서 결과 저장
        
        # recolor back to rgb
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # extract landmarks, 랜드마크 추출
        try:
            landmarks = results.pose_landmarks.landmark
            # 왼쪽 어깨, 팔꿈치, 손목 사이 각도 측정
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angle = calculate_angle(shoulder, elbow, wrist) # 어깨, 팔꿈피, 손목 좌표를 각도 계산 함수에 보내기
            # visualize angle
            cv2.putText(image, str(angle), tuple(np.multiply(elbow, [1280,720]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)    #[1280,720]에는 웹캠 크기
        except:
            pass

        # render detection, 이미지 탐지 적용
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,227,66), thickness=2, circle_radius=2), # landmark drawing spec 으로 화면에 표시 될 색 등 설정
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)  # connection drawing spec모델 내의 다른 라인 색 등 설정
                                  )   # 신체부위 감지해서 화면에 선과 점 등으로 표현
        cv2.imshow('Mediapipe Feed', image)
        
        if waitKey(10) & 0xFF == ord('q'):  # 피드 중단 후 실해할 내용, q나 그 키중 하나를 누를 시 화면을 닫으려는 여부 처리
            break

# 비디오장치 해제 및 윈도우 파괴
cap.release()
cv2.destroyAllWindows()


