import cv2
import mediapipe as mp
import numpy as np
from cv2 import VideoCapture
from cv2 import waitKey


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose #포즈추정모델을 가져온다. 

# video feed
cap = VideoCapture(0)  # 비디오 캡처 장치 설정, 매개변수로 비디오 캡처 장치 수 사용
while cap.isOpened():   # 비디오 캡처 장치가 열려있는 동안 실행
    ret, frame = cap.read() # 비디오 장치로 얻은 캡처 읽기, frame에는 얻은 캡처 이미지가 잇다. 
    cv2.imshow('Mediapipe Feed', frame) # frame 시각화하기
    if waitKey(10) & 0xFF == ord('q'):  # 피드 중단 후 실해할 내용, q나 그 키중 하나를 누를 시 화면을 닫으려는 여부 처리
        break
# 비디오장치 해제 및 윈도우 파괴
cap.release()
cv2.destroyAllWindows()

