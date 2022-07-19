import cv2
import mediapipe as mp
import numpy as np
from bubbles import speech_bubble, ml_text

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

txt = ml_text(["sample text", "Next line", "回転寿司"],  
  text_color=(0,0,0))

bubble = speech_bubble(txt, 
  (100, 100),
  max_txt_sz= (200, 200),
  tail_end = (100, 100), 
  angry = False)

cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    new_words = []
    success, image = cap.read()
    h, w, _ = image.shape
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    key = cv2.waitKey(5)

    if key == ord('a'):
      new_words = ["abc"]
    if key == ord('b'):
      new_words = ["abc", "def"]
    if key == ord('n'):
      new_words = ["abc", "def", "ghi"]
    if key == ord('l'):
      new_words = ["abc", "def", "ghi","abc", "def", "ghi","abc", "def", "ghi","abc", "def", "ghi"]
    if key == 27:
      break

    if results.multi_face_landmarks:
      face_landmarks = results.multi_face_landmarks[0]
      bubble.update(new_words, face_landmarks, True, (w,h), 30)

      # Draw bubble with text
      image = bubble.draw(image)
      
    cv2.imshow('MediaPipe Holistic', image)
cap.release()