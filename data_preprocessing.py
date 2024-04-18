import os

import cv2
import numpy as np
from utils import get_face_landmarks

DATA_DIR = './data'
output = []
for emotion_index, emotion in enumerate(os.listdir(DATA_DIR)):
    for image_path_ in os.listdir(os.path.join(DATA_DIR, emotion)):
        image_path = os.path.join(DATA_DIR, emotion, image_path_)
        
        image = cv2.imread(image_path)
        
        face_landmarks = get_face_landmarks(image)
        
        if len(face_landmarks) == 1404:
            face_landmarks.append(int(emotion_index))
            output.append(face_landmarks)
        
np.savetxt('data.txt', np.asarray(output))