# 定义层
import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
import os
import cv2
from mtcnn_face import face_model

# 狂阶图片指定尺寸
image_size = 48
target_size = (image_size, image_size) #fixed size for InceptionV3 architecture
age_labels = {'0-2': 0, '4-6': 1, '8-12': 2, '15-20': 3, '25-32': 4, '38-43': 5, '48-53': 6, '60+': 7}
facil_labels = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprised': 5, 'normal': 6}
# facil_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
# 画图函数
def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


def cv_imread(path):
    cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return cv_img

new_labels = {v : k for k, v in age_labels.items()} #value和key相互转换

model = load_model('./age_inception_model/weights.50.hdf5')
rootdir = 'D:/Project/classify/demo/dataset/age/hunhe_test/5/'
list = os.listdir(rootdir)
truenum = 0
falsenum = 0
face_detect_model = face_model.FaceRecognition()
for filename in list:
    path = os.path.join(rootdir, filename)
    path = './1.jpg'
    cv_img = cv_imread(path)
    # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    result = face_detect_model.predict_pil(cv_img, 0.5)

    for face in result:
        # img = face.container_image
        face.image = cv2.resize(face.image, (227, 227))
        # face.image = cv2.cvtColor(face.image, cv2.COLOR_BGR2GRAY)
        x = np.expand_dims(face.image, axis=0)
        # x = preprocess_input(x)

        pre = model.predict(x)
        # print(pre)
        preds = pre[0]
        label = new_labels[np.where(preds==np.max(max(preds),axis=0))[0][0]]
        print(label)

        x1 = face.bounding_box[0]
        x2 = face.bounding_box[2]
        y1 = face.bounding_box[1]
        y2 = face.bounding_box[3]
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        draw_label(cv_img, (x1, y1), label)
    cv2.imshow("result", cv_img)
    key = cv2.waitKey(-1) if rootdir else cv2.waitKey(30)

    if key == 27:  # ESC
        break

