from statistics import mode

import cv2
from keras.models import load_model
import numpy as np
from mtcnn_face import face_model


age_labels = {'0-2': 0, '4-6': 1, '8-12': 2, '15-20': 3, '25-32': 4, '38-43': 5, '48-53': 6, '60+': 7}
facil_labels = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprised': 5, 'normal': 6}
new_labels = {v : k for k, v in facil_labels.items()} #value和key相互转换

face_detect_model = face_model.FaceRecognition()
model = load_model('./weights.09.hdf5')
# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)


# starting lists for calculating modes
emotion_window = []

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

# 画图函数
def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


while True:
    rgb_image = video_capture.read()[1]
    # gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    # rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = face_detect_model.predict_pil(rgb_image)

    for face in faces:


        # img = face.container_image
        face.image = cv2.resize(face.image, (380, 380))
        x = np.expand_dims(face.image, axis=0)
        # x = preprocess_input(x)

        pre = model.predict(x)
        # print(pre)
        preds = pre[0]
        label = new_labels[np.where(preds == np.max(max(preds), axis=0))[0][0]]
        print(label)

        x1 = face.bounding_box[0]
        x2 = face.bounding_box[2]
        y1 = face.bounding_box[1]
        y2 = face.bounding_box[3]
        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        draw_label(rgb_image, (x1, y1), label)

    cv2.imshow('window_frame', rgb_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
