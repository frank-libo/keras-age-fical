from mtcnn_face import face_model
import cv2
import os


model = face_model.FaceRecognition()
root = 'D:/Project/classify/demo/dataset/age/temp/'
outdir = 'D:/Project/classify/demo/dataset/age/mtcnn_temp/'
files = os.listdir(root)
for temp in files:
        p = os.path.join(root, temp)
        files2 = os.listdir(p)
        for i, temp2 in enumerate(files2):

            src = os.path.join(p, temp2)
            try:
                result = model.predict(src, 0.5)
                label = src.split('\\')[0].split('/')[-1]
                filename = os.path.join(outdir, label) + '/' + src.split('\\')[1]

                if not os.path.exists(os.path.join(outdir, label)):
                    os.makedirs(os.path.join(outdir, label))

                for img in result:
                    cv2.imwrite(filename, img)

                if i %100 == 0:
                    print(i)

            except:
                print(src)




