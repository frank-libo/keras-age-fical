import cv2
import numpy as np
import mtcnn_face.face as face

class FaceRecognition(object):
	def __init__(self):
		self.face_recog = face.Detection()

	def cv_imread(self, path):
		cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
		return cv_img

	def predict(self, path, thresh=0.25):
		cv_img = self.cv_imread(path)
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		# faces = self.face_recog.identify(cv_img)
		faces = self.face_recog.find_faces(cv_img)
		ret_list = []
		if faces:
			for face in faces:
					# ret_list.append(face.image)
					ret_list.append(face)
		return ret_list

	# TODO: convert PIL to opencv image
	def predict_pil(self, image, thresh=0.25):
		# cv_img = self.cv_imread(path)
		# pil_image = image.convert('RGB')
		# opencv_image = np.array(pil_image)
		# opencv_image = opencv_image[:, :, ::-1].copy()
		
		# faces = self.face_recog.find_faces(opencv_image)
		faces = self.face_recog.find_faces(image)
		ret_list = []
		if faces:
			for face in faces:
				# if face.confidence > thresh:
				# 	tup = (face.name, str(face.confidence), str(face.bounding_box.astype(int)))
				# 	ret_list.append(tup)
				ret_list.append(face)
		return ret_list