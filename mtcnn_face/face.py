# coding=utf-8
"""Face Detection and Recognition"""

import os
import pickle
import time

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc

import mtcnn_face.facenet as facenet
from mtcnn_face.Detection import nms

gpu_memory_fraction = 0.3
debug = False


class Face:
	def __init__(self):
		self.name = None
		self.confidence = None
		self.bounding_box = None
		self.image = None
		self.container_image = None
		self.embedding = None
#

# class Recognition:
# 	def __init__(self):
#
# 		time_detect = time.time()
# 		self.detect = Detection()
# 		print("Detection cost time is equal to {}".format(time.time() - time_detect))
#
#
# 	def add_identity(self, image, person_name):
# 		faces = self.detect.find_faces(image)
#
# 		if len(faces) == 1:
# 			face = faces[0]
# 			face.name = person_name
# 			face.confidence = 0
# 			face.embedding = self.encoder.generate_embedding(face)
# 			return faces
#
# 	def identify(self, image):
# 		faces = self.detect.find_faces(image)
#
# 		for i, face in enumerate(faces):
# 			if debug:
# 				cv2.imshow("Face: " + str(i), face.image)
# 			face.embedding = self.encoder.generate_embedding(face)
# 			face.name, face.confidence = self.identifier.identify(face)
#
# 		return faces
#



class Detection(object):

	def __init__(self,
				 min_face_size=20,
				 stride=2,
				 threshold=[0.6, 0.7, 0.7],
				 scale_factor=0.79,
				 face_crop_margin = 32,
				 face_crop_size = 256
				 ):

		from mtcnn_face.Detection.detector import Detector
		from mtcnn_face.Detection.fcn_detector import FcnDetector
		from mtcnn_face.mtcnn_model import P_Net, R_Net, O_Net

		slide_window = False
		prefix = ['./mtcnn_face/MTCNN_model/PNet_landmark/PNet', './mtcnn_face/MTCNN_model/RNet_landmark/RNet',
				  './mtcnn_face/MTCNN_model/ONet_landmark/ONet']
		epoch = [18, 14, 16]
		batch_size = [2048, 256, 16]
		model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]

		self.pnet_detector = FcnDetector(P_Net, model_path[0])
		self.rnet_detector = Detector(R_Net, 24, batch_size[1], model_path[1])
		self.onet_detector = Detector(O_Net, 48, batch_size[2], model_path[2])
		self.min_face_size = min_face_size
		self.stride = stride
		self.thresh = threshold
		self.scale_factor = scale_factor
		self.slide_window = slide_window
		self.face_crop_margin = face_crop_margin
		self.face_crop_size = face_crop_size



	def cv_imread(self, path):
		cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
		return cv_img


	def convert_to_square(self, bbox):
		"""
            convert bbox to square
        Parameters:
        ----------
            bbox: numpy array , shape n x 5
                input bbox
        Returns:
        -------
            square bbox
        """
		square_bbox = bbox.copy()

		h = bbox[:, 3] - bbox[:, 1] + 1
		w = bbox[:, 2] - bbox[:, 0] + 1
		max_side = np.maximum(h, w)
		square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
		square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
		square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
		square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
		return square_bbox

	def calibrate_box(self, bbox, reg):
		"""
            calibrate bboxes
        Parameters:
        ----------
            bbox: numpy array, shape n x 5
                input bboxes
            reg:  numpy array, shape n x 4
                bboxes adjustment
        Returns:
        -------
            bboxes after refinement
        """

		bbox_c = bbox.copy()
		w = bbox[:, 2] - bbox[:, 0] + 1
		w = np.expand_dims(w, 1)
		h = bbox[:, 3] - bbox[:, 1] + 1
		h = np.expand_dims(h, 1)
		reg_m = np.hstack([w, h, w, h])
		aug = reg_m * reg
		bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
		return bbox_c

	def generate_bbox(self, cls_map, reg, scale, threshold):
		"""
            generate bbox from feature cls_map
        Parameters:
        ----------
            cls_map: numpy array , n x m
                detect score for each position
            reg: numpy array , n x m x 4
                bbox
            scale: float number
                scale of this detection
            threshold: float number
                detect threshold
        Returns:
        -------
            bbox array
        """
		stride = 2
		# stride = 4
		cellsize = 12
		# cellsize = 25

		t_index = np.where(cls_map > threshold)

		# find nothing
		if t_index[0].size == 0:
			return np.array([])
		# offset
		dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]

		reg = np.array([dx1, dy1, dx2, dy2])
		score = cls_map[t_index[0], t_index[1]]
		boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
								 np.round((stride * t_index[0]) / scale),
								 np.round((stride * t_index[1] + cellsize) / scale),
								 np.round((stride * t_index[0] + cellsize) / scale),
								 score,
								 reg])

		return boundingbox.T

	# pre-process images
	def processed_image(self, img, scale):
		height, width, channels = img.shape
		new_height = int(height * scale)  # resized new height
		new_width = int(width * scale)  # resized new width
		new_dim = (new_width, new_height)
		img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
		img_resized = (img_resized - 127.5) / 128
		return img_resized

	def pad(self, bboxes, w, h):
		"""
            pad the the bboxes, alse restrict the size of it
        Parameters:
        ----------
            bboxes: numpy array, n x 5
                input bboxes
            w: float number
                width of the input image
            h: float number
                height of the input image
        Returns :
        ------
            dy, dx : numpy array, n x 1
                start point of the bbox in target image
            edy, edx : numpy array, n x 1
                end point of the bbox in target image
            y, x : numpy array, n x 1
                start point of the bbox in original image
            ex, ex : numpy array, n x 1
                end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1
                height and width of the bbox
        """
		tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
		num_box = bboxes.shape[0]

		dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
		edx, edy = tmpw.copy() - 1, tmph.copy() - 1

		x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

		tmp_index = np.where(ex > w - 1)
		edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
		ex[tmp_index] = w - 1

		tmp_index = np.where(ey > h - 1)
		edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
		ey[tmp_index] = h - 1

		tmp_index = np.where(x < 0)
		dx[tmp_index] = 0 - x[tmp_index]
		x[tmp_index] = 0

		tmp_index = np.where(y < 0)
		dy[tmp_index] = 0 - y[tmp_index]
		y[tmp_index] = 0

		return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
		return_list = [item.astype(np.int32) for item in return_list]

		return return_list

	def detect_pnet(self, im):
		"""Get face candidates through pnet

        Parameters:
        ----------
        im: numpy array
            input image array

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
		# h, w, c = im.shape
		net_size = 12

		current_scale = float(net_size) / self.min_face_size  # find initial scale
		# print("current_scale", net_size, self.min_face_size, current_scale)
		im_resized = self.processed_image(im, current_scale)
		current_height, current_width, _ = im_resized.shape
		# fcn
		all_boxes = list()
		while min(current_height, current_width) > net_size:
			# return the result predicted by pnet
			# cls_cls_map : H*w*2
			# reg: H*w*4
			cls_cls_map, reg = self.pnet_detector.predict(im_resized)
			# boxes: num*9(x1,y1,x2,y2,score,x1_offset,y1_offset,x2_offset,y2_offset)
			boxes = self.generate_bbox(cls_cls_map[:, :, 1], reg, current_scale, self.thresh[0])

			current_scale *= self.scale_factor
			im_resized = self.processed_image(im, current_scale)
			current_height, current_width, _ = im_resized.shape

			if boxes.size == 0:
				continue
			keep = nms.py_nms(boxes[:, :5], 0.5, 'Union')
			boxes = boxes[keep]
			all_boxes.append(boxes)

		if len(all_boxes) == 0:
			return None, None, None

		all_boxes = np.vstack(all_boxes)

		# merge the detection from first stage
		keep = nms.py_nms(all_boxes[:, 0:5], 0.7, 'Union')
		all_boxes = all_boxes[keep]
		boxes = all_boxes[:, :5]

		bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
		bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1

		# refine the boxes
		boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
							 all_boxes[:, 1] + all_boxes[:, 6] * bbh,
							 all_boxes[:, 2] + all_boxes[:, 7] * bbw,
							 all_boxes[:, 3] + all_boxes[:, 8] * bbh,
							 all_boxes[:, 4]])
		boxes_c = boxes_c.T

		return boxes, boxes_c, None

	def detect_rnet(self, im, dets):
		"""Get face candidates using rnet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of pnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
		h, w, c = im.shape
		dets = self.convert_to_square(dets)
		dets[:, 0:4] = np.round(dets[:, 0:4])

		[dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
		num_boxes = dets.shape[0]
		cropped_ims = np.zeros((num_boxes, 24, 24, 3), dtype=np.float32)
		for i in range(num_boxes):
			tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
			tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
			cropped_ims[i, :, :, :] = (cv2.resize(tmp, (24, 24)) - 127.5) / 128
		# cls_scores : num_data*2
		# reg: num_data*4
		# landmark: num_data*10
		cls_scores, reg, _ = self.rnet_detector.predict(cropped_ims)
		cls_scores = cls_scores[:, 1]
		keep_inds = np.where(cls_scores > self.thresh[1])[0]
		if len(keep_inds) > 0:
			boxes = dets[keep_inds]
			boxes[:, 4] = cls_scores[keep_inds]
			reg = reg[keep_inds]
		# landmark = landmark[keep_inds]
		else:
			return None, None, None

		keep = nms.py_nms(boxes, 0.6)
		boxes = boxes[keep]
		boxes_c = self.calibrate_box(boxes, reg[keep])
		return boxes, boxes_c, None

	def detect_onet(self, im, dets):
		"""Get face candidates using onet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of rnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
		h, w, c = im.shape
		dets = self.convert_to_square(dets)
		dets[:, 0:4] = np.round(dets[:, 0:4])
		[dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
		num_boxes = dets.shape[0]
		cropped_ims = np.zeros((num_boxes, 48, 48, 3), dtype=np.float32)
		for i in range(num_boxes):
			tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
			tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
			cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 48)) - 127.5) / 128

		cls_scores, reg, landmark = self.onet_detector.predict(cropped_ims)
		# prob belongs to face
		cls_scores = cls_scores[:, 1]
		keep_inds = np.where(cls_scores > self.thresh[2])[0]
		if len(keep_inds) > 0:
			# pickout filtered box
			boxes = dets[keep_inds]
			boxes[:, 4] = cls_scores[keep_inds]
			reg = reg[keep_inds]
			landmark = landmark[keep_inds]
		else:
			return None, None, None

		# width
		w = boxes[:, 2] - boxes[:, 0] + 1
		# height
		h = boxes[:, 3] - boxes[:, 1] + 1
		landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
		landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
		boxes_c = self.calibrate_box(boxes, reg)

		boxes = boxes[nms.py_nms(boxes, 0.6, "Minimum")]
		keep = nms.py_nms(boxes_c, 0.6, "Minimum")
		boxes_c = boxes_c[keep]
		landmark = landmark[keep]
		return boxes, boxes_c, landmark

	# use for video
	def detect(self, img):
		"""Detect face over image
        """
		# ################################## add by libo
		#         factor_count = 0
		#         total_boxes = np.empty((0, 9))
		#         points = np.empty(0)
		#         h = img.shape[0]
		#         w = img.shape[1]
		#         minl = np.amin([h, w])
		#         m = 12.0 / self.min_face_size
		#         minl = minl * m
		#         # creat scale pyramid
		#         scales = []
		#         while minl >= 12:
		#             scales += [m * np.power(self.scale_factor, factor_count)]
		#             minl = minl * self.scale_factor
		#             factor_count += 1
		#         boxes_c = []
		#             # first stage
		#         for j in range(len(scales)):
		#             scale = scales[j]
		#             hs = int(np.ceil(h * scale))
		#             ws = int(np.ceil(w * scale))
		#             im_data = imresample(img, (hs, ws))
		#             im_data = (im_data - 127.5) * 0.0078125
		#             # img_x = np.expand_dims(im_data, 0)
		#             # img_y = np.transpose(img_x, (0, 2, 1, 3))
		#             boxes, boxes_a, _ = self.detect_pnet(im_data)
		#
		#             boxes_c.append(boxes_a)
		#             break
		#     #######################################3
		t = time.time()

		# pnet
		t1 = 0
		if self.pnet_detector:
			boxes, boxes_c, _ = self.detect_pnet(img)
			if boxes_c is None:
				return np.array([]), np.array([])

			t1 = time.time() - t
			t = time.time()

		# rnet
		t2 = 0
		if self.rnet_detector:
			boxes, boxes_c, _ = self.detect_rnet(img, boxes_c)
			if boxes_c is None:
				return np.array([]), np.array([])

			t2 = time.time() - t
			t = time.time()

		# onet
		t3 = 0
		if self.onet_detector:
			boxes, boxes_c, landmark = self.detect_onet(img, boxes_c)
			if boxes_c is None:
				return np.array([]), np.array([])

			t3 = time.time() - t
			t = time.time()
			print(
				"time cost " + '{:.3f}'.format(t1 + t2 + t3) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2,
																												t3))

		return boxes_c, landmark

	def find_faces(self, im):
		faces = []
		sum_time = 0
		im = np.array(im)
		allrect = im.shape[0] * im.shape[1]
		# pnet
		t1 = 0
		if self.pnet_detector:
			t = time.time()
			# ignore landmark
			boxes, boxes_c, landmark = self.detect_pnet(np.array(im))
			t1 = time.time() - t
			sum_time += t1
			if boxes_c is None:
				return faces

		# rnet
		t2 = 0
		if self.rnet_detector:
			t = time.time()
			# ignore landmark
			boxes, boxes_c, landmark = self.detect_rnet(im, boxes_c)
			t2 = time.time() - t
			sum_time += t2
			if boxes_c is None:
				return faces
		# onet
		t3 = 0
		if self.onet_detector:
			t = time.time()
			boxes, boxes_c, landmark = self.detect_onet(im, boxes_c)
			t3 = time.time() - t
			sum_time += t3
			if boxes_c is None:
				return faces

		for box in boxes_c:
			face = Face()
			face.container_image = im
			face.bounding_box = np.zeros(4, dtype=np.int32)

			img_size = np.asarray(im.shape)[0:2]
			face.bounding_box[0] = np.maximum(box[0] - self.face_crop_margin / 2, 0)
			face.bounding_box[1] = np.maximum(box[1] - self.face_crop_margin / 2, 0)
			face.bounding_box[2] = np.minimum(box[2] + self.face_crop_margin / 2, img_size[1])
			face.bounding_box[3] = np.minimum(box[3] + self.face_crop_margin / 2, img_size[0])
			cropped = im[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
			face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

			faces.append(face)

		return faces


	def detect_face_ratio(self, im):
		all_boxes = []  # save each image's bboxes
		landmarks = []
		sum_time = 0
		im = np.array(im)
		allrect = im.shape[0] * im.shape[1]
		# pnet
		t1 = 0
		if self.pnet_detector:
			t = time.time()
			# ignore landmark
			boxes, boxes_c, landmark = self.detect_pnet(np.array(im))
			t1 = time.time() - t
			sum_time += t1
			if boxes_c is None:
				print("boxes_c is None...")
				all_boxes.append(np.array([]))
				# pay attention
				landmarks.append(np.array([]))
				return 0

		# rnet
		t2 = 0
		if self.rnet_detector:
			t = time.time()
			# ignore landmark
			boxes, boxes_c, landmark = self.detect_rnet(im, boxes_c)
			t2 = time.time() - t
			sum_time += t2
			if boxes_c is None:
				all_boxes.append(np.array([]))
				landmarks.append(np.array([]))
				return 0
		# onet
		t3 = 0
		if self.onet_detector:
			t = time.time()
			boxes, boxes_c, landmark = self.detect_onet(im, boxes_c)
			t3 = time.time() - t
			sum_time += t3
			if boxes_c is None:
				all_boxes.append(np.array([]))
				landmarks.append(np.array([]))
				return 0

		radio = 0
		for box in boxes_c:
			detal_x = abs(box[2] - box[0])
			detal_y = abs(box[3] - box[1])
			if radio < float((detal_x * detal_y) / allrect):
				radio = float((detal_x * detal_y) / allrect)

		return radio


	def detect_face_ratio_path(self, path):
		all_boxes = []  # save each image's bboxes
		landmarks = []
		sum_time = 0
		im = self.cv_imread(path)
		allrect = im.shape[0] * im.shape[1]
		# pnet
		t1 = 0
		if self.pnet_detector:
			t = time.time()
			# ignore landmark
			boxes, boxes_c, landmark = self.detect_pnet(np.array(im))
			t1 = time.time() - t
			sum_time += t1
			if boxes_c is None:
				print("boxes_c is None...")
				all_boxes.append(np.array([]))
				# pay attention
				landmarks.append(np.array([]))
				return 0

		# rnet
		t2 = 0
		if self.rnet_detector:
			t = time.time()
			# ignore landmark
			boxes, boxes_c, landmark = self.detect_rnet(im, boxes_c)
			t2 = time.time() - t
			sum_time += t2
			if boxes_c is None:
				all_boxes.append(np.array([]))
				landmarks.append(np.array([]))
				return 0
		# onet
		t3 = 0
		if self.onet_detector:
			t = time.time()
			boxes, boxes_c, landmark = self.detect_onet(im, boxes_c)
			t3 = time.time() - t
			sum_time += t3
			if boxes_c is None:
				all_boxes.append(np.array([]))
				landmarks.append(np.array([]))
				return 0

		radio = 0
		for box in boxes_c:
			detal_x = abs(box[2] - box[0])
			detal_y = abs(box[3] - box[1])
			if radio < float((detal_x * detal_y) / allrect):
				radio = float((detal_x * detal_y) / allrect)

		return radio

	# TODO: convert PIL to opencv image
	def detect_face_ratio_path_pil(self, image):
		all_boxes = []  # save each image's bboxes
		landmarks = []
		sum_time = 0

		pil_image = image.convert('RGB')
		opencv_image = np.array(pil_image)
		im = opencv_image[:, :, ::-1].copy()

		# im = self.cv_imread(path)
		allrect = im.shape[0] * im.shape[1]
		# pnet
		t1 = 0
		if self.pnet_detector:
			t = time.time()
			# ignore landmark
			boxes, boxes_c, landmark = self.detect_pnet(np.array(im))
			t1 = time.time() - t
			sum_time += t1
			if boxes_c is None:
				print("boxes_c is None...")
				all_boxes.append(np.array([]))
				# pay attention
				landmarks.append(np.array([]))
				return 0

		# rnet
		t2 = 0
		if self.rnet_detector:
			t = time.time()
			# ignore landmark
			boxes, boxes_c, landmark = self.detect_rnet(im, boxes_c)
			t2 = time.time() - t
			sum_time += t2
			if boxes_c is None:
				all_boxes.append(np.array([]))
				landmarks.append(np.array([]))
				return 0
		# onet
		t3 = 0
		if self.onet_detector:
			t = time.time()
			boxes, boxes_c, landmark = self.detect_onet(im, boxes_c)
			t3 = time.time() - t
			sum_time += t3
			if boxes_c is None:
				all_boxes.append(np.array([]))
				landmarks.append(np.array([]))
				return 0

		radio = 0
		for box in boxes_c:
			detal_x = abs(box[2] - box[0])
			detal_y = abs(box[3] - box[1])
			if radio < float((detal_x * detal_y) / allrect):
				radio = float((detal_x * detal_y) / allrect)

		return radio