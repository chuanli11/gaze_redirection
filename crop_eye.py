import os
import string 
import dlib 
import cv2
import numpy as np


rootDir = '/home/ubuntu/arm/test/rawpred1'
distDir = '/home/ubuntu/arm/test/rawpred1_gaze_patch'


if not os.path.exists(distDir):
    os.makedirs(distDir)

if not os.path.exists(distDir + '/l_eye'):
    os.makedirs(distDir + '/l_eye')

if not os.path.exists(distDir + '/r_eye'):
    os.makedirs(distDir + '/r_eye')

_files = []

list_dirs = os.walk(rootDir)

for root, dirs, files in list_dirs:
	for f in files:
		# print(f)
		if f.endswith('.png'):# and f.path.join('_0P')!=-1:
			_files.append(os.path.join(root,f))


for fp in _files:
	print(fp)

	frame = cv2.imread(fp)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	points_keys = []
	PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(PREDICTOR_PATH)
	rects = detector(gray,1)

	for i in range(len(rects)):
		landmarks = np.matrix([[p.x,p.y] for p in predictor(gray, rects[i]).parts()])
		img = gray.copy()
		for idx, points in enumerate(landmarks):
			pos = (points[0,0],points[0,1])
			points_keys.append(pos)
			cv2.circle(img, pos, 2, (255,0,0),-1)

	eye_l = landmarks[36:41].astype(np.int32)
	eye_r = landmarks[42:48].astype(np.int32)

	(x_l,y_l), r_l =cv2.minEnclosingCircle(eye_l)
	(x_r,y_r), r_r =cv2.minEnclosingCircle(eye_r)
	x_l, x_r = int(x_l), int(x_r)
	y_l, y_r = int(y_l), int(y_r)
	r_l, r_r = int(1.7*r_l), int(1.7*r_r)

	eye_l_img = frame[y_l-r_l:y_l+r_l, x_l-r_l:x_l+r_l]
	eye_r_img = frame[y_r-r_r:y_r+r_r, x_r-r_r:x_r+r_r]
	
	_,fn = os.path.split(fp)

	dst_l = os.path.join(distDir, 'l_eye')
	dst_r = os.path.join(distDir, 'r_eye')

	cv2.imwrite(os.path.join(dst_l, fn), eye_l_img)
	cv2.imwrite(os.path.join(dst_r, fn), eye_r_img)