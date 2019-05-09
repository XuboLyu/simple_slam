#!/usr/bin/python2.7

import cv2
from extractor import FeatureExtractor
# import sdl2.ext
import numpy as np

cv2.namedWindow('test')
W = 1920//2
H = 1080//2

F = 250 
K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
print(K)
fe = FeatureExtractor(K)

def process_frame(img):
	img = cv2.resize(img, (W, H))
	matches, pose  = fe.extract(img)

	if pose is None:
		return
	# kp, des = orb.detectAndCompute(img,None)

	for pt1, pt2 in matches:
		# u1,v1 = map(lambda x: int(round(x)), pt1)
		# u2,v2 = map(lambda x: int(round(x)), pt2)
		
		u1, v1 = fe.denormalize(pt1)
		u2, v2 = fe.denormalize(pt2)

		cv2.circle(img, (u1,v1), color=(0,255,0), radius=3)
		cv2.line(img, (u1,v1), (u2,v2), color=(0,0,255))

	cv2.imshow('test', img)
	cv2.waitKey(1)
	# print(img.shape)
	

if __name__ == "__main__":
	cap = cv2.VideoCapture("test.mp4")
	while cap.isOpened():
		ret,frame = cap.read()
		if ret == True:
			process_frame(frame)
		else:
			break
