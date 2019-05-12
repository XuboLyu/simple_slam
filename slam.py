#!/usr/bin/python2.7

import cv2
from frame import Frame, extract, normalize, denormalize, match
import numpy as np

cv2.namedWindow('test')
W = 1920//2
H = 1080//2

F = 250 
K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
print(K)

frames = []
def process_frame(img):
	img = cv2.resize(img, (W, H))
	# frame object, include frame.pts and frame.des
	frame = Frame(img, K)
	frames.append(frame)
	
	if len(frames) <= 1:
		return

	ret, Rt = match(frames[-1], frames[-2])
	for pt1, pt2 in ret:
		u1, v1 = denormalize(frame.K, pt1)
		u2, v2 = denormalize(frame.K, pt2)

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
