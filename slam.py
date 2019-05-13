#!/usr/bin/python2.7

import cv2
from frame import Frame, extract, normalize, denormalize, match, IRt
import numpy as np

cv2.namedWindow('test')
W = 1920//2
H = 1080//2

F = 250 
K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
# print(K)

class Map(object):
	def __init__(self):
		self.frames = []
		self.points = []
	
	def display(self):
		for f in self.frames:
			print("current frame:", f.id)
			print("current pose:", f.pose)
mapp = Map()

class Point(object):
	# A Point is a 3-D point in world coordinate
	# each point is observed in multiple frames
	def __init__(self, mapp, loc):
		self.xyz = loc
		self.frames = []
		self.idxs = []

		self.id = len(mapp.points)
		mapp.points.append(self)

	def add_observation(self, frame, idx):
		self.frames.append(frame)
		self.idxs.append(idx)
		
def triangulate(pose1, pose2, pts1, pts2):
	return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T
		
def process_frame(img):
	img = cv2.resize(img, (W, H))
	# frame object, include frame.pts and frame.des
	frame = Frame(mapp, img, K)
	
	if len(mapp.frames) <= 1:
		return
	f1 = mapp.frames[-1]
	f2 = mapp.frames[-2]


	idx1, idx2, Rt = match(f1, f2)
	f1.pose = np.dot(Rt, f2.pose)
	# print(Rt)
	# print(pts.shape)
	
	# homogenous 3-D coords
	pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
	pts4d /= pts4d[:, 3:]
	
	# reject pts without enough "parallax"
	# reject points behind the camera
	good_pts4d = (np.abs(pts4d[:,3]) > 0.005) & (pts4d[:,2] > 0)
	
	for i,p in enumerate(pts4d):
		if not good_pts4d[i]:
			continue
		pt = Point(mapp, p)
		pt.add_observation(f1, idx1[i])
		pt.add_observation(f2, idx2[i])

	# visualize pts on screen
	for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
		u1, v1 = denormalize(frame.K, pt1)
		u2, v2 = denormalize(frame.K, pt2)

		cv2.circle(img, (u1,v1), color=(0,255,0), radius=3)
		cv2.line(img, (u1,v1), (u2,v2), color=(0,0,255))

	cv2.imshow('test', img)
	cv2.waitKey(1)
	
	mapp.display()

if __name__ == "__main__":
	cap = cv2.VideoCapture("test.mp4")
	while cap.isOpened():
		ret,frame = cap.read()
		if ret == True:
			process_frame(frame)
		else:
			break
