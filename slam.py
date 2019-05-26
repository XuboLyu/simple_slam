#!/usr/bin/python2.7

import cv2
from frame import Frame, extract, normalize, denormalize, match, IRt
import numpy as np

import OpenGL.GL as gl
import pangolin


cv2.namedWindow('test')
W = 1920//2
H = 1080//2

F = 500 
K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
# print(K)

from multiprocessing import Process, Queue


class Map(object):
	def __init__(self):
		self.frames = []
		self.points = []
		# self.viewer_init()
		self.state = None
		self.q = Queue()

		p = Process(target=self.viewer_thread, args=(self.q,))
		p.daemon = True
		p.start()

	def viewer_thread(self, q):
		self.viewer_init(1024, 768)
		while 1:
			self.viewer_refresh(q)

	def viewer_init(self, w, h):
		pangolin.CreateWindowAndBind('main',w,h)
		gl.glEnable(gl.GL_DEPTH_TEST)

		self.scam = pangolin.OpenGlRenderState(
			pangolin.ProjectionMatrix(w,h,420,420,w//2,h//2,0.2,10000),
			pangolin.ModelViewLookAt(0,-10,-8,
								 	 0, 0, 0,
									 0, -1, 0))
		self.handler = pangolin.Handler3D(self.scam)
		
		# Create Interactive View in window
		self.dcam = pangolin.CreateDisplay()
		self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -w/h)
		self.dcam.SetHandler(self.handler)

	def viewer_refresh(self, q):
		if self.state == None or not q.empty():
			self.state = q.get()
		 
		# turn state into points, np.asarray() keep all 'd' as array, not matrix
		# ppts = np.array([np.asarray(d)[:3,3] for d in self.state[0]])
		# spts = np.array(self.state[1])

		gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
		gl.glClearColor(1.0, 1.0, 1.0, 1.0)
		self.dcam.Activate(self.scam)

		# draw pose
		gl.glColor3f(0.0, 1.0, 0.0)
		pangolin.DrawCameras(self.state[0])
		
		# draw key points
		gl.glPointSize(2)
		gl.glColor3f(1.0, 0.0, 0.0)
		pangolin.DrawPoints(self.state[1])
	
		pangolin.FinishFrame()

	def display(self):
		poses, pts = [], []

		for f in self.frames:
			poses.append(f.pose)
		for p in self.points:
			pts.append(p.xyz)
		
		self.q.put((np.array(poses), np.array(pts)))
		# self.state = poses, pts
		# self.viewer_refresh()

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
		# print(spts.shape)
	# print(pts.shape)
	
	# homogenous 3-D coords
	pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
	pts4d /= pts4d[:, 3:]
	
	# rject pts without enough "parallax"
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
	cap = cv2.VideoCapture("test_ohio.mp4")
	while cap.isOpened():
		ret,frame = cap.read()
		if ret == True:
			process_frame(frame)
		else:
			break
