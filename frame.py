#!/usr/bin/python2.7
import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

# utility functions
def add_ones(x):
	return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
	
def normalize(Kinv, pt):
	return np.dot(Kinv, add_ones(pt).T).T[:, 0:2]

def denormalize(K, pt):
	ret = np.dot(K, np.array([pt[0], pt[1], 1.0]))
	return int(round(ret[0])), int(round(ret[1])) 


def extract(img):
	orb = cv2.ORB_create()

	# detection	
	pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)

	# extraction
	kps = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], _size=20) for pt in pts]
	kps, des = orb.compute(img,kps)
	
	return np.array([(kp.pt[0],kp.pt[1]) for kp in kps]), des


def extractRt(E):
	W = np.mat([[0,-1,0], [1,0,0],[0,0,1]],dtype=float)
	U,d,Vt = np.linalg.svd(E)
	assert np.linalg.det(U) > 0 

	if np.linalg.det(Vt) < 0:
		Vt *= -1.0
	R = np.dot(np.dot(U,W),Vt)
	if np.sum(R.diagonal()) < 0:
		R = np.dot(np.dot(U,W.T), Vt)
	t = U[:,2]
	Rt = np.concatenate([R,t.reshape(3,1)],axis=1)

	return Rt

# match two different frames
def match(f1, f2):
	
	# matching
	bf = cv2.BFMatcher(cv2.NORM_HAMMING)
	ret = []
	matches = bf.knnMatch(f1.des, f2.des, k=2)
	
	# Lowe's ratio check 
	for m,n in matches:
		# print(m,n)
		if m.distance < 0.75 * n.distance:
			p1 = f1.pts[m.queryIdx]
			p2 = f2.pts[m.trainIdx]
			ret.append((p1, p2))
	
	assert len(ret) > 0
	ret = np.array(ret)
	
	# filtering using ransac and essential matrix
	# print(ret)
	model, inliers = ransac(
					(ret[:,0], ret[:, 1]), 
					# FundamentalMatrixTransform, 
					EssentialMatrixTransform,
					min_samples=8, 
					residual_threshold=0.005, 
					max_trials=200)


	ret = ret[inliers]

	# obtain rotation and translation 
	Rt = extractRt(model.params)
	# print(Rt)

	return ret, Rt

class Frame(object):
	def __init__(self, img, K):
		# K is camera intrinsic param
		self.K = K
		self.Kinv = np.linalg.inv(self.K)

		# Here the pts is not normalized yet
		pts, self.des = extract(img)

		# (normalized!!) pts for computing essential matrix
		self.pts = normalize(self.Kinv, pts)
	



