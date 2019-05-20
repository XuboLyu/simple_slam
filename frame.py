#!/usr/bin/python2.7
import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform


# IRt = np.eye(4)

IRt = np.array([[1.,0.,0.,0.],
				[0.,1.,0.,0.],
				[0.,0.,1.,0.],
				[0.,0.,0.,1.]])
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
	
	newrow = [0, 0, 0, 1]
	Rt = np.vstack([Rt, newrow])
	return Rt

# match two different frames
def match(f1, f2):
	
	# matching
	bf = cv2.BFMatcher(cv2.NORM_HAMMING)
	matches = bf.knnMatch(f1.des, f2.des, k=2)
	
	ret = []
	idx1, idx2 = [], []
	# Lowe's ratio check 
	for m,n in matches:
		# print(m,n)
		if m.distance < 0.75 * n.distance:
			idx1.append(m.queryIdx)
			idx2.append(m.trainIdx)
			
			p1 = f1.pts[m.queryIdx]
			p2 = f2.pts[m.trainIdx]
			ret.append((p1, p2))
	
	assert len(ret) > 10
	ret = np.array(ret)
	idx1 = np.array(idx1)
	idx2 = np.array(idx2)
	
	# filtering using ransac and essential matrix
	# print(ret)
	model, inliers = ransac(
					(ret[:,0], ret[:, 1]), 
					# FundamentalMatrixTransform, 
					EssentialMatrixTransform,
					min_samples=8, 
					residual_threshold=0.005, 
					max_trials=200)
	
	# print("inliers:", inliers)
	
	ret = ret[inliers]

	# obtain rotation and translation 
	Rt = extractRt(model.params)
	# print(Rt)

	# return the indices of matching points in two frames
	return idx1[inliers], idx2[inliers], Rt

class Frame(object):
	def __init__(self, mapp, img, K):
		# K is camera intrinsic param
		self.K = K
		self.Kinv = np.linalg.inv(self.K)
		self.pose = IRt
		
		# Here the pts is not normalized yet
		pts, self.des = extract(img)

		# (normalized!!) pts for computing essential matrix
		self.pts = normalize(self.Kinv, pts)
		
		self.id = len(mapp.frames)
		mapp.frames.append(self)



