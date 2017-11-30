import numpy as np


class SegmentImplicitSurface(ImplicitSurface):
	def __init__(self, start_point=[0,0,0], end_point=[0,0,1], radius=1.):

	def potential_evaluator(self):
		def potential(x,y,z):
	            points = np.transpose(np.array([x,y,z]),range(1,np.array(x).ndim+1)+[0])
	            for list2 in points:
	            	for list1 in list2:
	            		for p in range(len(list1)):
	            			target = list1[p]
	            			dP1 = np.dist(start_point, target)
	            			dP2 = np.dist(end_point, target)
	            			dProj = 

	            			result = np.min(np.min(dP1,dP2),dProj)
	            			list1[p]=result
	        return points