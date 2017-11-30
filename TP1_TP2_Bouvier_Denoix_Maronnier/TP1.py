print "bonjoour"

class KDNode:
    def __init__(self, pivot = None, axis = None, left_child = None, right_child = None ,parent = None):
        self.pivot       = pivot
        self.axis        = axis
        self.left_child  = left_child
        self.right_child = right_child
        self.parent		 = parent 

def view_kdtree(kdtree, bbox=[[-1., 1.],[-1., 1.],[-1., 1.]], radius=0.05):
    import numpy as np
    import openalea.plantgl.all as pgl

    scene = pgl.Scene()
    sphere = pgl.Sphere(radius,slices=16,stacks=16)
    silver = pgl.Material(ambient=(49,49,49),diffuse=3.,specular=(129,129,129),shininess=0.4)
    gold = pgl.Material(ambient=(63,50,18),diffuse=3.,specular=(160,141,93),shininess=0.4)

    if isinstance(kdtree, KDNode):
        dim = kdtree.axis
        plane_bbox = [b for i,b in enumerate(bbox) if i != dim]
        plane_points = []
        plane_points += [np.insert([plane_bbox[0][0],plane_bbox[1][0]],dim,kdtree.pivot[dim])]
        plane_points += [np.insert([plane_bbox[0][0],plane_bbox[1][1]],dim,kdtree.pivot[dim])]
        plane_points += [np.insert([plane_bbox[0][1],plane_bbox[1][1]],dim,kdtree.pivot[dim])]
        plane_points += [np.insert([plane_bbox[0][1],plane_bbox[1][0]],dim,kdtree.pivot[dim])]

        left_bbox = np.copy(bbox).astype(float)
        right_bbox = np.copy(bbox).astype(float)
        left_bbox[dim,1] = kdtree.pivot[dim]
        right_bbox[dim,0] = kdtree.pivot[dim]

        scene += pgl.Shape(pgl.Translated(kdtree.pivot,sphere),gold)
        scene += view_kdtree(kdtree.left_child, bbox=left_bbox, radius=radius)
        scene += view_kdtree(kdtree.right_child, bbox=right_bbox, radius=radius)
	if dim == 0:
			scene += pgl.Shape(pgl.Polyline(plane_points+[plane_points[0]],width=2),pgl.Material(ambient=(255,0,0)))
	if dim == 1:
			scene += pgl.Shape(pgl.Polyline(plane_points+[plane_points[0]],width=2),pgl.Material(ambient=(0,255,0)))
	if dim == 2:
			scene += pgl.Shape(pgl.Polyline(plane_points+[plane_points[0]],width=2),pgl.Material(ambient=(0,0,255)))
        scene += pgl.Shape(pgl.FaceSet(plane_points,[range(4)]),pgl.Material(ambient=(0,0,0),transparency=0.6))
		
    else:
        assert (type(kdtree) == list) or (isinstance(kdtree,np.ndarray))
        for p in kdtree:
            scene += pgl.Shape(pgl.Translated(p,sphere),silver)

    return scene


def createkdtree (pointList, minbucketsize = 3, depth = 0):
	if len(pointList) > minbucketsize:
		dim = depth%3
		pointList.sort(key= lambda point : point[dim])
		median = len(pointList) / 2
		left_list = pointList[:median]
		right_list = pointList[median+1:]

		self = KDNode()
		self.left_child = createkdtree(left_list, minbucketsize, depth+1)
		self.right_child = createkdtree(right_list, minbucketsize, depth+1)
		self.pivot= pointList[median]
		self.axis = dim
		if isinstance(self.left_child, KDNode ):
			self.left_child.parent = self
		if isinstance(self.right_child, KDNode ):
			self.right_child.parent = self

		return self
	else:
		return pointList

import openalea.plantgl.all as pgl



def closest_point(kdtree, p):
    if isinstance(kdtree, KDNode):
		return brute_force_closest(p,[closest_point(kdtree.left_child), closest_point(kdtree.right_child), kdtree.pivot])
    return brute_force_closest(p, kdtree)






def brute_force_closest(point, pointlist):
    """ Find the closest points of 'point' in 'pointlist' using a brute force approach """
    import sys
    pid, d = -1, sys.maxint
    for i, p in enumerate(pointlist):
        nd = pgl.norm(point-p) 
        if nd < d:
            d = nd
            pid = i
    return pointlist[pid]

def generate_random_point(size=[1,1,1], distribution='uniform'):
    from random import uniform, gauss
    if distribution == 'uniform':
        return (uniform(-size[0],size[0]), uniform(-size[1],size[1]), uniform(-size[2],size[2])) 
    elif distribution == 'gaussian':
        return (gauss(0,size[0]/3.), gauss(0,size[1]/3.), gauss(0,size[1]/3.)) 

def generate_random_pointlist(size=[1,1,1], nb = 15, distribution='uniform'):
    return [generate_random_point(size, distribution=distribution) for i in xrange(nb)]

def test_kdtree(create_kdtree_func, closestpoint_func, nbtest=100, nbpoints=15, size=[1,1,1], minbucketsize=2):
    import time

    points = generate_random_pointlist(nb = nbpoints, size=size, distribution='uniform')
    mkdtree = create_kdtree_func(points, minbucketsize)
    pgl.Viewer.display(view_kdtree(mkdtree, radius=0.03, bbox=[[-float(s),float(s)] for s in size]))
    kdtime, bftime = 0,0
    for i in xrange(nbtest):
        testpoint = generate_random_point(size)
        t = time.time()
        kpoint = closestpoint_func(testpoint, mkdtree)
        kdtime += time.time()-t
        t = time.time()
        bfpoint = brute_force_closest(testpoint, points)
        bftime += time.time()-t
        if kpoint != bfpoint: 
            raise ValueError('Invalid closest point')
    print 'Comparative execution time : KD-Tree [', kdtime,'], BruteForce [', bftime,']'

    return kdtime, bftime

def print_kdtree(kdtree, depth = 0):
    if isinstance(kdtree, KDNode):
        print ('  '*depth) + 'Node :', kdtree.axis,  kdtree.pivot
        print_kdtree(kdtree.left_child, depth+1)
        print_kdtree(kdtree.right_child, depth+1)
    else:
        assert (type(kdtree) == list) or (isinstance(kdtree,np.ndarray))
        print ('  '*depth) + 'Leaf :', kdtree

print_kdtree(createkdtree(generate_random_pointlist()))

view_kdtree(createkdtree(generate_random_pointlist(nb=15)))
test_kdtree(createkdtree,closest_point)
