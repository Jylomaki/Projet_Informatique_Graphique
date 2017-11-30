import numpy as np
import math
import scipy.ndimage as nd
from scipy.cluster.vq import vq

from gmap_tools import read_ply_mesh, array_unique
from basicshapes import topopolygon
from array_dict import array_dict
from PyQt4.QtGui import *
import openalea.plantgl.all as pgl

class Point:
	def __init__(self,x,y,z):
		self.x = x
		self.y = y
		self.z = z
	def __add__(self, a):
		self.x += a.x
		self.y += a.y
		self.z += a.z
	def __mul__(self, a):
		self.x *= a
		self.y *= a
		self.z *= a

def factorial(n):
	if (n == 1 or n == 0):
		return 1
	else:
		return factorial(n - 1) * n


def bernstein(i, n, u):
	""" Bernstein Polynomials
	Initial conditions : $B_0^0 = 1$, $B_{-1}^n = B_{n+1}^n = 0$
	$B_i^{n+1}(u) =uB_{i-1}^n(u) + (1-u)B_{i}^n(u)$ 
	"""
	return factorial(n)/float((factorial(i)*factorial(n-i)))*np.power(u,i)*np.power((1-u),(n-i))


def spline(control_points, nb_points=10):
    """ Evaluate the spline curve defined by its control points. 
    return a list of nb_points.
    """
    list_of_points=[]
    for i in range(nb_points):
    	x=0
    	y=0
    	z=0
    	for j in range(len(control_points)):
			B = bernstein(j,len(control_points)-1,i/float(nb_points))
			x += B * control_points[j][0]
			y += B * control_points[j][1]
			z += B * control_points[j][2]
        list_of_points.append((x,y,z))
    return list_of_points

def BezierCurveByCasteljauRec(i,j,u,P):
	if i==0:
		return P[0][j]
	else:
		A = BezierCurveByCasteljauRec(i-1,j,u,P)
		B = BezierCurveByCasteljauRec(i-1,j+1,u,P)
		A.x *= (1-u)
		A.y *= (1-u)
		A.z *= (1-u)

		B.x *= u
		B.y *= u
		B.z *= u
		C = A
		C.x = A.x + B.x
		C.y = A.y + B.y
		C.z = A.z + B.z

		return C


def BezierCurveByCasteljau(control_points,nb_points=10):
    P = [[Point(0,0,0) for x in range(nb_points)] for y in range(len(control_points)-1)]
    list_of_points=[]
    for k in range(nb_points):
        u = k/float(nb_points);
        for i in range(len(control_points)):
            P[0][i].x = control_points[i][0];
            P[0][i].y = control_points[i][1];
            P[0][i].z = control_points[i][2];

        po = BezierCurveByCasteljauRec(len(control_points)-1,0,u,P);

        list_of_points.append((po.x,po.y,po.z))
    return list_of_points



def BezierSurfaceintoCurveByCasteljauRec(k,i,j,u,P,master):

    if k==0 :
        return P[0][i][j]
    else:
        if master == 'l':
			A = BezierSurfaceintoCurveByCasteljauRec(k-1,i,j,u,P,master)
			B = BezierSurfaceintoCurveByCasteljauRec(k-1,i+1,j,u,P,master)
			A.x *= (1-u)
			A.y *= (1-u)
			A.z *= (1-u)

			B.x *= u
			B.y *= u
			B.z *= u
			C = A
			C.x = A.x + B.x
			C.y = A.y + B.y
			C.z = A.z + B.z
			return C
        else:
			A = BezierSurfaceintoCurveByCasteljauRec(k-1,i,j,u,P,master)
			B = BezierSurfaceintoCurveByCasteljauRec(k-1,i,j+1,u,P,master)
			A.x *= (1-u)
			A.y *= (1-u)
			A.z *= (1-u)

			B.x *= u
			B.y *= u
			B.z *= u
			C = A
			C.x = A.x + B.x
			C.y = A.y + B.y
			C.z = A.z + B.z
			return C



def BezierSurfaceByCasteljauRec(k,i,j,u,v,P,nbControlPointU,nbControlPointV):
    if k==0:
        if nbControlPointU<nbControlPointV:
            return BezierSurfaceintoCurveByCasteljauRec(1,i,j,v,P,'c') 
        elif nbControlPointU>nbControlPointV:
            return BezierSurfaceintoCurveByCasteljauRec(1,i,j,u,P,'l')             
        else :
            return P[i][j]
    else:
		A = BezierSurfaceByCasteljauRec(k-1,i,j+1,u,v,P,nbControlPointU,nbControlPointV)
		B = BezierSurfaceByCasteljauRec(k-1,i+1,j+1,u,v,P,nbControlPointU,nbControlPointV)
		A.x *= (1-u)
		A.y *= (1-u)
		A.z *= (1-u)

		B.x *= u
		B.y *= u
		B.z *= u
		C = A
		C.x = A.x + B.x
		C.y = A.y + B.y
		C.z = A.z + B.z

		A = BezierSurfaceByCasteljauRec(k-1,i,j,u,v,P,nbControlPointU,nbControlPointV)
		B = BezierSurfaceByCasteljauRec(k-1,i+1,j,u,v,P,nbControlPointU,nbControlPointV)
		A.x *= (1-u)
		A.y *= (1-u)
		A.z *= (1-u)

		B.x *= u
		B.y *= u
		B.z *= u
		D = A
		D.x = A.x + B.x
		D.y = A.y + B.y
		D.z = A.z + B.z

		D.x *= (1-v)
		D.y *= (1-v)
		D.z *= (1-v)

		C.x *= v
		C.y *= v
		C.z *= v

		E = D
		E.x += C.x
		E.y += C.y
		E.z += C.z

		return E

def BezierSurfaceCasteljau(TabControlPoint,nb_points=10):
	nbControlPointU = len(TabControlPoint)
	nbControlPointV = len(TabControlPoint[0])
	P = [[Point(0,0,0) for y in range(nb_points)] for z in range(nb_points)]
	list_of_points = [[(0,0,0) for x in range(nb_points)] for y in range((nb_points))]
	for i in range(nb_points):
	    u=i/float(nb_points)
	    for j in range(nb_points):
	        v=j/float(nb_points)
	        for m in range(nbControlPointU):
	            for n in range(nbControlPointV):
	                P[m][n].x = TabControlPoint[m][n][0]
	                P[m][n].y = TabControlPoint[m][n][1]
	                P[m][n].z = TabControlPoint[m][n][2]
	        
	        po = BezierSurfaceByCasteljauRec(np.minimum(nbControlPointU-1,nbControlPointV-1),0,0,u,v,P,nbControlPointU,nbControlPointV);
	        list_of_points[i][j] = (po.x,po.y,po.z);
	return list_of_points



def plot_spline_crv(ctrls, pts):
    """ 
    Parameters
    ==========
      - ctrl: control points
      - pts : evaluated points on the curve
    """

    scene = pgl.Scene()
    crvContols = pgl.Shape(geometry=pgl.Polyline(ctrls), appearance=pgl.Material((12,125,12)))
    scene.add(crvContols)
    crv = pgl.Shape(geometry=pgl.Polyline(pts), appearance=pgl.Material((125,12,12)))
    scene.add(crv)
    
    # To complete: Draw the control points and the line between each ones.
 
    pgl.Viewer.display(scene)


def plot_spline_surface(ctrl_net, points):
    """
    Parameters
    ==========
      - ctrl_net : the net of control points (list of list)
      - points : a set of evaluated points (list of list)
    """
    scene = pgl.Scene()
    n = len(points)
    m = len(points[0])

    # Compute a mesh (i.e. TriangleSet) for the set of points
    pointList = [pt for rank in points for pt in rank]
    indexList = []

    for i in range(n-1):
        for j in range(m-1):
            ii = i*m+j
            i1 = (ii, ii+1, ii+m)
            i2 = (ii+1,ii+m+1,ii+m)
            indexList.append(i1)
            indexList.append(i2)

    
    surf = pgl.Shape(pgl.TriangleSet(pointList, indexList), appearance=pgl.Material((12,125,12)))
    scene.add(surf)

    
    # plot the control net
    n = len(ctrl_net)
    m = len(ctrl_net[0])
    for pts in ctrl_net:
        crv = pgl.Shape(geometry=pgl.Polyline(pts), appearance=pgl.Material((125,12,12)))
        scene.add(crv)
        for pt in pts:
            scene.add(pgl.Shape(pgl.Translated(pgl.Vector3(pt),pgl.Sphere(radius=0.1))))
            
    for i in range(m):
        pts = [ctrl_net[j][i] for j in range(n)]
        crv = pgl.Shape(geometry=pgl.Polyline(pts), appearance=pgl.Material((12,12,125)))
        scene.add(crv)
        
    pgl.Viewer.display(scene)

control_points = [(np.random.random_sample()*10.0,np.random.random_sample()*10.0,np.random.random_sample()*10.0), (np.random.random_sample()*10.0,np.random.random_sample()*10.0,np.random.random_sample()*10.0), (np.random.random_sample()*10.0,np.random.random_sample()*10.0,np.random.random_sample()*10.0), (np.random.random_sample()*10.0,np.random.random_sample()*10.0,np.random.random_sample()*10.0)]
n = int(np.random.random_sample()*2)+2
m = int(np.random.random_sample()*2)+2
control_points_surface = [[(0,0,0) for x in range(n)] for y in range(n)]
for i in range(n):
	for j in range(m):
		control_points_surface[i][j] = (np.random.random_sample()*10.0,np.random.random_sample()*10.0,np.random.random_sample()*10.0)

#plot_spline_crv(control_points,spline(control_points,255))
#plot_spline_crv(control_points,BezierCurveByCasteljau(control_points,255))
plot_spline_surface(control_points_surface,BezierSurfaceCasteljau(control_points_surface,20))