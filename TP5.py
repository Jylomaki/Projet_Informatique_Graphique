import numpy as np
 
class GMap:
    def __init__(self, degree=2):
        """ 
        Constructor 
        """
 
        self.maxid = 0
        self.alphas = { 0 : {}, 1 : {}, 2 : {} }
        self.positions = {}
 
    def darts(self): 
        """ 
        Return a list of id representing the darts of the structure 
        """
        return self.alphas[0].keys()
 
    def alpha(self, degree, dart):
        """ Return the application of the alpha_deg on dart """
        return self.alphas[degree][dart]
 
    def alpha_composed(self, list_of_alpha_value, dart):
        """ 
        Return the application of a composition of alphas on dart 
        """
        for alpha in list_of_alpha_value:
            dart = self.alpha(alpha, dart)
        return dart
 
    def is_free(self, degree, dart):
        """ 
        Test if dart is free for alpha_degree (if it is a fixed point) 
        """
        return self.alpha(degree,dart) == dart
 
    def add_dart(self):
        """ 
        Create a new dart and return its id. 
        Set its alpha_i to itself (fixed points) 
        """
        dart = self.maxid
        self.maxid += 1
        for degree in self.alphas.keys():
            self.alphas[degree][dart] = dart
        return dart
 
    def is_valid(self):
        """ 
        Test the validity of the structure. 
        Check if there is pending dart for alpha_0 and alpha_1 (fixed point) 
        """
        for dart, alpha_0_of_dart in self.alphas[0].items():
             if dart == alpha_0_of_dart : return False # no fixed point
             if dart != self.alpha(0,alpha_0_of_dart) : return False # alpha_0 is an involution
 
        for dart, alpha_1_of_dart in self.alphas[1].items():
             if dart == alpha_1_of_dart : return False # no fixed point
             if dart != self.alpha(1,alpha_1_of_dart) : return False # alpha_1 is an involution
 
        for dart in self.darts(): # alpha_0 alpha_2 is an involution
            if self.alpha_composed([0,2,0,2],dart) != dart: return False
 
        return True
 
    def link_darts(self,degree, dart1, dart2): 
        """ 
        Link the two darts with a relation alpha_degree
        """
        assert self.is_free(degree,dart1) and self.is_free(degree,dart2)
        self.alphas[degree][dart1] = dart2
        self.alphas[degree][dart2] = dart1

    def force_link_darts(self,degree, dart1, dart2): 
        """ 
        Link the two darts with a relation alpha_degree
        """
        self.alphas[degree][dart1] = dart2
        self.alphas[degree][dart2] = dart1
 
    def print_alphas(self):
        """ 
        Print for each dart, the value of the different alpha applications.
        """ 
        try:
            from colorama import Style, Fore
        except:
            print "Try to install colorama (pip install colorama) for a better-looking display!"
            for d in self.darts():
                print d," | ",self.alpha(0,d),self.alpha(1,d),self.alpha(2,d) # , self.get_position(d)
        else:
            print ""
            for d in self.darts():
                print d," | ",Fore.MAGENTA+str(self.alpha(0,d))," ",Fore.GREEN+str(self.alpha(1,d))," ",Fore.BLUE+str(self.alpha(2,d))," ",Style.RESET_ALL 
 
 
    def orbit(self, dart, list_of_alpha_value):
        """ 
        Return the orbit of dart using a list of alpha relation.
        Example of use : gmap.orbit(0,[0,1]).
        In Python, you can use the set structure to process only once all darts of the orbit.  
        """
        orbit = []
        marked = set([])
        toprocess = [dart]
 
        while len(toprocess)>0:
            d = toprocess.pop(0)
            if not d in marked:
                orbit.append(d)
                marked.add(d)
                for degree in list_of_alpha_value:
                    toprocess.append(self.alpha(degree,d))
 
        return orbit
 
 
    def orderedorbit(self, dart, list_of_alpha_value):
        """
        Return the ordered orbit of dart using a list of alpha relations by applying
        repeatingly the alpha relations of the list to dart.
        Example of use. gmap.orderedorbit(0,[0,1]).
        Warning: No fixed point for the given alpha should be contained.
        """
        orbit = []
        current_dart = dart
        current_alpha_index = 0
        n_alpha = len(list_of_alpha_value)
        while (current_dart != dart) or orbit==[]:
            orbit.append(current_dart)
            current_alpha = list_of_alpha_value[current_alpha_index]
            current_dart = self.alpha(current_alpha,current_dart)
            current_alpha_index = (current_alpha_index+1) % n_alpha
        return orbit
 
 
    def sew_dart(self, degree, dart1, dart2, merge_attribute = True):
        """
        Sew two elements of degree 'degree' that start at dart1 and dart2.
        Determine first the orbits of dart to sew and heck if they are compatible.
        Sew pairs of corresponding darts, and if they have different embedding 
        positions, merge them. 
        """
        if degree == 1:
            self.link_darts(1, dart1, dart2)
        else:
            alpha_list = [0]
            orbit1 = self.orbit(dart1, alpha_list)
            orbit2 = self.orbit(dart2, alpha_list)
            if len(orbit1) != len(orbit2):
                raise ValueError('Incompatible orbits', orbit1, orbit2)
            for d1,d2 in zip(orbit1, orbit2):
                self.link_darts(degree, d1, d2)
                if merge_attribute:
                    d1e = self.get_embedding_dart(d1, self.positions)
                    d2e = self.get_embedding_dart(d2, self.positions)
                    if d1e in self.positions and d2e in self.positions:
                            pos = (self.positions[d1e] + self.positions[d2e]) / 2.
                            del self.positions[d2e]
                            self.positions[d1e] = pos
 
 
 
    def elements(self, degree):
        """ 
        Return one dart per element of degree. For this, consider all darts as initial set S. 
        Take the first dart d, remove from the set all darts of the orbit starting from d and 
        corresponding to element of degree degree. Take then next element from set S and do the 
        same until S is empty. 
        Return all darts d that were used. """
        
        elements = []
        darts = set(self.darts())
 
        list_of_alpha_value = range(3)
        list_of_alpha_value.remove(degree)
 
        while len(darts) > 0:
            dart = darts.pop()
            elementi = self.orbit(dart, list_of_alpha_value)
            darts -= set(elementi)
            elements.append(dart)
 
        return elements
 
 
    def incident_cells(self, dart, degree, incidentdegree):
        """
        Return all the element of degree incidentdegree
        that are incident to the element dart of degree degree.
        (Typically all edges around a point)
        For this iterate over all the dart of the orbit of (dart, degree).
        For each dart d of this orbit, get all the darts coresponding
        to the orbit of the element (d, incidentdegree) and remove them
        from the original set.
        """
        results = []
 
        alphas = range(3)
        alphas.remove(degree) 
 
        incidentalphas = range(3)
        incidentalphas.remove(incidentdegree) 
 
        marked = set()
 
        for d in self.orbit(dart, alphas):
            if not d in marked:
                results.append(d)
                marked |= set(self.orbit(d, incidentalphas))
 
        return results
        
 
    def insert_edge(self, dart):
        """ 
        Insert an edge at the point represented by dart.
        Return a dart corresponding to the dandling edge end.
        """
 
        dart1 = self.alpha(1, dart)
        newdarts = [self.add_dart() for i in xrange(4)]
        
        self.link_darts(0, newdarts[0], newdarts[1])
        self.link_darts(0, newdarts[3], newdarts[2])
        
        self.link_darts(2, newdarts[0], newdarts[3])
        self.link_darts(2, newdarts[1], newdarts[2])
 
        self.alphas[1][dart] = newdarts[0]
        self.alphas[1][newdarts[0]] = dart
 
        self.alphas[1][dart1] = newdarts[3]
        self.alphas[1][newdarts[3]] = dart1
 
        return newdarts[1]
 
 
    def split_face(self, dart1, dart2=None):
        """
        Split face by inserting an edge between dart1 and dart2 
        """
 
        if dart2 is None:
            dart2 = self.alpha_composed([0,1,0],dart1)
 
        dedge = self.insert_edge(dart1)
 
        dart2a1 = self.alpha(1,dart2)
        dedgea2 = self.alpha(2, dedge)
 
        self.alphas[1][dart2] = dedge
        self.alphas[1][dedge] = dart2
 
        self.alphas[1][dart2a1] = dedgea2
        self.alphas[1][dedgea2] = dart2a1
 
 
    def split_edge(self, dart):
        """ 
        Operator to split an edge. 
        Return a dart corresponding to the new points
        """
        orbit1 = self.orbit(dart,[2])
        orbit2 = self.orbit(self.alpha(0,dart),[2])
 
        newdart1 = [self.add_dart() for i in orbit1]
        newdart2 = [self.add_dart() for i in orbit2]
        
        for d, nd in zip(orbit1+orbit2, newdart1+newdart2):
            self.alphas[0][d] = nd
            self.alphas[0][nd] = d
        
        for nd1, nd2 in zip(newdart1, newdart2):
            self.link_darts(1, nd1, nd2)
        
        for nd in newdart1+newdart2:
            if self.is_free(2, nd) and not self.is_free(2, self.alpha(0, nd)):
                self.link_darts(2,nd, self.alpha(0,self.alpha(2,self.alpha(0,nd))))
 
        return newdart1[0]
 
 
    def get_embedding_dart(self, dart, propertydict ):#OLD
        """ 
        Check if a dart of the orbit representing the vertex has already been 
        associated with a value in propertydict. If yes, return this dart, else
        return the dart passed as argument.
        """
        for d in self.orbit(dart,[1,2]):
            if propertydict.has_key(d):
                return d
        return dart

    def get_embedding_dart(self, dart, propertydict, degree=0):#NEW
        """ 
        Check if a dart of the orbit representing the vertex has already been 
        associated with a value in propertydict. If yes, return this dart, else
        return the dart passed as argument.
        """
        alphas = range(3)
        alphas.remove(degree)
        for d in self.orbit(dart, alphas):
            if propertydict.has_key(d):
                return d
        return dart
 
 
    def get_position(self, dart):
        """
        Retrieve the coordinates associated to the vertex <alpha_1, alpha_2>(dart) 
        """
        return self.positions.get(self.get_embedding_dart(dart,self.positions))
 
	def get_position(self, dart):
		return self.positions.get(self.get_embedding_dart(dart,self.positions,0))
 
 
    def set_position(self, dart, position) :
        """
        Associate coordinates with the vertex <alpha_1,alpha_2>(dart)
        """
        self.positions[self.get_embedding_dart(dart,self.positions)] = position

    def set_position(self, dart, position) :
        """
        Associate coordinates with the vertex &lt;alpha_1,alpha_2&gt;(dart)
        """
        self.positions[self.get_embedding_dart(dart,self.positions)] = np.array(position)
    
 
    def display(self, color = [190,205,205], add = False):
        from openalea.plantgl.all import Scene, Shape, Material, FaceSet, Viewer
        from random import randint
        s = Scene()
        for facedart in self.elements(2):
                lastdart = facedart
                positions = []
                for dart in self.orderedorbit(facedart,[0,1]):
                        if self.alpha(0, dart) != lastdart:
                                positions.append(self.get_position(dart))
                        lastdart = dart
                if color is None:
                        mat = Material((randint(0,255),randint(0,255),randint(0,255)))
                else:
                        mat = Material(tuple(color),diffuse=0.25)
                s.add(Shape(FaceSet(positions, [range(len(positions))]) , mat, facedart ))
        if add : 
            Viewer.add(s)
        else : 
            Viewer.display(s)
 
 
    def element_center(self, dart, degree):
        list_of_alpha_value = range(3)
        list_of_alpha_value.remove(degree)
 
        element_positions = [self.get_position(d) for d in self.orbit(dart,list_of_alpha_value)]
        return np.mean(element_positions,axis=0)
 
 
    def dart_display(self, radius=0.1, coef=0.8, add=False):
        import openalea.plantgl.all as pgl
 
        sphere = pgl.Sphere(radius,slices=16,stacks=16)
        coal = pgl.Material(ambient=(8,10,13),diffuse=3.,specular=(89,89,89),shininess=0.3)
        purple = pgl.Material(ambient=(72,28,72),diffuse=2.,specular=(89,89,89),shininess=0.3)
        green = pgl.Material(ambient=(0,88,9),diffuse=2.,specular=(89,89,89),shininess=0.3)
        blue = pgl.Material(ambient=(9,0,88),diffuse=2.,specular=(89,89,89),shininess=0.3)
 
        font = pgl.Font(size=10)
 
        s = pgl.Scene()
 
        dart_points = {}
 
        for dart in self.darts():
            dart_point = self.get_position(dart)
            dart_face_center = self.element_center(dart,2)
            dart_edge_center = self.element_center(dart,1)
 
            dart_face_point = dart_face_center + coef*(dart_point-dart_face_center)
            dart_face_edge_center = dart_face_center + coef*(dart_edge_center-dart_face_center)
 
            dart_edge_point = dart_face_edge_center + coef*(dart_face_point-dart_face_edge_center)
            dart_middle_edge_point = dart_face_edge_center + 0.33*(dart_edge_point-dart_face_edge_center)
 
            dart_points[dart] = [dart_edge_point,dart_middle_edge_point]
 
            s += pgl.Shape(pgl.Translated(dart_points[dart][0],sphere),coal)
            # s += pgl.Shape(pgl.Translated(np.mean(dart_points[dart],axis=0), pgl.Text(str(dart),fontstyle=font)), coal, id=dart)
            s += pgl.Shape(pgl.Polyline(dart_points[dart],width=2),coal)
 
        for dart in self.darts():
            alpha_0_points = []
            alpha_0_points += [dart_points[dart][1]]
            alpha_0_points += [dart_points[self.alpha(0,dart)][1]]
            s += pgl.Shape(pgl.Polyline(alpha_0_points,width=5),purple)
 
            alpha_1_points = []
            alpha_1_points += [0.66*dart_points[dart][0] + 0.33*dart_points[dart][1]]
            alpha_1_points += [0.66*dart_points[self.alpha(1,dart)][0] + 0.33*dart_points[self.alpha(1,dart)][1]]
            s += pgl.Shape(pgl.Polyline(alpha_1_points,width=5),green)
 
            alpha_2_points = []
            alpha_2_points += [0.33*dart_points[dart][0] + 0.66*dart_points[dart][1]]
            alpha_2_points += [0.33*dart_points[self.alpha(2,dart)][0] + 0.66*dart_points[self.alpha(2,dart)][1]]
            s += pgl.Shape(pgl.Polyline(alpha_2_points,width=5),blue)
 
        if add : 
            pgl.Viewer.add(s)
        else : 
            pgl.Viewer.display(s)

    def adjacent_cells(self, dart, degree):
        """ 
        Return all the elements of degree degree
        that are adjacent to the element dart with respect
        to the alpha relation of degree degree.
        (Typically all points sharing an edge with a point)
        For this iterate over all the darts of the orbit of (dart, degree).
        For each dart d of this orbit, get its neighbor n (alpha degree)
        and remove its orbit (n, degree) from the set of darts
        to consider.
        See function incident_cells for inspiration.
        """
        list_of_alpha_value = range(3)
        list_of_alpha_value.remove(degree)
        neighborhood = self.orbit(dart,list_of_alpha_value)
        marked = []
        adjacentDarts = []
        for i in range(len(neighborhood)):
            neighborhood[i] = self.alphas[degree][neighborhood[i]]
            alreadyFound = False
            for j in self.orbit(neighborhood[i],list_of_alpha_value):
                if j in marked:
                    alreadyFound = True
                else:
                    marked.append(j)
            if not(alreadyFound):
                adjacentDarts.append(neighborhood[i])
        return adjacentDarts      

def gmap_laplacian_smoothing(gmap, coef=0.5):
    """ Compute the new position of all elements of degree 0 in the
    GMap by moving them towards the isobarycenter of their neighbors :
    pos(i)* <- pos(i) + coef * sum_{j in N(i)} (1/valence(i))*(pos(j) - pos(i))
    """
 
    # Compute a dictionary of laplacian vectors of the vertices
    # Iterate over all the vertices
        # Compute an array of all the vectors to the neighbors of a vertex
        # Compute the sum of all those vectors normalized by the valence of the vertex
 
    # Iterate over all the vertices
        # Compute the new position of the vertex using the coef and update it
    vertices = []
    for i in gmap.darts():
        dart = gmap.get_embedding_dart(i,gmap.positions)
        if not(dart in vertices):
            vertices.append(dart)
            verticesNeightboor = gmap.adjacent_cells(dart,0)
            verticesNeightboorPosition = []
            for j in verticesNeightboor:
                verticesNeightboorPosition.append(gmap.get_position(j))
            isoBarycentre = np.mean(verticesNeightboorPosition)    
            vecDiff = isoBarycentre - gmap.get_position(dart)
            gmap.set_position(dart,gmap.get_position(dart) + vecDiff * coef)

import math

def gmap_gaussian_smoothing(gmap, coef=0.5, gaussian_sigma=None):
    """
    Compute the new position of all elements of degree 0 in the
    GMap by moving them towards the a weighted barycenter of their neighbors 
    where the weights are a gaussian function of the edge lengths:
    pos(i)* <- pos(i) + coef * sum_{j in N(i)} (Gij/sum_k Gik)*(pos(j) - pos(i))
    Gij = e^{-lij^2 / sigma^2}
    """
    vertices = []
    for i in gmap.darts():
        dart = gmap.get_embedding_dart(i,gmap.positions)
        if not(dart in vertices):
            vertices.append(dart)
            verticesNeightboor = gmap.adjacent_cells(dart,0)
            verticesNeightboorPosition = []
            edgeLength = []
            for j in verticesNeightboor:
                edgeLen = (np.linalg.norm(gmap.get_position(j) - gmap.get_position(dart)))
                edgeLen = np.exp(np.power(edgeLen,2)/(2*math.pi*np.power(gaussian_sigma,2)))
                edgeLength.append(edgeLen)
                verticesNeightboorPosition.append(gmap.get_position(j))

            isoBarycentre = np.average(verticesNeightboorPosition,axis=0,weights=edgeLength)    
            vecDiff = isoBarycentre - gmap.get_position(dart)
            gmap.set_position(dart,gmap.get_position(dart) + vecDiff * coef)

def gmap_taubin_smoothing(gmap, coef_pos=0.33, coef_neg=0.34, gaussian_sigma=None):
    """
    Compute the new position of all elements of degree 0 in the
    GMap by applying two opposite Gaussian smoothing iterations
    """
    gmap_gaussian_smoothing(gmap, coef=coef_pos, gaussian_sigma=gaussian_sigma)
    gmap_gaussian_smoothing(gmap, coef=-coef_neg, gaussian_sigma=gaussian_sigma)

def add_square(gmap):
    darts = [gmap.add_dart() for i in xrange(8)]
    for i in xrange(4):
        gmap.link_darts(0, darts[2*i], darts[2*i+1])
    for i in xrange(4):
        gmap.link_darts(1, darts[2*i+1], darts[(2*i+2) % 8])
    return darts
 
def cube(xsize = 5, ysize  = 5 , zsize = 5):
    g = GMap()
    squares = [add_square(g) for i in xrange(6)]
 
    # sew top square to lateral squares
    g.sew_dart(2, squares[0][0], squares[1][1] )
    g.sew_dart(2, squares[0][2], squares[4][1] )
    g.sew_dart(2, squares[0][4], squares[3][1] )
    g.sew_dart(2, squares[0][6], squares[2][1] )
 
    # sew bottom square to lateral squares
    g.sew_dart(2, squares[5][0], squares[1][5] )
    g.sew_dart(2, squares[5][2], squares[2][5] )
    g.sew_dart(2, squares[5][4], squares[3][5] )
    g.sew_dart(2, squares[5][6], squares[4][5] )
 
    # sew lateral squares between each other
    g.sew_dart(2, squares[1][2], squares[2][7] )
    g.sew_dart(2, squares[2][2], squares[3][7] )
    g.sew_dart(2, squares[3][2], squares[4][7] )
    g.sew_dart(2, squares[4][2], squares[1][7] )
 
    for darti, position in zip([0,2,4,6],[ [xsize, ysize, zsize], [xsize, -ysize, zsize] , [-xsize, -ysize, zsize], [-xsize, ysize, zsize]]):
        dart = squares[0][darti]
        g.set_position(dart, position)
     
    for darti, position in zip([0,2,4,6],[ [xsize, -ysize, -zsize], [xsize, ysize, -zsize] , [-xsize, +ysize, -zsize], [-xsize, -ysize, -zsize]]):
        dart = squares[5][darti]
        g.set_position(dart, position)
 
    return g
 

def holeshape(xsize = 5, ysize = 5, zsize = 5, internalratio = 0.5):
    assert 0 < internalratio < 1
 
    g = GMap()
    squares = [add_square(g) for i in xrange(16)]
 
    # sew upper squares between each other
    g.sew_dart(2, squares[0][2], squares[1][1] )
    g.sew_dart(2, squares[1][4], squares[2][3] )
    g.sew_dart(2, squares[2][6], squares[3][5] )
    g.sew_dart(2, squares[3][0], squares[0][7] )
 
    # sew upper squares with external lateral
    g.sew_dart(2, squares[0][0], squares[8][1] )
    g.sew_dart(2, squares[1][2], squares[9][1] )
    g.sew_dart(2, squares[2][4], squares[10][1] )
    g.sew_dart(2, squares[3][6], squares[11][1] )
 
    # # sew upper squares with internal lateral
    g.sew_dart(2, squares[0][5], squares[12][0] )
    g.sew_dart(2, squares[1][7], squares[13][0] )
    g.sew_dart(2, squares[2][1], squares[14][0] )
    g.sew_dart(2, squares[3][3], squares[15][0] )
 
    # sew lower squares between each other
    g.sew_dart(2, squares[4][6], squares[5][1] )
    g.sew_dart(2, squares[5][4], squares[6][7] )
    g.sew_dart(2, squares[6][2], squares[7][5] )
    g.sew_dart(2, squares[7][0], squares[4][3] )
 
    # sew lower squares with external lateral
    g.sew_dart(2, squares[4][0], squares[8][5] )
    g.sew_dart(2, squares[5][6], squares[9][5] )
    g.sew_dart(2, squares[6][4], squares[10][5] )
    g.sew_dart(2, squares[7][2], squares[11][5] )
 
    # sew lower squares with internal lateral
    g.sew_dart(2, squares[4][5], squares[12][4] )
    g.sew_dart(2, squares[5][3], squares[13][4] )
    g.sew_dart(2, squares[6][1], squares[14][4] )
    g.sew_dart(2, squares[7][7], squares[15][4] )
 
    # sew external lateral squares between each other
    g.sew_dart(2, squares[8][7], squares[9][2] )
    g.sew_dart(2, squares[9][7], squares[10][2] )
    g.sew_dart(2, squares[10][7], squares[11][2] )
    g.sew_dart(2, squares[11][7], squares[8][2] )
 
    # sew internal lateral squares between each other
    g.sew_dart(2, squares[12][2], squares[13][7] )
    g.sew_dart(2, squares[13][2], squares[14][7] )
    g.sew_dart(2, squares[14][2], squares[15][7] )
    g.sew_dart(2, squares[15][2], squares[12][7] )
 
    pos = { 
            (0,0) : [xsize,  ysize,  zsize] ,
            (1,2) : [xsize,  -ysize, zsize] ,
            (2,4) : [-xsize, -ysize, zsize] ,
            (3,6) : [-xsize, ysize,  zsize] ,
 
            (0,5) : [xsize*internalratio,  ysize*internalratio,  zsize] ,
            (1,7) : [xsize*internalratio,  -ysize*internalratio, zsize] ,
            (2,1) : [-xsize*internalratio, -ysize*internalratio, zsize] ,
            (3,3) : [-xsize*internalratio, ysize*internalratio,  zsize] ,
 
            (4,1) : [xsize,  ysize,  -zsize] ,
            (5,7) : [xsize,  -ysize, -zsize] ,
            (6,5) : [-xsize, -ysize, -zsize] ,
            (7,3) : [-xsize, ysize,  -zsize] ,
 
            (4,4) : [xsize*internalratio,  ysize*internalratio,  -zsize] ,
            (5,2) : [xsize*internalratio,  -ysize*internalratio, -zsize] ,
            (6,0) : [-xsize*internalratio, -ysize*internalratio, -zsize] ,
            (7,6) : [-xsize*internalratio, ysize*internalratio,  -zsize] ,
          }
 
    for darti, position in pos.items():
        sqid, dartid = darti
        dart = squares[sqid][dartid]
        g.set_position(dart, position)
 
    return g
def square():
    gmap = GMap()
    add_square(gmap)
    return gmap

def triangular_gmap_split_edge(gmap, dart):
    """
    Perform a topological split operation on a triangular GMap
    by adding a new vertex in the middle of the edge and linking
    it to the opposite vertices of the adjacent triangles.
    """
    # Compute the position of the edge center
    # Split the edge and get the new vertex dart
    # Update the position of the new vertex to the edge center
    # Split the face(s) incident to the new vertex
    d0 = dart
    d1 = gmap.alpha_composed([0],dart)
    d2 = gmap.alpha_composed([0,2],dart)
    d3 = gmap.alpha_composed([2],dart)
    addedDarts = []
    for _ in range(4):
        addedDarts.append(gmap.add_dart())
    #Liaison des darts sur les edges
    gmap.link_darts(2,addedDarts[0], addedDarts[1])
    gmap.link_darts(2,addedDarts[2], addedDarts[3])

    #liaison vers la suite de la edge splitée
    gmap.link_darts(1,addedDarts[0], addedDarts[2])
    gmap.link_darts(1,addedDarts[1], addedDarts[3])

    gmap.set_position(addedDarts[0],(gmap.get_position(d0)-gmap.get_position(d2))/2)

    #liaison vers les anciennes darts
    gmap.force_link_darts(0,addedDarts[0], d0)
    gmap.force_link_darts(0,addedDarts[1], d1)
    gmap.force_link_darts(0,addedDarts[2], d2)
    gmap.force_link_darts(0,addedDarts[3], d3)

    ### Split du triangle >> creation de nouvelle arrete
    gmap.split_face(addedDarts[0])
    gmap.split_face(addedDarts[2])

import scipy.ndimage as nd
from scipy.cluster.vq import vq

from gmap_tools import read_ply_mesh, array_unique
from basicshapes import topopolygon
from array_dict import array_dict

def gmap_from_triangular_mesh(points, triangles, center=False):
    gmap = GMap()

    triangle_edges = np.sort(np.concatenate([np.transpose([v,list(v[1:])+[v[0]]]) for v in triangles]))

    edges = array_unique(triangle_edges)
    triangle_edges = (vq(triangle_edges,edges)[0]).reshape((len(triangles),3))

    triangle_edge_vertices = np.concatenate([np.transpose([v,list(v[1:])+[v[0]]]) for v in triangles])
    triangle_edge_orientation = (triangle_edge_vertices[:,1]>triangle_edge_vertices[:,0]).reshape((len(triangles),3))

    triangle_darts = {}
    for fid,t in enumerate(triangles):
        gmap, tri = topopolygon(3,gmap)
        triangle_darts[fid] = tri

    for eid,e in enumerate(edges):
        fids_to_sew, darts_to_sew = np.where(triangle_edges==eid)
        orientations_to_sew = triangle_edge_orientation[(fids_to_sew, darts_to_sew)]
        if len(orientations_to_sew) > 1:
            gmap.sew_dart(2,triangle_darts[fids_to_sew[0]][2*darts_to_sew[0]+(1-orientations_to_sew[0])],triangle_darts[fids_to_sew[1]][2*darts_to_sew[1]+(1-orientations_to_sew[1])])

    mesh_center = np.mean(points,axis=0) if center else np.zeros(3)
    for fid,t in enumerate(triangles):
        for i,p in enumerate(t):
            gmap.set_position(triangle_darts[fid][2*i], points[p]-mesh_center)

    return gmap

def gmap_edge_split_optimization(gmap, maximal_length=1.0):
    """
    Perform one iteration of edge split optimization:
    Rank the GMap edges by length and iterativelty split 
    those whose length exceeds maximal_length
    """

    vertex_positions = array_dict([gmap.get_position(v) for v in gmap.darts()],gmap.darts())
    vertex_valence = array_dict(np.array(map(len,[gmap.orbit(v,[1,2]) for v in gmap.darts()]))/2,gmap.darts())
    edge_vertices = np.array([(e,gmap.alpha(0,e)) for e in gmap.elements(1)])
    edge_lengths = array_dict(np.linalg.norm(vertex_positions.values(edge_vertices[:,1]) - vertex_positions.values(edge_vertices[:,0]),axis=1),keys=gmap.elements(1))

    sorted_edge_length_edges = np.array(gmap.elements(1))[np.argsort(-edge_lengths.values(gmap.elements(1)))]
    sorted_edge_length_edges = sorted_edge_length_edges[edge_lengths.values(sorted_edge_length_edges)>maximal_length]
    
    n_splits = 0
    print "--> Splitting edges"
    for e in sorted_edge_length_edges:
        triangular_gmap_split_edge(gmap,e)
        n_splits += 1
    print "<-- Splitting edges (",n_splits," edges split)"

    return n_splits

def triangular_gmap_flip_edge(gmap, dart):
    """
    Perform a topological flip operation on a triangular GMap
    by modifying the alpha_1 relationships of the darts impacted :
    6 (reciprocal) relationships to update + make sure that 
    position dictionary is not impacted
    """

    # Compute a dictionary of the new alpha_1 relationships of :
    # dart, alpha_0(dart), alpha_2(dart), alpha_0(alpha_2(dart)),
    # alpha_1(dart), alpha_1(alpha_0(dart)) 

    # Make sure that no dart in the orbit 1 of dart is a key of
    # the positions dictionary, otherwise transfer the position
    # to another embedding dart

    # Assert that the new alpha_1 is still without fixed points

    # Set the alphas of the GMap to their new values 
    # (not forgetting the reciprocal alpha_1)

    # Return the list of all darts  whose valence will be
    # impacted by the topological change
    d0 = dart
    d1 = gmap.alpha_composed([0],dart)
    d2 = gmap.alpha_composed([0,2],dart)
    d3 = gmap.alpha_composed([2],dart)

    p0 = gmap.get_embedding_dart(d0,gmap.positions)
    p1 = gmap.get_embedding_dart(d1,gmap.positions)
    
    if (p0 in [d0,d3]):
        pass
    if (p1 in [1,2]):
        pass

    gmap.force_link_darts(1,gmap.alpha_composed([1],d0),gmap.alpha_composed([1],d3))
    gmap.force_link_darts(1,gmap.alpha_composed([1],d2),gmap.alpha_composed([1],d1))

    gmap.force_link_darts(1,gmap.alpha_composed([1,0],d0),d0)
    gmap.force_link_darts(1,gmap.alpha_composed([1,0],d1),d1)
    gmap.force_link_darts(1,gmap.alpha_composed([1,0],d2),d2)
    gmap.force_link_darts(1,gmap.alpha_composed([1,0],d3),d3)
    return gmap.alphas


   


def gmap_edge_flip_optimization(gmap, target_neighborhood=6):
    """
    Perform one iteration of edge flip optimization:
    Identify the GMap edges that can be flipped and 
    compute the neighborhood error variation induced by
    their flip. Rank them along this variation and 
    perform allowed edge flips for edges with a negative
    variation.
    """

    vertex_positions = array_dict([gmap.get_position(v) for v in gmap.darts()],gmap.darts())
    vertex_valence = array_dict(np.array(map(len,[gmap.orbit(v,[1,2]) for v in gmap.darts()]))/2,gmap.darts())

    edge_vertices = np.array([(e,gmap.alpha(0,e)) for e in gmap.elements(1)])
    edge_lengths = array_dict(np.linalg.norm(vertex_positions.values(edge_vertices[:,1]) - vertex_positions.values(edge_vertices[:,0]),axis=1),keys=gmap.elements(1))
    edge_flipped_vertices = np.array([[gmap.alpha(0,gmap.alpha(1,e)),gmap.alpha(0,gmap.alpha(1,gmap.alpha(2,e)))] for e in gmap.elements(1)])

    flippable_edges = np.array(gmap.elements(1))[edge_flipped_vertices[:,0] != edge_flipped_vertices[:,1]]
    
    flippable_edge_vertices = edge_vertices[edge_flipped_vertices[:,0] != edge_flipped_vertices[:,1]]
    flippable_edge_flipped_vertices = np.array([ e for e in edge_flipped_vertices[edge_flipped_vertices[:,0] != edge_flipped_vertices[:,1]]])

    flippable_edge_triangle_vertices = np.array([[np.concatenate([e,[v]]) for v in f] for (e,f) in zip(flippable_edge_vertices,flippable_edge_flipped_vertices)])
    flippable_edge_flipped_triangle_vertices = np.array([[np.concatenate([f,[v]]) for v in e] for (e,f) in zip(flippable_edge_vertices,flippable_edge_flipped_vertices)])

    from gmap_tools import triangle_geometric_features
    flippable_edge_triangle_areas = np.concatenate([triangle_geometric_features(flippable_edge_triangle_vertices[:,e],vertex_positions,features=['area']) for e in [0,1]],axis=1)
    flippable_edge_flipped_triangle_areas = np.concatenate([triangle_geometric_features(flippable_edge_flipped_triangle_vertices[:,e],vertex_positions,features=['area']) for e in [0,1]],axis=1)
           
    average_area = np.nanmean(flippable_edge_triangle_areas)
    flippable_edge_flipped_triangle_areas[np.isnan(flippable_edge_flipped_triangle_areas)] = 100.
    wrong_edges = np.where(np.abs(flippable_edge_triangle_areas.sum(axis=1)-flippable_edge_flipped_triangle_areas.sum(axis=1)) > average_area/10.)

    flippable_edges = np.delete(flippable_edges,wrong_edges,0)
    flippable_edge_vertices = np.delete(flippable_edge_vertices,wrong_edges,0)
    flippable_edge_triangle_vertices = np.delete(flippable_edge_triangle_vertices,wrong_edges,0)
    flippable_edge_flipped_vertices = np.delete(flippable_edge_flipped_vertices,wrong_edges,0)
    flippable_edge_flipped_triangle_vertices = np.delete(flippable_edge_flipped_triangle_vertices,wrong_edges,0)
    flippable_edge_triangle_areas = np.delete(flippable_edge_triangle_areas,wrong_edges,0)
    flippable_edge_flipped_triangle_areas =  np.delete(flippable_edge_flipped_triangle_areas,wrong_edges,0)
                
    flippable_edge_neighborhood_error = np.power(vertex_valence.values(flippable_edge_vertices)-target_neighborhood,2.0).sum(axis=1)
    flippable_edge_neighborhood_error += np.power(vertex_valence.values(flippable_edge_flipped_vertices)-target_neighborhood,2.0).sum(axis=1)
    flippable_edge_neighborhood_flipped_error = np.power(vertex_valence.values(flippable_edge_vertices)-1-target_neighborhood,2.0).sum(axis=1)
    flippable_edge_neighborhood_flipped_error += np.power(vertex_valence.values(flippable_edge_flipped_vertices)+1-target_neighborhood,2.0).sum(axis=1)

    n_flips = 0 
    if len(flippable_edges)>0:

        flippable_edge_energy_variation = array_dict(flippable_edge_neighborhood_flipped_error-flippable_edge_neighborhood_error,flippable_edges)

        flippable_edge_sorted_energy_variation_edges = flippable_edges[np.argsort(flippable_edge_energy_variation.values(flippable_edges))]
        flippable_edge_sorted_energy_variation_edges = flippable_edge_sorted_energy_variation_edges[flippable_edge_energy_variation.values(flippable_edge_sorted_energy_variation_edges)<0] 
        modified_darts = set()
        print "--> Flipping edges"

        for e in flippable_edge_sorted_energy_variation_edges:

            flippable_edge = (len(modified_darts.intersection(set(gmap.orbit(e,[1,2])))) == 0)
            flippable_edge = flippable_edge and (len(modified_darts.intersection(set(gmap.orbit(gmap.alpha(0,e),[1,2])))) == 0)
            flippable_edge = flippable_edge and (len(modified_darts.intersection(set(gmap.orbit(gmap.alpha(0,gmap.alpha(1,e)),[1,2])))) == 0)
            flippable_edge = flippable_edge and (len(modified_darts.intersection(set(gmap.orbit(gmap.alpha(0,gmap.alpha(1,gmap.alpha(2,e))),[1,2])))) == 0)

            if flippable_edge:
                n_e = len(gmap.elements(1))
                mod = triangular_gmap_flip_edge(gmap,e)
                modified_darts = modified_darts.union(set(mod))
                n_flips += 1
        print "<-- Flipping edges (",n_flips," edges flipped)"

    return n_flips


def get_edgepoint(edgepoints, gmap, edart):
    	return gmap.get_embedding_dart(edart, edgepoints,1)

def get_facepoint(facepoints, gmap, fdart):
    	return gmap.get_embedding_dart(fdart, facepoints,2)

def catmullclark(gmap):
    # Compute a dictionary of face points equal to the face center
    # (use functions elements and element_center)
    facepoints = dict([   ])
    face_list = gmap.elements(2) #get 1 dart per face
    print("expected facepoints added number: " + str(len(face_list)))
    print("expected edge added number:" + str(len(gmap.elements(1))))
    for dart in face_list:
    	position = gmap.element_center(dart,2)
    	"""
    	#Il faut créer autant de darts qu'il y en a sur l'orbit de la face
    	face_orbit = gmap.orbit(dart, [0,1])
    	face_point_list=[]
    	for i in len(face_orbit):
    		face_point.append(gmap.add_dart())
    		if(i>=1):
    			if i%2==0:
    				gmap.link_darts(face_point_list[i],face_point_list[i+1],2)
    			else:
    				gmap.link_darts(face_point_list[i],face_point_list[i+1],1)

    	gmap.set_position(face_point_list[0],position)"""
    	facepoints[dart]=position
    # Create a local function to access to the face point from any dart of the face
    # (use function get_embedding_dart with the right degree)
    

    # Define a function to compute the position of an edge points:
    # For this, list the positions of the edge vertices (use incident_cells and get_position)
    # and the face points of the edge faces (use incident_cells and get_facepoint) and
    # return the mean
    #def compute_edgepoint(edart):

    # Compute a dictionary of edge points just as for facepoints
	edgepoints = dict([   ])
	edge_list = gmap.elements(1)
	for dart in edge_list:
		positions = []
		positions.append(gmap.get_position(dart))
		positions.append(gmap.get_position(gmap.alpha(0,dart)))
		positions.append(gmap.get_position(get_facepoint(facepoints,gmap, dart)))
		positions.append(gmap.get_position(get_facepoint(facepoints,gmap,gmap.alpha(2,dart))))
		edgepoints[dart] = gmap.element_center(dart,1)


    # Create a local function to access the edgepoint from any dart of the edge
    


    # Define a function to compute the new position of a vertex:
    # For this, compute the mean of edgepoints of incident edges (E),
    # the mean of facepoints of incident faces (F) and the current
    # position of the vertex (V). (use incident_cells, get_position,
    # get_edgepoint and get_facepoint)
    # Use the valence of the vertex (k, number of incident edges) to 
    # compute the new position:
    # V* &lt;- ((k-3)V + 2E + F)/3
    #def compute_vertexpoint(vdart):
    vertexpoint = dict([  ])
    vertex_list = gmap.elements(0)
    print("expected vertex number:" + str(len(face_list)))
    for dart in vertex_list:
    	done_list=[]
    	incidence=0
    	#Faire la moyenne des edge points
    	Eposition= []
    	while not(dart in done_list):
    		incidence = incidence+1
    		Eposition.append(edgepoints[get_edgepoint(edgepoints,gmap,dart)])
    		done_list.append(dart)
    		dart = gmap.alpha_composed(dart=dart,list_of_alpha_value=[1,2])

    	E = np.mean(Eposition, axis=0)
    	done_list=[]
    	#faire la moyenne des face points
    	Fposition= []
        weigh=0
    	while not(dart in done_list):

    		Fposition.append(facepoints[get_facepoint(facepoints,gmap,dart)])
    		done_list.append(dart)
    		dart = gmap.alpha_composed(dart=dart,list_of_alpha_value=[1,2])

    	F = np.mean(Fposition, axis=0)

    	#Appliquer la formule
        #print("1/incidence * (F + 2E + dartPos * incidence-3)")
        tmp_pos = 1./incidence * (F + 2.*E + gmap.get_position(dart)*(incidence-3.))
        #print("1/ "+ str(incidence)+"*(" + str(F) +" + 2*" + str(E) +" + "+ str(gmap.get_position(dart)) + "* (" + str(incidence) + "-3)) = " + str(tmp_pos))
    	vertexpoint[dart]= tmp_pos

    # Set the new position to the vertex points
    for dart in vertex_list:
    	gmap.set_position(dart, vertexpoint[dart])
    # Create new vertices in the topological structure corresponding
    # to edge points:
    # For this, go through all the edges and split them
    # (use split_edge) and set the position to the new vertex.
    # Doing so, fill a list with all the new inserted darts 
    # (vertex orbits of the new darts created by the split)
    edgepoint_darts = []
    edge_list = gmap.elements(1)
    for dart in edge_list:
    	current_edart = gmap.split_edge(dart)
    	gmap.set_position(current_edart, edgepoints[dart])
    	edgepoint_darts =  edgepoint_darts + gmap.orbit(current_edart,[1,2])


    # Finally, create the new vertices in the topological structure 
    # corresponding to face points, and connect them to the edge point 
    # vertices inserted previously:
    # Iterate over all the faces
    for dart in face_list:
        # Store the postion of the face point corresponding to the face
    	position = facepoints[ get_facepoint(facepoints,gmap,dart)]
        #print("accessed face position :" + str(dart))
        # Iterate over all the incident vertices of the face
        incident_darts= gmap.orbit(dart,[0,1])
        to_process= []
        if len(incident_darts) == 0:
        	print("ITS FUCKING APOCALYPSE THERE IS AN EMPTY RJOI%PEQHS")
        
        for darti in incident_darts:
            # Check if the vertex corresponds to an edge point (inserted 
            # at the previous step)
            if darti in edgepoint_darts:
                # If it is an edge point store it in a list to process
                to_process.append(darti)
            else:
            	#print ("----  " + str(len(gmap.orbit(darti,[1,2]))) + "    I iz: " + str(i))
            	pass
        # Iterate over all the edge points to process
        #print("CC : after ####### : Len of list of element of degree 0 :" + str(len(gmap.elements(0))))
        elm0 = len(gmap.elements(0))
        ptslinked=0
        to_face_process = []
        if len(to_process) == 0:
        	print("ITS FUCKING APOCALYPSE THERE IS AN EMPTY RJOI%PEQHS")

        marked = []
        for darti in to_process:
            # Create a new edge from the vertex (use insert_edge)
            marked.append(gmap.alpha(1,darti))
            if not(darti in marked):
                current_edge=gmap.insert_edge(darti)
                # Store the darts at the other end of the edge (vertex orbits
                # of the new darts created by the insertion) in a list to process
                to_face_process.append(current_edge)
                to_face_process.append(gmap.alpha(2,current_edge))
        if len(to_face_process) == 0:
        	print("ITS FUCKING APOCALYPSE THERE IS AN EMPTY RJOI%PEQHS")

        for darta in to_face_process:
            #print("begin" + str(gmap.orbit(darta,[1,2])))
            pass
        #print("Len of positions dict in gmap:" + str(len(gmap.positions.keys())))
        for darta in to_face_process:
        # Iterate over all the end darts to process
            # If the dart d is free by alpha_1:
        	if gmap.is_free(1,darta):
                #Find its next dart in the new (quad) face :
                # alpha_0(alpha_1(alpha_0(alpha_1(alpha_0(alpha_1(alpha_0(d)))))))"""
				next_dart = gmap.alpha_composed([0,1,0,1,0,1,0],darta)
                if gmap.is_free(1,darta):
                    gmap.link_darts(1,darta,next_dart)
                    ptslinked = ptslinked + 1
                    gmap.set_position(darta,position)

        #print("Face point adding delta elm0: " + str(len(gmap.elements(0))-elm0))
        #print("Len of positions dict in gmap:" + str(len(gmap.positions.keys())))
        # Select one of the end darts and set is position to the face point 
    print("facepoints : " + str(len(facepoints.keys())))
    #print(facepoints)
    print("edgepoints : " + str(len(edgepoints.keys())))
    #print(edgepoints)
    print("vertexpoints : " + str(len(vertexpoint.keys())))
    #print(vertexpoint)



filename = 'cow.ply'#spider,bunny,cow
#points, triangles = read_ply_mesh(filename)
#gmap = gmap_from_triangular_mesh(points, triangles, center=True)

le_map = holeshape() # holeshape(), square() , cube()
if le_map.is_valid():
    print (" all is 'k the map is valid")
else:
    print ("Fuck me it's all wrong")
    
#le_map.print_alphas()
#le_map.dart_display()

def gmap_add_uniform_noise(gmap, coef=0.01):
    characteristic_distance = np.linalg.norm(np.std([gmap.get_position(v) for v in gmap.elements(0)],axis=0))
    for v in gmap.elements(0):
        gmap.set_position(v,gmap.get_position(v) + (1.-2.*np.random.rand(3))*coef*characteristic_distance)


#gmap_add_uniform_noise(gmap, coef=0.05)

"""for _ in range(3):
    gmap_taubin_smoothing(gmap,0.5,0.5,0.33)"""
#gmap_edge_flip_optimization(gmap)
gmap = cube()
print("PRE CatmullClark: Len of list of element of degree (0,1,2) : (" + str(len(gmap.elements(0))) + ", " + str(len(gmap.elements(1)))+ ", " + str(len(gmap.elements(2)))+ ")")
print("PRE CatmullClark: Len of positions dict in gmap:" + str(len(gmap.positions.keys())))
catmullclark(gmap)
catmullclark(gmap)
catmullclark(gmap)
print("Len of list of element of degree (0,1,2) : (" + str(len(gmap.elements(0))) + ", " + str(len(gmap.elements(1)))+ ", " + str(len(gmap.elements(2)))+ ")")
print("Len of positions dict in gmap:" + str(len(gmap.positions.keys())))
#print(gmap.positions)
gmap.display()