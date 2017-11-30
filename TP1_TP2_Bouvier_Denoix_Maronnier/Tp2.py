

class GMap:
    def __init__(self):
        """ 
        Constructor of GMap of degree=2
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
        current = dart
        for i in list_of_alpha_value:
          current = self.alphas[i][current]
        return current

    def is_free(self, degree, dart):
        """ 
        Test if dart is free for alpha_degree (if it is a fixed point) 
        """
        return self.alphas[degree][dart] == dart

    def add_dart(self):
        """ 
        Create a new dart and return its id. 
        Set its alpha_i to itself (fixed points) 
        """
        dart_id = self.maxid
        for i in self.alphas:
          self.alphas[i][dart_id] = dart_id
        self.maxid = dart_id +1
        return dart_id

    def is_involution(self, dart, list_of_alpha):
		current_dart = dart
		for _ in range(2):
			for i in list_of_alpha:
				current_dart = self.alphas[i][current_dart]
		return dart == current_dart
		
    def is_valid(self):
        """ 
        Test the validity of the structure. 
        Check that alpha_0 and alpha_1 are involutions with no fixed points.
		Check that any points, ( a1 o a2) is involution
		 >> get 0 >> get 2 >> get 0 >> get 2
        """
        for i in [0,1]:
          for dart in range(self.maxid):
			return (not(self.is_involution(dart,[0])) and self.is_involution(dart,[2,0]))
        return True

	
    def link_darts(self,degree, dart1, dart2): 
        """ 
        Link the two darts with a relation alpha_degree if they are both free.
        """
        if (self.is_free(degree=degree,dart=dart1) and self.is_free(degree=degree,dart=dart2)):
          self.alphas[degree][dart1]=dart2
          self.alphas[degree][dart2]=dart1
        

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
            print "d     ?0  ?1  ?2"
            for d in self.darts():
                print d," | ",Fore.MAGENTA+str(self.alpha(0,d))," ",Fore.GREEN+str(self.alpha(1,d))," ",Fore.BLUE+str(self.alpha(2,d))," ",Style.RESET_ALL 
        

    def orbit(self, dart, list_of_alpha_value):
        """ 
        Return the orbit of dart using a list of alpha relation.
        Example of use : gmap.orbit(0,[0,1]).
        """
        result = []
        toprocess = [dart]


        while len(toprocess) >0:
          d = toprocess.pop()
          if not( d in result):
            result.append(d)
            for i in list_of_alpha_value:
              toprocess.append(self.alphas[i][d])
        return result
        
        

    def elements(self, degree):
        """ 
        Return one dart per element of degree. For this, consider all darts as initial set S. 
        Take the first dart d, remove from the set all darts of the orbit starting from d and 
        corresponding to element of degree degree. Take then next element from set S and do the 
        same until S is empty. 
        Return all darts d that were used. 
        """
        
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
        
        
    def get_embedding_dart(self, dart, propertydict ):
        """ 
        Check if a dart of the orbit representing the vertex has already been 
        associated with a value in propertydict. If yes, return this dart, else
        return the dart passed as argument.
        """
        orbit = self.orbit(dart,[1,2])
        for d in orbit:
          if d in propertydict:
             return d
        return dart


    def get_position(self, dart):
        """
        Retrieve the coordinates associated to the vertex &lt;alpha_1, alpha_2&gt;(dart) 
        """
        return self.positions.get(self.get_embedding_dart(dart,self.positions))


    def set_position(self, dart, position) :
        """
        Associate coordinates with the vertex &lt;alpha_1,alpha_2&gt;(dart)
        """
        self.positions[self.get_embedding_dart(dart,self.positions)] = position        
        
    def sew_dart(self, degree, dart1, dart2, merge_attribute = True):
        """
        Sew two elements of degree 'degree' that start at dart1 and dart2.
        Determine first the orbits of dart to sew and check if they are compatible.
        Sew pairs of corresponding darts, and if they have different embedding 
        positions, merge them. 
        """
        if self.is_free(degree, dart1) and self.is_free(degree, dart2):
          if degree==2:
			# involution sur le degre:
            self.link_darts(2,dart1,dart2)
			
			# involution sur (a0 a2)
            a0_b1 = self.alphas[0][dart1]
            a0_b2 = self.alphas[0][dart2]
			
            self.link_darts(2,a0_b1, a0_b2)
            

          self.link_darts(degree,dart1, dart2)
	    #if merge_attribute:
		#  pass#self.set_position(dart, np.mean([get_position(dart1), get_position(dart2)]))
 
 
    def dart_display(self, radius=0.1, coef=0.8, add=False):
        import openalea.plantgl.all as pgl
 
        sphere = pgl.Sphere(radius,slices=16,stacks=16)
        coal = pgl.Material(ambient=(8,10,13),diffuse=3.,specular=(89,89,89),shininess=0.3)
        purple = pgl.Material(ambient=(72,28,72),diffuse=2.,specular=(89,89,89),shininess=0.3)
        green = pgl.Material(ambient=(0,88,9),diffuse=2.,specular=(89,89,89),shininess=0.3)
        blue = pgl.Material(ambient=(9,0,88),diffuse=2.,specular=(89,89,89),shininess=0.3)
 
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
 
    def display(self, color = (190,205,205), add = False):
      """
      Display the 2-cells of a 2-G-Map using the ordered orbit of its darts in PlantGL.
      For each face element, retrieve the position of its ordered face darts and add a FaceSet PlantGL object to the scene.
      Example : s += pgl.Shape(pgl.FaceSet( [[0,0,0],[1,0,0],[1,1,0],[0,1,0]], [[0,1,2,3]]) , pgl.Material((0,100,0))) # for a green square
      """ 
      s += pgl.Shape(pgl.FaceSet( [[0,0,0],[1,0,0],[1,1,0],[0,1,0]], [[0,1,2,3]]) , pgl.Material((0,100,0)))           
 
   
    def element_center(self, dart, degree):
        import numpy as np
        list_of_alpha_value = range(3)
        list_of_alpha_value.remove(degree)
        return np.mean([self.get_position(d) for d in self.orbit(dart,list_of_alpha_value)])       
    def element_center(self, dart, degree):
        import numpy as np
        list_of_alpha_value = range(3)
        list_of_alpha_value.remove(degree)
        return np.mean([self.get_position(d) for d in self.orbit(dart,list_of_alpha_value)])


    def dart_display(self, radius=0.1, coef=0.8, add=False):
        import openalea.plantgl.all as pgl

        sphere = pgl.Sphere(radius,slices=16,stacks=16)
        coal = pgl.Material(ambient=(8,10,13),diffuse=3.,specular=(89,89,89),shininess=0.3)
        purple = pgl.Material(ambient=(72,28,72),diffuse=2.,specular=(89,89,89),shininess=0.3)
        green = pgl.Material(ambient=(0,88,9),diffuse=2.,specular=(89,89,89),shininess=0.3)
        blue = pgl.Material(ambient=(9,0,88),diffuse=2.,specular=(89,89,89),shininess=0.3)

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

    def display(self, color = (190,205,205), add = False):
		"""
		Display the 2-cells of a 2-G-Map using the ordered orbit of its darts in PlantGL.
		For each face element, retrieve the position of its ordered face darts and add a FaceSet PlantGL object to the scene.
		Example : s += pgl.Shape(pgl.FaceSet( [[0,0,0],[1,0,0],[1,1,0],[0,1,0]], [[0,1,2,3]]) , pgl.Material((0,100,0))) # for a green square
		"""
		pass
	
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

 
    
le_map = square()
add_square(le_map)
le_map.sew_dart(2,0,8)
le_map.print_alphas()
print(le_map.orbit(1,[2]))

print(le_map.sew_dart(2,1,8))

le_map.print_alphas()
le_map = cube() # holeshape(), square() , cube()
if le_map.is_valid():
	print (" all is 'k")
else:
	print ("Fuck me it's all wrong")
	
le_map.print_alphas()
le_map.dart_display()

