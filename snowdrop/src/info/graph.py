"""
Python class that demonstrates the essential facts and functionalities of graphs.
This class represents a directed graph using adjacency list representation 
It finds biconnected components in a given undirected graph with complexity : O(V + E) 

Original version of the code can be found at:
    https://www.python-course.eu/graphs_python.php
    https://tutorialspoint.dev/data-structure/graph-data-structure/biconnected-components

Modified by A.Goumilevski
"""
from collections import defaultdict 

instances:int = 0

class Graph(object):
    """Graph class."""
    
    __graph_dict = {}; __weights = {}; __vertices = {}; __inv_vertices = {}
    __graph = []
    
    def __init__(self, graph_dict=dict(), m=None, weights=None):
        """Instantiate graph object.
        
        If no dictionary or None is given, an empty dictionary will be used.
        """
        global instances
        instances += 1
        
        self.__graph_dict = graph_dict
        
        for k in graph_dict:
            if bool(graph_dict[k]):
                if not weights is None:
                    w = weights[k]
                else:
                    w = 1
                for v in graph_dict[k]:
                    self.__addEdge(k,v,w)
            else:
                self.__addEdge(k,k,0) # Self loop
                
        # Nodes dictionary
        if not m is None:
            self.m = m
            self.im = dict((v,k) for k,v in m.items())
        
        # No. of vertices 
        self.V = len(m) if m else 0
          
        # default dictionary to store graph 
        self.graph = defaultdict(list) 
          
        # default dictionary to store components 
        self.components = defaultdict(list)  
          
        # time is used to find discovery times 
        self.Time = 0 
          
        # Count is number of biconnected components 
        self.count = 0          

    def vertices(self):
        """Return the vertices of a graph."""
        return list(self.__graph_dict.keys())

    def edges(self):
        """Return the edges of a graph."""
        return self.__generate_edges()

    def add_vertex(self, vertex):
        """Add graph vertex.
        
        If the vertex is not in self.__graph_dict, a key "vertex" with an empty
        list as a value is added to the dictionary. Otherwise nothing has to be done. 
        """
        if vertex not in self.__graph_dict:
            self.__graph_dict[vertex] = []
        else:
            pass
        
        n = len(self.__vertices)
        self.__vertices[vertex] = n
        self.__inv_vertices[n] = vertex

    def add_edge(self, edge, w=None):
        """Add an edge.
        
        It is assumed that edge is of type set, tuple or list; between two vertices can be multiple edges! 
        """
        edge = set(edge)
        vertex1 = edge.pop()
        if edge:
            # not a loop
            vertex2 = edge.pop()
        else:
            # a loop
            vertex2 = vertex1
        if vertex1 in self.__graph_dict:
            self.__graph_dict[vertex1].append(vertex2)
        else:
            self.__graph_dict[vertex1] = [vertex2]
            
        if not vertex1 in self.__vertices:
            i = len(self.__vertices)
            self.__vertices[vertex1] = i
            self.__inv_vertices[i] = vertex1
        else:
            i = self.__vertices[vertex1]
        if not vertex2 in self.__vertices:
            j = len(self.__vertices)
            self.__vertices[vertex2] = j
            self.__inv_vertices[j] = vertex2
        else:
            j = self.__vertices[vertex2]
        k = str([min(i,j),max(i,j)])
        weight = None
        if w is None:
            if k in self.__weights:
                weight = self.__weights[k]
            else:
                weight = 1
        else:
            weight = w
            
        self.__addEdge(vertex1,vertex2,weight)     

    def __addEdge(self,u,v,w=None): 
        """Add an edge that connects two vertices to a graph.

        Args:
        u : str.
            Vertex name.
        v : str.
            Vertex name.
        w : int, optional
            Edge weight. The default is None.

        Returns:
            None.
        """
        if not u in self.__vertices:
            i = len(self.__vertices)
            self.__vertices[u] = i
            self.__inv_vertices[i] = u
        else:
            i = self.__vertices[u]
        if not v in self.__vertices:
            j = len(self.__vertices)
            self.__vertices[v] = j
            self.__inv_vertices[j] = v
        else:
            j = self.__vertices[v]
            
        k = str([min(i,j),max(i,j)])
        weight = None
        if w is None:
            if k in self.__weights:
                weight = self.__weights[k]
            else:
                weight = 1
        else:
            weight = w
        
        self.__weights[k] = weight
        self.__graph.append([i,j,weight]) 
        
    def addEdge(self, x, y): 
        """Function to add an edge to graph.""" 
        u = self.m[x]
        v = self.m[y]
        self.graph[u].append(v)  
        self.graph[v].append(u) 
     
    def __generate_edges(self):
        """
        Generate the edges of the graph.
        
        Edges are represented as sets with one (a loop back to the vertex) 
        or two vertices.
        """
        edges = []
        for vertex in self.__graph_dict:
            for neighbour in self.__graph_dict[vertex]:
                if {neighbour, vertex} not in edges:
                    edges.append({vertex, neighbour})
        return edges
 
    def find_isolated_vertices(self):
        """Return a list of isolated vertices."""
        graph = self.__graph_dict
        isolated = []
        for vertex in graph:
            print(isolated, vertex)
            if not graph[vertex]:
                isolated += [vertex]
        return isolated

    def BCCUtil(self, u, parent, low, disc, st): 
        """A recursive function that finds and prints strongly connected components using DFS traversal 
        
        u --> The vertex to be visited next 
        disc[] --> Stores discovery times of visited vertices 
        low[] -- >> earliest visited vertex (the vertex with minimum 
                   discovery time) that can be reached from subtree 
                   rooted with current vertex 
        st -- >> To store visited edges
        """
  
        # Count of children in current node  
        children = 0
  
        # Initialize discovery time and low value 
        disc[u] = self.Time 
        low[u] = self.Time 
        self.Time += 1
  
        # Recur for all the vertices adjacent to this vertex 
        for v in self.graph[u]: 
            # If v is not visited yet, then make it a child of u 
            # in DFS tree and recur for it 
            if disc[v] == -1 : 
                parent[v] = u 
                children += 1
                st.append((u, v)) # store the edge in stack 
                self.BCCUtil(v, parent, low, disc, st) 
  
                # Check if the subtree rooted with v has a connection to one of the ancestors of u 
                # Case 1 -- per Strongly Connected Components Article 
                low[u] = min(low[u], low[v]) 
  
                # If u is an articulation point, pop all edges from stack till (u, v) 
                if parent[u] == -1 and children > 1 or parent[u] != -1 and low[v] >= disc[u]:
                    w = -1
                    lst = []
                    while w != (u, v): 
                        w = st.pop() 
                        lst.append(w)
                    #     print(w)
                    # print()  
                    self.components[self.count] = lst
                    self.count += 1 # increment count 
              
            elif v != parent[u] and low[u] > disc[v]: 
                '''Update low value of 'u' only of 'v' is still in stack 
                (i.e. it's a back edge, not cross edge). 
                Case 2  
                -- per Strongly Connected Components Article'''
  
                low[u] = min(low [u], disc[v]) 
      
                st.append((u, v)) 
        return 
  
    def BCC(self): 
        """Function to do DFS traversal. It uses recursive BCCUtil()."""
           
        # Initialize disc and low, and parent arrays 
        disc = [-1] * (self.V) 
        low = [-1] * (self.V) 
        parent = [-1] * (self.V) 
        st = [] 
         
        # Call the recursive helper function to find articulation points 
        # in DFS tree rooted with vertex 'i' 
        for i in range(self.V): 
            if disc[i] == -1: 
                self.BCCUtil(i, parent, low, disc, st) 
         
            # If stack is not empty, pop all edges from stack 
            if st: 
                lst = []
                while st: 
                    w = st.pop() 
                    lst.append(w)
                #     print (w) 
                # print()
                self.components[self.count] = lst
                self.count = self.count + 1
        return
  
    def find_path(self, start_vertex, end_vertex, path=[]):
        """Find a path from start_vertex to end_vertex in graph."""
        graph = self.__graph_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return path
        if start_vertex not in graph:
            return None
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_path = self.find_path(vertex,end_vertex,path)
                if extended_path: 
                    return extended_path
        return None
    

    def find_all_paths(self, start_vertex, end_vertex, path=[]):
        """Find all paths from start_vertex to end_vertex in graph."""
        graph = self.__graph_dict 
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return [path]
        if start_vertex not in graph:
            return []
        paths = []
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_paths = self.find_all_paths(vertex, end_vertex, path)
                for p in extended_paths: 
                    paths.append(p)
        return paths
    
    
    def find_shortest_path(self, start_vertex, end_vertex, path=[]):
        """Find shortest path from start_vertex to end_vertex."""
        graph = self.__graph_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return path
        if start_vertex not in graph:
            return []
        shortest = []
        for vertex in graph[start_vertex]:
            if vertex not in path:
                newpath = self.find_shortest_path(vertex, end_vertex, path)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest
    
    def find_distance(self, start_vertex, end_vertex):
        """Find number of edges from start_vertex to end_vertex."""
        if start_vertex == end_vertex:
            return 0
        else:
            shortest = self.find_shortest_path(start_vertex, end_vertex)
            dist = len(shortest)
            return dist-1
    
    def get_vertices(self):
        """
        Get graph vertices.

        Returns:
            vertices : list
                List of vertices.
        """
        vertices = self.__vertices.keys()
        return vertices
    
    def get_connected_vertices(self, start_vertex, vertices_encountered=None, gdict=None):
        """Determine if graph is connected."""
        if gdict is None:
            gdict = self.__graph_dict   
        if vertices_encountered is None:
            vertices_encountered=list()
        vertices_encountered.append(start_vertex)
        vertices = list(gdict.keys())
        #vertices = self.get_vertices()
        if bool(vertices):
            if start_vertex in gdict:
                for vertex in gdict[start_vertex]:
                    if vertex not in vertices_encountered:
                        self.get_connected_vertices(vertex, vertices_encountered)
        return vertices_encountered
    
    def get_components(self):
        """Find isolated components of a graph."""
        lst = self.__getComponents()
        n = len(lst)
        components = list()
        for i in range(n):
            c1 = set(lst[i])
            b = True
            for j in range(i+1,n):
                c2 = set(lst[j])
                if len(c1) < len(c2) and c1.issubset(c2):
                    b = False
                    break
            if b:
                components.append(lst[i])
        return components
               
    def __getComponents(self):
        """Find isolated vertices."""
        # gdict = self.__graph_dict        
        # vertices = set(gdict.keys())
        vertices = set(self.get_vertices())
        components = list()
        while bool(vertices):
            start_vertex = vertices.pop()
            comp = self.get_connected_vertices(start_vertex)
            components.append(comp)
            vertices -= set(comp)
            
        return components
            
    def is_connected(self, vertices_encountered = None,start_vertex=None):
        """Determine if graph is connected."""
        if vertices_encountered is None:
            vertices_encountered = set()
        gdict = self.__graph_dict        
        vertices = list(gdict.keys())
        #vertices = self.get_vertices()
        if not start_vertex:
            # chosse a vertex from graph as a starting point
            start_vertex = vertices[0]
        vertices_encountered.add(start_vertex)
        if len(vertices_encountered) != len(vertices):
            for vertex in gdict[start_vertex]:
                if vertex not in vertices_encountered:
                    if self.is_connected(vertices_encountered, vertex):
                        return True
        else:
            return True
        return False
    
    def vertex_degree(self, vertex):
        """Find vertex degree.
        
        The degree of a vertex is the number of edges connecting
        it, i.e. the number of adjacent vertices. Loops are counted 
        double, i.e. every occurence of vertex in the list 
        of adjacent vertices. 
        """ 
        adj_vertices =  self.__graph_dict[vertex]
        degree = len(adj_vertices) + adj_vertices.count(vertex)
        return degree

    def degree_sequence(self):
        """Calculate the degree sequence."""
        seq = []
        for vertex in self.__graph_dict:
            seq.append(self.vertex_degree(vertex))
        seq.sort(reverse=True)
        return tuple(seq)

    @staticmethod
    def is_degree_sequence(sequence):
        """Find degree sequence.
        
        Method returns True, if the "sequence" is a 
        degree sequence, i.e. a non-increasing sequence. 
        Otherwise returns False.
        """
        # check if the sequence sequence is non-increasing:
        return all( x>=y for x, y in zip(sequence, sequence[1:]))

    def delta(self):
        """Get the minimum degree of the vertices."""
        minv = 1e10
        for vertex in self.__graph_dict:
            vertex_degree = self.vertex_degree(vertex)
            if vertex_degree < minv:
                minv = vertex_degree
        return minv
        
    def Delta(self):
        """Get the maximum degree of the vertices."""
        maxv = 0
        for vertex in self.__graph_dict:
            vertex_degree = self.vertex_degree(vertex)
            if vertex_degree > maxv:
                maxv = vertex_degree
        return maxv

    def density(self):
        """Calculate the density of a graph."""
        g = self.__graph_dict
        V = len(g.keys())
        E = len(self.edges())
        return 2.0 * E / (V *(V - 1))

    def diameter(self):
        """Calculate the diameter of the graph."""
        v = self.vertices() 
        pairs = [ (v[i],v[j]) for i in range(len(v)) for j in range(i+1, len(v)-1)]
        smallest_paths = []
        for (s,e) in pairs:
            paths = self.find_all_paths(s,e)
            if len(paths) >  0:
                smallest = sorted(paths, key=len)[0]
                smallest_paths.append(smallest)

        smallest_paths.sort(key=len)

        # longest path is at the end of list, 
        # i.e. diameter corresponds to the length of this pathz
        diameter = len(smallest_paths[-1]) - 1
        return diameter

    @staticmethod
    def erdoes_gallai(dsequence):
        """Check if the condition of the Erdoes-Gallai inequality is fullfilled."""
        if sum(dsequence) % 2:
            # sum of sequence is odd
            return False
        if Graph.is_degree_sequence(dsequence):
            for k in range(1,len(dsequence) + 1):
                left = sum(dsequence[:k])
                right =  k * (k-1) + sum([min(x,k) for x in dsequence[k:]])
                if left > right:
                    return False
        else:
            # sequence is increasing
            return False
        return True

    def find(self, parent, i): 
        """Find a set of an element i.
        
        This algorithm uses path compression technique.

        Args:
            parent : list
                Parent list.
            i : int
                Element.

        Returns:
            Set of an element i.

        """
        if parent[i] == i: 
            k = i 
        else:
            k = self.find(parent, parent[i]) 
            
        return k
 
    def union(self, parent, rank, x, y): 
        """Return  union of two sets of x and y.
        
        This algoritm sorts uses union by rank.

        Args:
            parent : list
                Parent list of vertices.
            rank : list
                Rank of tree.
            x : list
                Graph vertices.
            y : list
                Graph vertices.

        Returns:
            None.
        """
        xroot = self.find(parent, x) 
        yroot = self.find(parent, y) 
  
        # Attach smaller rank tree under root of high rank tree (Union by Rank) 
        if rank[xroot] < rank[yroot]: 
            parent[xroot] = yroot 
        elif rank[xroot] > rank[yroot]: 
            parent[yroot] = xroot 
  
        # If ranks are same, then make one as root and increment its rank by one 
        else : 
            parent[yroot] = xroot 
            rank[xroot] += 1          
 
    def isCyclic(self): 
        """
        Check whether a given graph contains a cycle or not.

        Returns:
            bool
                True if graph is cyclic and False otherwise.

        """
        # Allocate memory for creating V subsets and 
        # Initialize all subsets as single element sets 
        V = len(self.__vertices)
        parent = [-1]*V
  
        # Iterate through all edges of graph, find subset of both 
        # vertices of every edge, if both subsets are same, then 
        # there is cycle in graph. 
        g = self.__graph
        for k in range(len(g)):
            i,j,w = g[k]
            x = self.find(parent, i)  
            y = self.find(parent, j) 
            if x == y: 
                return True
            self.union(parent,x,y)            
                
    def getMinimumSpanningTree(self): 
        """
        Kruskal's algorithm to find Minimum Spanning Tree (MST)of a given connected, undirected and weighted graph.

        Time Complexity: O(ElogE) or O(ElogV). Here E is the number of edges and V is the number of vertices.
        Sorting of edges takes O(ELogE) time. After sorting, we iterate through all edges and apply find-union algorithm. 
        The find and union operations can take atmost O(LogV) time. So overall complexity is O(ELogE + ELogV) time. 
        The value of E can be atmost O(V2), so O(LogV) are O(LogE) same. Therefore, overall time complexity is O(ElogE) or O(ElogV).
        
        https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/

        Args:
            self: Graph object.
            
        Returns:
            None.

        """
        mst =[] #This will store the resultant MST 
  
        # Step 1:  Sort all the edges in non-decreasing # order of their # weight.  
        # If we are not allowed to change the given graph, we can create a copy of graph 
        self.__graph =  sorted(self.__graph,key=lambda item: item[2]) 
  
        parent = [] ; rank = [] 
  
        # Create V subsets with single elements 
        V = len(self.__vertices)
        for node in range(V): 
            parent.append(node) 
            rank.append(0) 
      
        i = 0 # An index variable, used for sorted edges 
        e = 0 # An index variable, used for result[] 
        
        g = self.__graph
        #vertices = self.__vertices
        inv_vertices = self.__inv_vertices
        weights = self.__weights
        
        # Number of edges to be taken is equal to V-1 
        while e < V and i < len(g): 
  
            # Step 2: Pick the smallest edge and increment the index for next iteration 
            u,v,w =  g[i] 
            #print(i,e,V)
            i += 1
            x = self.find(parent, u) 
            y = self.find(parent ,v)
            #print(u,v,x,y)
  
            # If including this edge does't cause cycle,  
            # Include it in result and increment the index of result for next edge 
            if x != y: 
                e += 1     
                k = str([min(u,v),max(u,v)])
                v1 = inv_vertices[v]
                u1 = inv_vertices[u]
                w = weights[k]
                mst.append([u1,v1,w]) 
                self.union(parent,rank,x,y) 
            else:
                pass # discard the edge 
  
        return mst
            
    def connected(self,t):
        """
        Return connected vertices of a minimum spanning tree.

        Args:
            t : list
                List of vertices.

        Returns:
            Connected vertices of a minimum spanning tree.

        """
        from utils.util import findVariableLag,findVariableLead
        d = {}
        for i,x in enumerate(t):
            u,v,w = x
            if u in d:
                d[u] += [i]
            else:
                d[u] = [i]
            if v in d:
                d[v] += [i]
            else:
                d[v] = [i]
                
        ll = LinkedList()
        u,v,w = t[0]
        self.__connect(ll,d,t,u,v)
            
        current = ll.head
        s = ""; i=0
        while current is not None:
            v = current.value
            if "_plus_" in v:
                lead = findVariableLead(v)
                ind = v.index("_plus")
                v = v[:ind] + "(" + str(lead) + ")"
            elif "_minus_" in v:
                lag = findVariableLag(v)
                ind = v.index("_minus_")
                v = v[:ind] + "(" + str(lag) + ")"
            s += str(v) + "->"
            i += 1
            if i%10 == 0:
                s += "\n"
            current = current.next
        
        return s[:-2]
                          
    def __connect(self,lli,d,t,u,v):
        """
        Build a list of connected nodes.

        Args:
            lli : LinkedList
                List of nodes.
            d : dict
                Dictionary that has node labels as keys and their edge numbers as values.
            t : list
                Minimum spanning tree.
            u : str
                Node label.
            v : str
                Node label.

        Returns:
            List of connected nodes.
        """
        if u==v:
            return
        
        head = lli.head
        lst = LinkedList.toList(lli)
        if not u in lst:
            if not head is None and v == head.value:
                lli.push(u)
            else:
                lli.append(u)
                   
        lst = LinkedList.toList(lli)
        if not v in lst:
            if not head is None and u == head.value:
                lli.push(v)
            else:
                lli.append(v)
            
        lst = LinkedList.toList(lli)
        #LinkedList.display(li)
        for k in d[u]:
            u1,v1,w1 = t[k]
            if not u1 in lst or not v1 in lst:
                self.__connect(lli,d,t,u1,v1)
                lst = LinkedList.toList(lli)
        for k in d[v]:
            u2,v2,w2 = t[k]
            if not u2 in lst or not v2 in lst:
                self.__connect(lli,d,t,u2,v2)
                lst = LinkedList.toList(lli)
                  
    def __str__(self):
        """Represent Graph object as a string."""
        # d = self.__graph_dict
        res = "\nModel Equations Graph Information:\n----------------------------------"
        # res += "\nVertices: "
        # for k in d:
        #     res += str(k) + " "
        # res += "\nEdges: "
        # edges = self.__generate_edges()
        # for edge in edges:
        #     res += str(edge) + " "
        # v = self.__vertices
        # res += "\nNumber of vertices: " + str(len(v))
        res += "\nDiameter: {}".format(self.diameter())
        res += "\nIs cyclic: {}".format(self.isCyclic())
        res += "\nMinimum degree: {}".format(self.delta())
        res += "\nMaximum degree: {}".format(self.Delta())
        res += "\nDensity: {:.2f}".format(self.density())
        # components = self.get_components()
        # res += "\nNumber of components: {}".format(len(components))
        # res += "\nComponents: {}".format(components)
        # mst = self.getMinimumSpanningTree()
        # print the contents of result[] to display the built MST 
        # res += "\nFollowing are the edges in the constructed MST:\n"
        # for u,v,weight  in mst: 
        #     res += "Edge: {0}--{1},  weight: {2}\n".format(u,v,weight)
            
        # res += "\nLinked list of connected vertices of minimum spanning tree:\n"
        # res += self.connected(mst)
        res += "\n\n"
        return res
      
    def __repr__(self):
        """Graph object represntation."""
        return self.__str__()
               
    def printBiconnectedComponents(self):
        print (f"\nThere are {self.count} biconnected components in a graph:")
        for i,c in enumerate(self.components):
            comp = self.components[c]
            x = [self.im[s[0]] + "->" + self.im[s[1]] for s in comp]
            print(f"{i+1}: {', '.join(x)}\n")
                           
    def summary(self):
        import numpy as np
        #print (f"\nNumber of graph vertices {self.V}")
        print (f"There are {self.count} biconnected components in a graph:")
        components = self.getBiconnectedComponents()
        all_sizes = list()
        for comp in components:
            sz = len(comp)
            all_sizes.append(sz)
        sizes = sorted(set(all_sizes))
        all_sizes = np.array(all_sizes)
        for x in sizes:
            n = sum(all_sizes==x)
            print(f"{n} component(s) of size {x}")
            
    def getBiconnectedComponents(self):
        """Get biconnected components in a graph."""
        components = list()
        for c in self.components:
            comp = self.components[c]
            lst = [(self.im[s[0]],self.im[s[1]]) for s in comp]
            components.append(lst)
            
        return components
    

class Node(object):
    """Node class."""
    
    value = None
    next = None
    
    def __init__(self, value, next=None):
        self.value = value
        self.next = next
                          
    def __str__(self):
        """Representats Graph object as a string."""
        res = str(self.value)
        
        return res
                
    def __repr__(self):
        """Graph object represntation."""
        return self.__str__()


class LinkedList(object):
    """A Simple linked list class."""
    
    def __init__(self, sequence=None):
        if sequence is None:
            self.head = None
        else:
            self.head = Node(sequence[0])
            current = self.head
            for item in sequence[1:]:
                current.next = Node(item)
                current = current.next           
            
    def push(self, new_data): 
        """Insert a new node at the beginning."""
        # 1 & 2: Allocate the Node & 
        #        Put in the data 
        new_node = Node(new_data) 
        # 3. Make next of new Node as head 
        new_node.next = self.head 
        # 4. Move the head to point to new Node 
        self.head = new_node   
    
    def insertAfter(self, prev_node, new_data): 
        """Insert a new node after the given prev_node."""
        # 1. check if the given prev_node exists 
        if prev_node is None: 
            print("The given previous node must exist in the Linked List.")
            return
        #  2. create new node & 
        #      Put in the data 
        new_node = Node(new_data) 
        # 4. Make next of new Node as next of prev_node 
        new_node.next = prev_node.next
        # 5. make next of prev_node as new_node 
        prev_node.next = new_node 
     
    def getParentNode(self,v):
        """Get previous node in the linked list."""
        n = self.head 
        while (n): 
            if n.next == v:
                return n
        return None
                    
    def insertBefore(self, node, new_data): 
        """Insert a new node before the given node."""
        # 1. check if the given prev_node exists 
        if node is None: 
            print("The given node must exist in the Linked List.")
            return
        parentNode = self.getParentNode(node)
        if not parentNode is None:
            #  2. create new node and put in the data 
            new_node = Node(new_data) 
            # 4. Make next of new Node as next of prev_node 
            parentNode.next = new_node
            # 5. make next of prev_node as new_node 
            new_node.next = node 
            
    def append(self, new_data): 
        """Append a new node at the end."""
        # 1. Create a new node 
        # 2. Put in the data 
        # 3. Set next as None 
        new_node = Node(new_data) 
        # 4. If the Linked List is empty, then make the new node as head 
        if self.head is None: 
            self.head = new_node 
        else:
            # 5. Else traverse till the last node 
            last = self.head 
            while (last.next): 
                last = last.next
            # 6. Change the next of last node 
            last.next =  new_node 
                           
    def getNode(self,v):
        """Return node of a linked list by value."""
        n = self.head 
        while (n): 
            if v == n.value:
                return n
            n = n.next
            
        return None
       
    @staticmethod
    def toList(self):
        """Convert a linked list to a list."""
        temp = self.head 
        lst = []
        while (temp): 
            lst.append(temp.value) 
            temp = temp.next
        return lst
    
    @staticmethod
    def display(self): 
        """Display the linked list."""
        temp = self.head 
        while (temp): 
            print(temp.value)
            temp = temp.next
                                      
    def __str__(self):
        """Representats LinkedList object as a string."""
        res = ""
        temp = self.head 
        while (temp): 
            res += str(temp) + "\n"
            temp = temp.next
        return res       
        
    def __repr__(self):
        """Graph object represntation."""
        return self.__str__()
            
          
if __name__ == '__main__':
    """ Test methods of Graph class. """
      
    # Create a graph given in the above diagram 
    nodes = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11}
    g = Graph(m=nodes) 
    g.addEdge('A','B')
    g.addEdge('B','C')
    g.addEdge('B','D')
    g.addEdge('C','D')
    g.addEdge('C','E')
    g.addEdge('D','E')
    g.addEdge('B','F')
    g.addEdge('A','G')
    g.addEdge('F','G')
    g.addEdge('F','H')
    g.addEdge('F','I')
    g.addEdge('H','I')
    g.addEdge('I','J')
    g.addEdge('K','L')
      
    g.BCC() 
    g.printBiconnectedComponents()
      
    if False:
        g = { "a" : ["d"],
              "b" : ["c","e"],
              "c" : ["b", "c", "d", "e"],
              "d" : ["a", "c","b"],
              "e" : ["c"],
              "f" : []
            }
    
        graph = Graph(graph_dict=g)
        print(graph)
        
        print("Veryices degree:")
        for node in graph.vertices():
            print("{0}  -  {1}".format(node,graph.vertex_degree(node)))
    
        print("List of isolated vertices:")
        print(graph.find_isolated_vertices())
    
        print("""A path from "a" to "e":""")
        print(graph.find_path("a", "e"))
    
        print("""A shortest path from "a" to "e":""")
        print(graph.find_shortest_path("a", "e"))
        
        print("""All pathes from "a" to "e":""")
        print(graph.find_all_paths("a","e"))
    
        print("""Check graph connectivity from "a":""")
        print(graph.is_connected(start_vertex="a"))
        
        print(""""Find distance from "a" to "e":""")
        print(graph.find_distance("a", "e"))
        
        print(""""Find connected vertices from "a":""")
        print(graph.get_connected_vertices("a"))
        
        print(""""Find connected vertices from "f":""")
        print(graph.get_connected_vertices("f"))
        
        print("The maximum degree of the graph is:")
        print(graph.Delta())
    
        print("The minimum degree of the graph is:")
        print(graph.delta())
    
        print("Edges:")
        print(graph.edges())
    
        print("Degree Sequence: ")
        ds = graph.degree_sequence()
        print(ds)
    
        fullfilling = [ [2, 2, 2, 2, 1, 1], 
                        [3, 3, 3, 3, 3, 3],
                        [3, 3, 2, 1, 1]
                      ] 
        non_fullfilling = [ [4, 3, 2, 2, 2, 1, 1],
                            [6, 6, 5, 4, 4, 2, 1],
                            [3, 3, 3, 1] 
                          ]
    
        for sequence in fullfilling + non_fullfilling :
            print(sequence, Graph.erdoes_gallai(sequence))
    
        print("Add vertex 'z':")
        graph.add_vertex("z")
        print(graph)
    
        print("Add edge ('x','y'): ")
        graph.add_edge(('x', 'y'))
        print(graph)
    
        print("Add edge ('a','d'): ")
        graph.add_edge(('a', 'd'))
        print(graph)
        
        print("Add edge ('a','x'): ")
        graph.add_edge(('a', 'x'))
        print(graph)
    
