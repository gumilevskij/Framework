from functools import reduce
from operator import mul
from quantecon import cartesian
import numpy as np
from numpy import linspace

def prod(l): 
	return reduce(mul, l, 1.0)


def mlinspace(a,b,orders,out=None):
    """Return Cartesian grid."""
    sl = [linspace(a[i],b[i],orders[i]) for i in range(len(a))]   
    if out is None:
        out = cartesian(sl)
    else:
        cartesian(sl, out)
    return out
	
	
class Grid:
    """
    Grid class is a parent class of EmptyGrid, PointGrid, UnstructuredGrid, 
    CartesianGrid, NonUniformCartesianGrid, and SmolyakGrid sub-classes.
    """

    def nodes(self):
        return self.__nodes__

    def n_nodes(self):
        return self.__nodes__.shape[0]

    def node(self, i):
        return self.__nodes__[i,:]


class EmptyGrid(Grid):
    """
    EmptyGrid is a sub-class of Grid class.
    
    .. currentmodule: numeric.dp
    .. autoclass:: numeric.dp.grids.Grid
       :members:
    """

    type = 'empty'

    def nodes(self):
        return None
    def n_nodes(self):
        return 0
    def node(self, i):
        return None


class PointGrid(Grid):
    """
    PointGrid is a sub-class of Grid class.
    
    .. currentmodule: numeric.dp
    .. autoclass:: numeric.dp.grids.Grid
       :members:
    """
    type = 'point'

    def __init__(self, point):
        self.point = np.array(point)

    def nodes(self):
        return None
    def n_nodes(self):
        return 1
    def node(self, i):
        return None


class UnstructuredGrid(Grid):
    """
    UnstructuredGrid is a sub-class of Grid class.
    
    .. currentmodule: numeric.dp
    .. autoclass:: numeric.dp.grids.Grid
       :members:
    """
    type = 'unstructured'

    def __init__(self, nodes):
        nodes = np.array(nodes, dtype=float)
        self.min = nodes.min(axis=0)
        self.max = nodes.max(axis=0)
        self.__nodes__ = nodes


class CartesianGrid(Grid):
    """
    CartesianGrid is a sub-class of Grid class.
    
    .. currentmodule: numeric.dp
    .. autoclass:: numeric.dp.grids.Grid
       :members:
    """
    type = 'cartesian'

    def __init__(self, min, max, n=[]):

        self.min = np.array(min, dtype=float)
        self.max = np.array(max, dtype=float)
        if len(n) == 0:
            self.n = np.zeros(n, dtype=int) + 20
        else:
            self.n = np.array(n, dtype=int)
        self.__nodes__ = mlinspace(self.min, self.max, self.n)


class NonUniformCartesianGrid(Grid):
    """
    NonUniformCartesianGrid is a sub-class of Grid class.
    
    .. currentmodule: numeric.dps.grid
    .. autoclass::  numeric.dp.grids.Grid
       :members:
    """
    type = "NonUniformCartesian"

    def __init__(self, list_of_nodes):
        list_of_nodes = [np.array(l) for l in list_of_nodes]
        self.min = [min(l) for l in list_of_nodes]
        self.max = [max(l) for l in list_of_nodes]
        self.__nodes__ = cartesian(list_of_nodes)


class SmolyakGrid(Grid):
    """
    SmolyakGrid is a sub-class of Grid class.
    
    .. currentmodule: numeric.dps.grid
    .. autoclass:: numeric.dp.grids.Grid
       :members:
    """
    type = "Smolyak"

    def __init__(self, min, max, mu=2):

        print(min, max, mu)
        from interpolation.smolyak import SmolyakGrid as ISmolyakGrid
        min = np.array(min)
        max = np.array(max)
        self.min = min
        self.max = max
        self.mu = mu
        d = len(min)
        print(mu)
        sg = ISmolyakGrid(d, mu, lb=min, ub=max)
        self.sg = sg
        self.__nodes__ = sg.grid


def cat_grids(grid_1, grid_2):
    """
    Concatenate grids.

    Parameters:
        grid_1 : Grid
            The first grid object.
        grid_2 : Grid
            The second grid object.

    Raises:
        Exception if grid is not Cartesian nor empty. is not Car

    Returns:
        Grid object.

    """

    if isinstance(grid_1, EmptyGrid):
        return grid_2
    if isinstance(grid_1, CartesianGrid) and isinstance(grid_2, CartesianGrid):
        min = np.concatenate([grid_1.min, grid_2.min])
        max = np.concatenate([grid_1.max, grid_2.max])
        n = np.concatenate([grid_1.n, grid_2.n])
        return CartesianGrid(min, max, n)
    else:
        raise Exception("Not Implemented.")

# compat
def node(grid, i): 
    return grid.node(i)

def nodes(grid): 
    return grid.nodes()

def n_nodes(grid): 
    return grid.n_nodes()


def Plot(grid,title):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8,6))
    n,m = grid.shape
    if m == 2:
        ax = fig.add_subplot(111)
        for i in range(n):
            X,Y = grid[i]
            ax.scatter(X,Y,linewidths=4)
    elif m == 3:
        ax = fig.add_subplot(111, projection='3d')
        for i in range(n):
            X,Y,Z = grid[i]
            ax.scatter(X,Y,Z,linewidths=4)
        
    plt.box(True)
    plt.grid(True)
    plt.title(title,fontsize = 'x-large')
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.show(block=False)
    
    
if __name__ == "__main__":
    """Main entry point"""
    
    print("\nCartesian Grid:")
    grid = CartesianGrid([0.1, 0.3], [9, 0.4], [50, 10])
    nodes_c = nodes(grid)
    print(grid.nodes())
    print(nodes_c)
    
    #Plot(grid=nodes_c,title='Cartesian Grid')

    print("\nUnstructuredGrid:")
    ugrid = UnstructuredGrid([[0.1, 0.3], [9, 0.4], [50, 10]])
    nodes_ug = nodes(ugrid)
    print(nodes_ug)
    print(node(ugrid,0))
    print(n_nodes(ugrid))
    
    #Plot(grid=nodes_ug,title='Unstructured Grid')

    print("\nNon Uniform CartesianGrid:")
    ugrid = NonUniformCartesianGrid([[0.1, 0.3], [9, 0.4], [50, 10]])
    nodes_nuc = nodes(ugrid)
    print(nodes_nuc)
    print(node(ugrid,0))
    print(n_nodes(ugrid))
    
    Plot(grid=nodes_nuc,title='Uniform Cartesian Grid')

    print("\nSmolyak Grid:")
    sg = SmolyakGrid([0.1, 0.2], [1.0, 2.0], 2)
    nodes_sg = nodes(sg)
    print(nodes_sg)
    print(node(sg, 1))
    print(n_nodes(sg))
    
    Plot(grid=nodes_sg,title='Smolyak Grid')
    