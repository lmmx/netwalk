# determining adjacent tiles efficiently
from .tiling import tileset

class tiling_adjacencies(object):
    """
    The set (with repeats) of all adjacencies of a tileset. An adjacency
    is an interface
    """
    def __init__(self, tset: tileset):
        self.__parent__ = tset
        self.interfaces = interface_set(self)
        self.adjacencies = generate_adjacencies(self)
        return

    def generate_adjacencies(self):
        """
        The set of all ``adjacency`` objects, i.e. the set with one
        ``adjacency`` per tile for all tiles in a ``tileset``.
        """
        return

class adjacency(object):
    """
    A vector of four, separate, directional ``adjacence`` instances of a
    given tile, i.e. a clockwise vector of ``interface`` references,
    which represent the shared interfaces along a tile's four edges.
    """
    def __init__(self):
        self.adj = [self.get_adjacence(n) for n in np.arange(4)]
        return

    def get_adjacence(self, n: np.typeDict['int']):
        """
        Get a single adjacence, for the direction (0-3) indicated by ``n``.
        """
        adj = adjacence(n) # TODO: need to pass in... tile?
        return adj

class adjacence(object):
    """
    A single directional adjacence, containing [a reference to] a single
    interface object (which belongs to the interface set of the tileset).
    """
    def __init__(self, n: np.typeDict['int']):
        # TODO: self.interface = ?
        return

class interface(object):
    """
    The edge shared by two tiles is an 'interface'. Each interface has an
    'address' in the form of an integer in ``range(0, (n)*(2*n))``, i.e.
    the count of each element of an n by 2n array. The ``before`` attribute
    indicates the left/upper tile (depending on whether the interface in
    question is horizontal/vertical) and ``after`` indicates its complement.
    """
    def __init__(self, i: np.typeDict['int'], tset: tileset):
        self.__tileset__ = tset
        self.index = i
        self.horizontal = self.is_horizontal()
        self._pre = None
        self._post = None
        self.map_pre_post()
        return

    def is_horizontal(self):
        n = len(self.__tileset__.tiles)
        return is_h = self.index % (2*n) < n

    def map_pre_post(self):
        """
        Initialise interface tile references.
        """
        self.map_pre()
        self.map_post()
        assert None not in (self.pre, self.post)
        return

    @property
    def pre(self):
        return self._pre

    @pre.setter(self, val):
        self._pre = val
        return

    @property
    def post(self):
        return self._post

    @post.setter(self, val):
        self._post = val
        return

    def map_pre(self):
        """
        Set adjacent tile 'before' an interface edge (upper/left).
        """
        if self.horizontal:
            self.pre = self.get_adjacent_tile(self.__tileset__, 0)
        else:
            self.pre = self.get_adjacent_tile(self.__tileset__, 3)
        return

    def map_post(self):
        """
        Set adjacent tile 'after' an interface edge (below/right).
        """
        if self.horizontal:
            self.post = self.get_adjacent_tile(self.__tileset__, 2)
        else:
            self.post = self.get_adjacent_tile(self.__tileset__, 1)
        return

    def get_adjacent_tile(self, tset: tileset, a: int):
        """
        Handle the retrieval of tiles from the parent tileset, including
        edge wrapping logic using the index of the interface. Note that
        the edge logic applies to the n by 2n interface set, not to the
        tileset, which simplifies the task of handling edge wrapping.
        """
        assert self.horizontal == (a % 2 == 0) and 0 <= a < 4
        n = len(tset.tiles)
        i = self.index.view()
        if a == 0 and i < n:
            # tile is on the top edge of tileset, so edge wraps
            return tset.tiles[-1][i]
        elif a == 1 and i % n == (n-1):
            # tile is on the right edge of tileset, so edge wraps
            return tset.tiles[int((i + 1) / n)][0]
        # N.B. a=2 on the bottom edge does not need to be handled,
        # as interfaces on the bottom row are vertical only
        elif a == 3 and i % n == 0:
            # tile is on the left edge of tileset, so edge wraps
            return tset.tiles[int(i / n)][-1]
        else:
            # the adjacent tile does not cross tileset outer edge
            if self.horizontal:
                if a == 2:
                    # decrement the interface index by n per row to
                    # retrieve the corresponding tileset index t_i
                    t_i = int(((i - (i % n))/2) + (i % n))
                    return tset.tiles[int((t_i - (t_i % n)) / n)][t_i % n]
                else:
                    # a == 0 so as for a == 2, but also decrement i by 2n
                    # (the 0th row was removed by the ``i < n`` test)
                    t_i = int(((i - 2*n - (i % n))/2) - n + (i % n))
                    return tset.tiles[int((t_i - (t_i % n)) / n)][t_i % n]
            else:
                if a == 1:
                    # as for a == 2, but decrement n to 0-base the 1st row
                    t_i = int(((i - n - (i % n))/2) + (i % n))
                    return tset.tiles[int((t_i - (t_i % n)) / n)][t_i % n]
                else:
                    # a == 3 so as for a == 1, but also decrement the row
                    # offset (i.e. modulus remainder) by 1 (the 0th column
                    # was removed by the ``i % n == 0`` test)
                    t_i = int(((i - n - (i % n))/2) + (i % n) - 1)
                    return tset.tiles[int((t_i - (t_i % n)) / n)][t_i % n]

class interface_set(object):
    """
    The set (without repeats) of all interfaces of a tileset. A
    tileset of dimensions n by n will have ``interface_set`` dimensions
    of [width] n by [height] 2n. N.B. the interface set is NOT the set
    of all edges, which has repeated elements (and is therefore
    implemented in ``tiling_adjacencies``, with integers indexing the
    reference elements of the ``interface_set``).
    """
    def __init__(self, adj: tiling_adjacencies):
        self.__parent__ = adj
        self.__tileset__ = adj.__parent__
        self._interfaces = []
        self.generate_interfaces()
        return

    @property
    def interfaces(self):
        return self._interfaces

    @interfaces.setter
    def interfaces(self, val: list):
        self._interfaces = val
        return

    def generate_interfaces(self):
        assert len(self.interfaces == 0) # only used upon initialisation
        self.interfaces = generate_tile_interfaces(self.__tileset__)
        return

def generate_tile_interfaces(tset: tileset) -> list:
    """
    Using the ``tiles`` [attribute] from a ``tileset``, i.e. a list of
    rows of ``tile`` instances, produce the full set of interfaces.
    """
    n = len(tset.tiles)
    interface_index = np.array(np.arange(n*2*n)).reshape(2*n,n)
    interfaces = [interface(i, tset) for i in interface_index]
    return interfaces
