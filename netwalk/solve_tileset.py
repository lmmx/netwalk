# solver methods at the tileset level

class tileset_solver(object):
    """
    A class with various methods to solve a NetWalk tileset (stored as
    the ``solver`` attribute of the ``tileset`` class).
    """
    def __init__(self, tiling):
        assert not tiling.solved
        self.__parent__ = tiling
        self.solved = False
        self.solved_tiles = self.get_solved_tiles()
        
    def get_solved_tiles(self):
        """
        Call the parent [tileset] method to update its ``solved_tiles``
        attribute, and return this.
        """
        # Q: is this good OOP? should I just put this call in the attribute?
        # A: using it for now, seems nicer to abstract away ``__parent__``
        return self.__parent__.get_solved_tiles()

    def set_tile_solved(self, tile_pos: list):
        """
        Set one or more tiles to 'solved', where each list item is a tuple
        of [row, column]
        """
        # - need to first modify the appropriate tile's ``solved``
        #   attribute in the ``self.__parent__.tiles`` array,
        # - then need to call ``self.get_solved_tiles`` to update
        #   ``self.__parent__.solved_tiles``
        for row_pos, col_pos in tile_pos:
            self.__parent__.tiles[row_pos][col_pos].solve()
        # don't need to receive the result, just ensure it completes:
        assert type(self.get_solved_tiles()) == np.ndarray
        return

    def resolve(self):
        """
        Attempt to solve the puzzle when at least one attempt already made.
        """
        # TODO
        return

    def __repr__(self):
        if self.solved:
            m = 'unsolved'
            n = np.count_nonzero(self.get_solved_tiles())
        else:
            m = 'solved'
            n = 'all'
        return f"The puzzle is {m} ({n} of {n.size} tiles) solved"
