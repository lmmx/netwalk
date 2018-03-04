# solver methods at the tileset level
import numpy as np
from .solve_tile import tile_solvers

class tileset_solver(object):
    """
    A class with various methods to solve a NetWalk tileset (stored as
    the ``solver`` attribute of the ``tileset`` class).
    """
    def __init__(self, tiling):
        assert not tiling.solved
        self.__parent__ = tiling
        self._solved = False
        self._solved_tiles = self.solved_tiles
        self.solve() # perform initial solve

    def solve(self):
        """
        Perform initial solve...
        """
        print(self)
        for row in self.__parent__.tiles:
            for t in row:
                if self.solved:
                    return
                tile_solvers(t)
        return

    @property
    def solved(self):
        """
        Determine whether the puzzle has been solved or not (recheck every time).
        """
        self._solved = np.all(self.solved_tiles)
        return self._solved

    @property
    def solved_tiles(self):
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
        # - then need to call ``self.solved_tiles`` to update
        #   ``self.__parent__.solved_tiles`` (EDIT: huh?) confirm...
        for row_pos, col_pos in tile_pos:
            self.__parent__.tiles[row_pos][col_pos].solve()
        # don't need to receive the result, just ensure it completes:
        assert type(self.solved_tiles) == np.ndarray
        return

    def resolve(self):
        """
        Attempt to solve the puzzle when at least one attempt already made.
        """
        self.solve()
        print(self)
        return

    def __repr__(self):
        m = ''
        if not self.solved: m += 'un'
        n = np.count_nonzero(self.solved_tiles)
        return f"The puzzle is {m}solved ({n} of {self.solved_tiles.size} tiles solved)"
