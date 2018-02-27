# solver methods at the tile level
from .sym_dict import server, terminal, l_wire, c_wire, t_wire
from .tiling import tile

def tile_solvers(t: tile):
    """
    Run a series of solvers for a tile.
    """
    if type(t.component) == terminal:
        for a, t_a in enumerate(t.adjacent_tiles):
            # TODO: (once get_adjacent_tiles implemented)
