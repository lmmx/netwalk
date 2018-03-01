# solver methods at the tile level
from .sym_dict import server, terminal, l_wire, c_wire, t_wire
from .tiling import tile

def tile_solvers(t: tile):
    """
    Run a series of solvers for a tile.
    """
    # first: if adjacent to blank tile, avoid that direction
    for (a, t_a) in t.adjacent_tiles.items():
        a_inv = (a + 2) % 4
        if t_a.component is None:
            t.set_avoid([a])
            assert t_a.solved # blank tiles were solved at tileset init
    # second: if 2 adjacent terminals, or separated by 1 tile
    if type(t.component) == terminal:
        for (a, t_a) in t.adjacent_tiles.items():
            a_inv = (a + 2) % 4
            if type(t_a.component) == terminal:
                t.set_avoid([a])
                t_a.set_avoid([a_inv])
            elif type(t_a.adjacent_tiles[a]) == terminal:
                t_a.set_avoid([a, a_inv])
