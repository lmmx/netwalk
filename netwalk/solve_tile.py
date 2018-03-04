# solver methods at the tile level
import numpy as np
from .sym_dict import server, terminal, l_wire, c_wire, t_wire

def tile_solvers(t):
    """
    Run a series of solvers for a tile.
    """
    if not t.solved:
        print("Solving tile...")
    # blank tiles should have been solved upon ``tile.initialise_solve_state``
    if t.component is None:
        assert t.solved
        return
    # second: if 2 adjacent terminals, or separated by 1 tile
    if type(t.component) == terminal:
        for (a, t_a) in t.adjacent_tiles.items():
            print(f"Testing {a}")
            a_inv = (a + 2) % 4
            if t_a.component is not None:
                if np.sum(t_a.component.directions) == 1:
                    # don't connect two single-connection components
                    t_a.set_avoid([a_inv])
                    t.set_avoid([a])
            if type(t_a.component) == terminal:
                print(f"Avoiding {a}")
                t.set_avoid([a])
                t_a.set_avoid([a_inv])
            elif type(t_a.adjacent_tiles[a]) == terminal:
                # no connecting terminals (avoid enclosing pair of terminals)
                t_a.set_avoid([a, a_inv])
                off_axis = [int(o) for o in np.arange(4) if o not in (a, a_inv)]
                for o in off_axis:
                    if type(t_a.adjacent_tiles[o]) == terminal:
                        # must connect at least to the other direction
                        o_inv = (o + 2) % 4
                        assert type(t_a.adjacent_tiles[o_inv]) != terminal
                        t_a.fix_connection([o_inv])
    # ==REMOVED BELOW==: tried to fix erroneous (unavailable) sides => error
    # needed to solve, so adding back with check: ``np.sum(fixed, avoid) < 4``
    for (a, t_a) in t.adjacent_tiles.items():
        if t_a.component is None:
            continue
        a_inv = (a + 2) % 4
        if np.sum([t.fixed, t.avoid]) == 4:
            continue
        no_meddling = np.sum([t_a.fixed, t_a.avoid]) == 4
        if t_a.solved and t_a.component.directions[a_inv]:
            assert t_a.fixed[a_inv]
            t.fix_connection([a])
            if not no_meddling:
                print("Let the meddling commence... >:-)")
                t_a.fix_connection([a_inv])
        if t_a.solved and not t_a.component.directions[a_inv]:
            assert t_a.avoid[a_inv]
            if not no_meddling:
                print("Let the meddling commence... >:-)")
                t_a.set_avoid([a_inv])
            t.set_avoid([a])
