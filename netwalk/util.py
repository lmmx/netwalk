# convenience functions to be imported by other modules
from datetime import datetime as dt
from numpy import ones, logical_and, triu, rot90

def ptime():
    """
    Print the current time (hours:minutes:seconds), for debug time profiling.
    """
    t_now = dt.now()
    t = f"{t_now.hour}:{t_now.minute}-{t_now.second}s:{t_now.microsecond}"
    return t

def get_orientation_quadrants(ref_array_width: int) -> dict:
    """
    Using a numpy array [specified by width N, presumed NxN] as reference
    (for the tile image), create and segment a tile mask into 4 triangular
    quadrants between the 2 diagonals, which will then be used to determine
    the direction of a given component (i.e. wire/s).
    """
    print(f"{ptime()} Generating quadrants...")
    mask = ones(shape=[ref_array_width]*2, dtype=bool)
    top_tri = logical_and(triu(mask, k=1), ~triu(mask)[::-1])
    orientations = ['up', 'right', 'down', 'left']
    # rotate top quadrant for right, bottom, and left bool matrices
    tris = [top_tri] + [rot90(top_tri, k=n) for n in range(3,0,-1)]
    orient_dict = dict(zip(orientations, tris))
    return orient_dict
