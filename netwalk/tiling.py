# Classes for handling puzzle tiles
import numpy as np
from imageio.core.util import Image

class tileset(object):
    """
    A class connecting the parsed image file to the internal grid layout
    and component representation.
    """
    def __init__(self, img: Image, tile_segments: np.ndarray):
        self.segments = segment(tile_segments)
        # self.tiles = seg_to_tiles(img, self.segments)
        # TODO: instantiate ``tile`` objects from these segmented arrays
        #       using the ``img`` parameter's provided RGB information

    def __repr__(self):
        return f"A set of puzzle tiles"

def segment(tiling_grid: np.ndarray):
    """
    Creates ``tile_segment``` objects when a ``tileset`` is instantiated,
    i.e. it's segment as a verb (this function creates segment sets). As
    each line of the image is read in, start and end points of the tiles
    are recorded, then extended down up to the point they end (which is
    the first line on which there are no segments, i.e. the tile border).
    The top line of a segment set toggles ``in_seg`` to True, and each
    tile so identified is instantiated as a ``tile_segment``.
    """
    in_seg = False
    seg_store = []
    for n, line in enumerate(tiling_grid):
        if not in_seg:
            if np.any(line):
                # a new segment set starts on this line
                in_seg = True
                seg_starts = []
                seg_ends = []
                for i in range(1,len(line)):
                    if line[i] is not line[i-1]:
                        if line[i]:
                            seg_starts.append(i)
                        else:
                            seg_ends.append(i-1)
                segs = list(zip(seg_starts, seg_ends))
                seg_store.append({'start_y': n, 'end_y': None,
                                  'x_ranges': segs})
        else:
            if not np.any(line):
                # a segment set ended on the previous line
                seg_store[-1]['end_y'] = n-1
                in_seg = False
            # else it's a continuation of the current segment set, do nothing
    if in_seg:
        # final seg set ended on final line (n doesn't increment at the end)
        seg_store[-1]['end_y'] = n
    seg_set = []
    for seg_row in seg_store:
        # turn segment ranges into (x,y) coordinates, [top left, bottom right]
        seg_list = [[(x[0], seg_row['start_y']),
                     (x[1], seg_row['end_y'])  ] for x in seg_row['x_ranges']]
        seg_set.append(seg_list)
    return seg_set
    # TODO: turn seg_set into tile_set by instantiating tile classes per seg.
    # return tile_set

class tile(object):
    """
    ...
    """
    def __init__(self, image_segment: np.ndarray):
        self.corners = None
