# Classes for handling puzzle tiles
import numpy as np
from imageio.core.util import Image
from itertools import chain
from .colour_dict import game_colours
from .sym_dict import server, terminal, l_wire, c_wire, t_wire
from .util import ptime

class tileset(object):
    """
    A class connecting the parsed image file to the internal grid layout
    and component representation.
    """
    def __init__(self, img: Image, tile_mask: np.ndarray):
        self.segments = segment(tile_mask)
        self.tiles = tile_segments(img, self.segments)
        print(f"{ptime()} Initialising tileset...")

    def __repr__(self):
        return f"A set of {len(self.segments[0])}x{len(self.segments)} tiles."

def segment(tiling_grid: np.ndarray) -> list:
    """
    Creates ``tile_segment``` objects when a ``tileset`` is instantiated,
    i.e. it's segment as a verb (this function creates segment sets). As
    each line of the image is read in, start and end points of the tiles
    are recorded, then extended down up to the point they end (which is
    the first line on which there are no segments, i.e. the tile border).
    The top line of a segment set toggles ``in_seg`` to True, and each
    tile so identified is instantiated as a ``tile_segment``.
    """
    #print(f"{ptime()} Segmenting tiling grid...")
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

def tile_segments(img: Image, segments: list) -> list:
    """
    Turn the list of segments returned from ``segment`` into a list of
    rows of ``tile`` objects.
    """
    tile_set = []
    for i, seg_row in enumerate(segments):
        print(f"{ptime()} Scanning row {i}")
        tile_row = []
        for j, seg in enumerate(seg_row):
            print(f"{ptime()} Scanning row {i} tile {j}")
            (xs, ys), (xe, ye) = seg
            t = tile(img[ys:ye+1, xs:xe+1], seg, i, j)
            tile_row.append(t)
        tile_set.append(tile_row)
    return tile_set

def detect_colour(colour: list, img: Image) -> bool:
    """
    Detect the given colour in an image tile by matching at least 1 RGB pixel
    across the rows of the image data. Return ``True`` if colour is detected.
    """
    tile_palette = np.unique(list(chain.from_iterable(img.tolist())), axis=0)
    return np.any([np.array_equal(x, colour) for x in tile_palette])

def scan_tile(img: Image) -> dict:
    """
    Run component detection on the tiled image region and return
    a boolean dict corresponding to that in ``.colour_dict.game_colours``.
    """
    print(f"{ptime()} Scanning tile...")
    detected_cols = dict(zip(game_colours.keys(), [False] * len(game_colours)))
    if np.all(img == game_colours['space']):
        detected_cols['space'] = True
        return detected_cols
    for label, rgb in game_colours.items():
        if detect_colour(rgb, img):
            # print(f"Detected {label}...")
            detected_cols[label] = True
    return detected_cols

def get_orientation_quadrants(ref_array: Image) -> dict:
    """
    Using a numpy array for reference (of the tile image), segment the
    tile into 4 triangular quadrants between the 2 diagonals, which are
    then used to determine the direction of a given component (wires).
    """
    print(f"{ptime()} Generating quadrants...")
    mask = np.ones(shape=ref_array.shape[0:2], dtype=bool)
    top_tri = np.logical_and(np.triu(mask, k=1), ~np.triu(mask)[::-1])
    orientations = ['up', 'right', 'down', 'left']
    # rotate top quadrant for right, bottom, and left bool matrices
    tris = [top_tri] + [np.rot90(top_tri, k=n) for n in range(3,0,-1)]
    orient_dict = dict(zip(orientations, [np.argwhere(x) for x in tris]))
    return orient_dict

def detect_wire_orientation(on: bool, img: Image):
    """
    Detect the output direction(s) of the wire shown in the tile image.
    """
    if on:
        colour = game_colours['wire_on_in']
    else:
        colour = game_colours['wire_off_in']
    print(f"{ptime()} Scanning wire coords...")
    wire = np.argwhere(np.all(img == colour, axis=-1))
    quads = get_orientation_quadrants(img)
    # iterate over 4 quadrants, return labels of those containing wire coords
    return detect_quad_members(quads, wire)

def detect_quad_members(quadrants: dict, member_coords: np.ndarray):
    """
    Detect which quadrants the wire(s) on a given tile are in.
    """
    print(f"{ptime()} Detecting quadrant members...")
    orientation_vec = []
    for quadrant in quadrants.items():
        # TODO: speed up: skip contiguous pixels when one is categorised?
        if detect_quad_member(quadrant, member_coords):
            print(f"{ptime()} Detected {quadrant[0]} quadrant")
            orientation_vec += [quadrant[0]]
        else:
            print(f"{ptime()} Nothing in {quadrant[0]} quadrant")
    #[q for (q,tri) in quadrants.items() if np.any([np.any([\
    #    a.min() for a in b]) for b in [x==tri for x in member_coords]])]
    print(orientation_vec)
    return orientation_vec

def detect_quad_member(quadrant: tuple, member_coords: np.ndarray):
    """
    For a single quadrant, detect the presence of at least one coordinate
    in the associated coordinate array, returning as soon as possible.
    """
    assert len(quadrant) == 2
    for x in member_coords:
        q, tri = quadrant
        #print(f"Looking for {x} in {q} quadrant ({tri[0]}-{tri[-1]})")
        for b in x == tri:
            if np.all(b):
                return True
            #else:
            #    print("Nothing here")
    return False

def read_palette(palette: dict, img: Image):
    """
    Read in the scanned tile dictionary and interpret it to produce the
    proper component class. Must return a component else raises an error.
    """
    print(f"{ptime()} Reading palette...")
    # as interpreted in .sym_dict.out_1_state.out_to_direction:
    orient_enc = dict(zip(["up","right","down","left"], range(0,4)))
    # if border panel/grid then raise error - indicates segmentation failed
    assert not palette['border_grid'] and not palette['border_panel']
    # if only space is True it's blank 
    if palette['space'] and not np.any([palette[x] for x in \
        [k for k in palette.keys() if k != 'space']]):
        # TODO: make a blank tile class? For now just return ``None``
        return None
    # if terminal on in / terminal off in then terminal class
    elif palette['terminal_on_in'] or palette['terminal_off_in']:
        # use palette['terminal_on_in'] as on bool
        term_on = palette['terminal_on_in']
        term_out_dir = detect_wire_orientation(term_on, img)
        # each terminal only has one output direction:
        assert len(term_out_dir) == 1
        term = terminal(orient_enc[term_out_dir[0]], term_on)
        return term
    # if server in then server class
    elif palette['server_in']:
        serv_out_dir = detect_wire_orientation(True, img)
        # each server only has one output direction:
        if len(serv_out_dir) == 1:
            serv = server(orient_enc[serv_out_dir[0]])
        else:
            serv_out_enc = [orient_enc[x] for x in serv_out_dir]
            serv_out_vec = [x in serv_out_enc for x in range(0,4)]
            assert sum(serv_out_vec) == len(serv_out_dir)
            serv = server(np.array(serv_out_vec))
        return serv
    # if wire on in / wire off in then determine wire class
    elif palette['wire_on_in'] or palette['wire_off_in']:
        # detect wire shape and orientation
        wire_out_dir = detect_wire_orientation(palette['wire_on_in'], img)
        assert 1 < len(wire_out_dir) < 4
        # TODO: get wire type from detect_wire_orientation output for wires
        dir_vec = [orient_enc[x] for x in wire_out_dir]
        if len(wire_out_dir) == 2:
            if np.diff(dir_vec) == 2:
                # straight wire / line wire
                is_horizontal = min(dir_vec) == 1
                return l_wire(is_horizontal, palette['wire_on_in'])
            else:
                # corner wire
                if np.diff(dir_vec) != 1:
                    return c_wire(min(dir_vec), palette['wire_on_in'])
                else:
                    return c_wire(max(dir_vec), palette['wire_on_in'])
        else:
            # must be ``t_wire`` - N.B. ``facing`` has same parity as sum
            facing = [x for x in dir_vec if x % 2 == (sum(dir_vec) % 2)][0]
            return t_wire(facing, palette['wire_on_in'])
    else:
        # if this loop finishes something has gone wrong, raise an error
        raise ValueError("Tile is not blank, but no component detected")

class tile(object):
    """
    A class for the image content and coordinate data of a single tile.
    """
    def __init__(self, image_segment: np.ndarray, xys_xye: list,
                 tile_row: int, tile_n: int):
        self.image = image_segment
        self.palette = scan_tile(self.image)
        self.component = read_palette(self.palette, self.image)
        assert len(xys_xye) == len(xys_xye[0]) == len(xys_xye[1]) == 2
        assert np.all([[type(i) == int for i in j] for j in xys_xye])
        self.xy_coords = xys_xye
        self.row = tile_row
        self.col = tile_n

    def __repr__(self):
        return f"Tile: {self.xy_coords[0]}, {self.xy_coords[1]} " \
             + f"(row {self.row}, column {self.col})."
