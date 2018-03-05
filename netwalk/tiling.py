# Classes for handling puzzle tiles
import numpy as np
from imageio.core.util import Image
from itertools import chain
from .colour_dict import game_colours
from .sym_dict import server, terminal, l_wire, c_wire, t_wire
from .util import get_orientation_quadrants, ptime
from .solve_tileset import tileset_solver
from .adjacency import tiling_adjacencies

class tileset(object):
    """
    A class connecting the parsed image file to the internal grid layout
    and component representation, and which initiates a solver.
    """
    def __init__(self, img: Image, tile_mask: np.ndarray):
        print(f"{ptime()} Initialising tileset...")
        global oriented
        oriented = False
        self.segments = segment(tile_mask)
        self.source_image = img
        self.rotated_image = None
        self.tiles = tile_segments(self)
        self.shape = (len(self.tiles), len(self.tiles[0]))
        self.adjacencies = tiling_adjacencies(self)
        self.initialise_tile_adjacencies()
        self._rotation_grid = self.rotation_grid
        self._solved = False
        self.solved_tiles = np.zeros_like(self.tiles, dtype=bool)
        self.solver = None
        self.solve() # perform initial solve
        if not self.solved:
            assert not self.solver is None
            self.solve()
            if not self.solved:
                print("Better luck next time...")
                self.solve()
        print(self.solver)
        if self.solved:
            print("Minimal rotation grid for solved puzzle state:")
            for row in self.rotation_grid:
                r_str = []
                for r in row:
                    if r > -1:
                        r_str.append(' '+str(r))
                    else:
                        r_str.append(str(r))
                print('[ ' + ' | '.join(r_str) + ' ]')

    @property
    def solved(self):
        if not self.solver is None:
            self._solved = self.solver.solved
        return self._solved

    def solve(self):
        """
        Instantiate a solver on the tileset [which will then run].
        """
        assert not self.solved
        if self.solver is None:
            self.solver = tileset_solver(self)
        else:
            self.solver.resolve()
        if self.solved:
            self.populate_rotation_grid()
            self.populate_rotated_image()
        return

    def get_solved_tiles(self) -> np.ndarray:
        """
        Determine which tiles [if any] have been solved and return
        a boolean numpy array accordingly.
        """
        s = np.array([[t.solved for t in r] for r in self.tiles])
        self.solved_tiles = s
        return self.solved_tiles

    def initialise_tile_adjacencies(self):
        """
        Once the ``tiling_adjacencies`` class has a populated ``adjacencies``
        attribute, the ``tileset`` init method calls this function to trigger
        the tile-level updating of all tiles' ``initialise_adjacency`` method,
        such that the ``adjacent_tiles`` attribute of each tile in the tileset
        becomes populated, using the tileset-level ``adjacencies`` attribute.
        """
        for row in self.tiles:
            for tile in row:
                tile.initialise_adjacency()
        for row in self.tiles:
            for tile in row:
                tile.initialise_solve_state()
        return

    def populate_rotation_grid(self):
        self._rotation_grid = self.rotation_grid
        return

    @property
    def rotation_grid(self):
        rot_grid = [[t.rotation for t in row] for row in self.tiles]
        self._rotation_grid = rot_grid
        return self._rotation_grid

    def populate_rotated_image(self):
        self.rotated_image = rotate_image_segments(self)
        return

    def __repr__(self):
        return f"A set of {self.shape[0]}x{self.shape[1]} tiles."

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
    # TODO: is it simpler/faster to replace this with ``sp.ndimage.label``?
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

def rotate_image_segments(tset: tileset) -> Image:
    """
    Using the rotation grid of a [solved] tileset, rotate the corresponding
    segment of (a copy of) the source image per tile to a solved orientation.
    """
    rot_img = tset.source_image.copy() # do not modify source_image attribute
    for (i, row) in enumerate(tset.tiles):
        for (j, t) in enumerate(row):
            (xs, ys), (xe, ye) = tset.segments[i][j]
            r = tset.rotation_grid[i][j]
            r *= -1
            r %= 4
            rot_img[ys:ye+1, xs:xe+1] = np.rot90(t.image, r)
    return rot_img

def rotate_part_solved_image_segments(tset: tileset) -> Image:
    """
    Since an unsolved tileset has no rotation grid, rotate the corresponding
    segment of (a copy of) the source image per tile to a solved orientation
    on the fly, for those which have been solved.
    """
    rot_img = tset.source_image.copy() # do not modify source_image attribute
    for (i, row) in enumerate(tset.tiles):
        for (j, t) in enumerate(row):
            if not t.solved:
                continue
            (xs, ys), (xe, ye) = tset.segments[i][j]
            r = t.rotation
            r *= -1
            r %= 4
            t_img = t.image
            t_bg_ind = np.where(np.all(t_img == [255,255,255], axis=-1))
            t_bg_co = list(zip(t_bg_ind[0], t_bg_ind[1]))
            for coord in t_bg_co:
                t_img[coord] = [0,200,0]
            rot_img[ys:ye+1, xs:xe+1] = np.rot90(t_img, r)
    return rot_img

def tile_segments(tset: tileset) -> list:
    """
    Turn the list of segments returned from ``segment`` into a list of
    rows of ``tile`` objects.
    """
    img, segments = tset.source_image, tset.segments
    assert type(img) == Image and type(segments) == list
    tile_set = []
    for i, seg_row in enumerate(segments):
        #print(f"{ptime()} Scanning row {i}")
        tile_row = []
        for j, seg in enumerate(seg_row):
            #print(f"{ptime()} Scanning row {i} tile {j}")
            (xs, ys), (xe, ye) = seg
            t = tile(tset, img[ys:ye+1, xs:xe+1], seg, i, j)
            tile_row.append(t)
        tile_set.append(tile_row)
    print(f"{ptime()} All tiles scanned")
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
    global oriented
    if not oriented:
        assert img.shape[0] == img.shape[1]
        global orient_dict
        orient_dict = get_orientation_quadrants(img.shape[0])
        assert len(orient_dict) == 4
        oriented = True
    detected_cols = dict(zip(game_colours.keys(), [False] * len(game_colours)))
    if np.all(img == game_colours['space']):
        detected_cols['space'] = True
        return detected_cols
    for label, rgb in game_colours.items():
        if detect_colour(rgb, img):
            detected_cols[label] = True
    return detected_cols

def detect_wire_orientation(on: bool, img: Image):
    """
    Detect the output direction(s) of the wire shown in the tile image.
    """
    if on:
        colour = game_colours['wire_on_in']
    else:
        colour = game_colours['wire_off_in']
    wire_activation = np.all(img == colour, axis=-1)
    # iterate over 4 quadrants, return labels of those containing wire coords
    return detect_quad_members(wire_activation)

def detect_quad_members(member_activations: np.ndarray):
    """
    Detect which quadrants the wire(s) on a given tile are in.
    """
    assert member_activations.dtype == bool
    orientation_vec = []
    global orient_dict
    for i in np.arange(4):
        quad_name = list(orient_dict.keys())[i]
        if detect_quad_member(quad_name, member_activations):
            orientation_vec += [quad_name]
        else:
            pass
    return orientation_vec

def detect_quad_member(quad: str, member_activations: np.ndarray):
    """
    For a single quadrant, detect the presence of at least one coordinate
    in the associated coordinate array, returning as soon as possible.
    """
    assert member_activations.dtype == bool
    global orient_dict
    assert quad in orient_dict.keys()
    quad_activations = np.logical_and(member_activations, orient_dict[quad])
    return np.any(quad_activations)

def read_palette(palette: dict, img: Image):
    """
    Read in the scanned tile dictionary and interpret it to produce the
    proper component class. Must return a component else raises an error.
    """
    # as interpreted in .sym_dict.out_1_state.out_to_direction:
    orient_enc = dict(zip(["up","right","down","left"], np.arange(4)))
    # if border panel/grid then raise error - indicates segmentation failed
    assert not palette['border_grid'] and not palette['border_panel']
    # if only space is True it's blank 
    non_blank_colours = [k for k in palette.keys() if k != 'space']
    if not np.any([palette[x] for x in non_blank_colours]):
        # empty tile - set component to ``None``
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
            serv_out_vec = [x in serv_out_enc for x in np.arange(4)]
            assert sum(serv_out_vec) == len(serv_out_dir)
            serv = server(np.array(serv_out_vec))
        return serv
    # if wire on in / wire off in then determine wire class
    elif palette['wire_on_in'] or palette['wire_off_in']:
        # detect wire shape and orientation
        wire_out_dir = detect_wire_orientation(palette['wire_on_in'], img)
        assert 1 < len(wire_out_dir) < 4
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
    def __init__(self, tset: tileset, image_segment: np.ndarray,
                 xys_xye: list, tile_row: int, tile_n: int):
        self.__parent__ = tset
        self.image = image_segment
        self.fixed = np.zeros(4, dtype=bool)
        self.avoid = np.zeros(4, dtype=bool)
        self.palette = scan_tile(self.image)
        self.component = read_palette(self.palette, self.image)
        self.solved = None
        self._reconfigured = False
        assert len(xys_xye) == len(xys_xye[0]) == len(xys_xye[1]) == 2
        assert np.all([[type(i) == int for i in j] for j in xys_xye])
        self.xy_coords = xys_xye
        self.row = tile_row
        self.col = tile_n
        self._adjacent_tiles = None
        self._rotation = 0

    def initialise_adjacency(self):
        self.adjacent_tiles = self.get_adjacent_tiles()
        return

    def initialise_solve_state(self):
        """
        Technically not initialisation, it's initialised as None. Sets
        ``self.solved`` up as a boolean, and importantly also is now able
        to set ``self.avoid[a]`` for each blank tile ``t`` as well as for
        all ``t_a.avoid[a_inv]`` (i.e. the edge of the inverse direction,
        across the border of the blank tile). This function is called from
        within ``tile.initialise_adjacency`` meaning it can access adjacent
        tiles, which warrants postponing this outside of ``tile.__init__``.
        """
        if self.component is None:
            if self.solved is None:
                self.solve()
            else:
                # should have only been solved via ``set_avoid`` if blank
                assert self.solved
            for (a, t_a) in self.adjacent_tiles.items():
                if t_a.solved:
                    continue
                a_inv = (a + 2) % 4
                t_a.set_avoid([a_inv]) # no edges to fix, only to avoid
        else:
            if self.solved is None:
                self.solved = False
                # I was going to run a solver here, but instead I've moved
                # this into its own module, called after initialisation
        return

    @property
    def rotation(self):
        if self.component is not None:
            self.check_rotation()
        return self._rotation

    @rotation.setter
    def rotation(self, rot):
        self._rotation = rot
        return

    def check_rotation(self):
        """
        For a non-blank tile, if reconfigured, compare configuration to
        the starting configuration (i.e. without rotation), and return
        an integer between -1 and 2 indicating the rotation undergone.
        """
        # don't call this method for a blank tile please!
        assert not self.component is None
        # annoyingly it turns out I can't trust the reconfigured param
        # to tell the truth - if it's found to be reconfigured I'll
        # set it here, but it should have been set by the call to reconfigure
        # if not self.reconfigured:
        #     assert self._rotation == 0
        #     return
        if np.array_equal(self.component.directions, \
                          self.component.start_config):
            if self._rotation != 0:
                # refactor this weird mix of @property and direct attrib use?
                self.rotation = 0
            return
        start_ind = np.where(self.component.start_config)[0]
        ind = np.where(self.component.directions)[0]
        if ind.size == 3:
            # t_wire, so find the rotation from the facing side
            start_facing = np.setdiff1d(np.arange(4), start_ind)[0]
            facing = np.setdiff1d(np.arange(4), ind)[0]
            r = facing - start_facing
            r %= 4
            if r == 3:
                r = -1
            self.rotation = r
            return
        for r in [-1, 1, 2]:
            ind_poss = (start_ind + r) % 4
            if np.array_equal(ind, ind_poss):
                self.rotation = r
                return
        # if that didn't work then use the inverse directions
        start_ind_inv = np.setdiff1d(np.arange(4), start_ind)
        ind_inv = np.setdiff1d(np.arange(4), ind)
        for r in [-1, 1, 2]:
            ind_poss = (start_ind_inv + r) % 4
            if np.array_equal(ind_inv, ind_poss):
                self.rotation = r
                return
        # and if THAT didn't work then it must be r = 2 (and a c_wire)
        assert type(self.component) == c_wire
        self.rotation = 2
        return

    @property
    def adjacent_tiles(self):
        return self._adjacent_tiles

    @adjacent_tiles.setter
    def adjacent_tiles(self, vals: dict):
        assert list(vals.keys()) == list(np.arange(4))
        self._adjacent_tiles = vals
        return

    def fix_connection(self, a: list):
        assert type(a) == list
        assert not np.any(self.avoid[a])
        assert np.all(np.isin(a, np.arange(4)))
        if self.component is None:
            return
        self.component.find_configuration(self.avoid, self.fixed, enforce=a)
        assert np.all(self.fixed[a])
        if not self.solved:
            self.check_solvable() # call nesting begins here
        return

    def set_avoid(self, a: list):
        assert type(a) == list
        assert not np.any(self.fixed[a])
        assert np.all(np.isin(a, np.arange(4)))
        # assert so as not to set_avoid an already fixed direction!
        assert not np.any(np.isin(a, np.where(self.fixed)))
        if np.all(self.avoid[a]):
            # redundant call, this has already been set
            return
        self.avoid[a] = True
        # TODO: return early if the avoid is set? or assert need to set it?
        if not self.component is None:
            self.reconfigure()
        if not self.solved:
            # this enters a subloop to ``find_configuration``* of the tile
            # component (if it got ``reconfigure``d above), calling
            # ``set_avoid`` and ``fix_connection`` (see sym_dict.py)
            self.check_solvable() # call nesting begins here
        return

    @property
    def reconfigured(self):
        return self._reconfigured

    @reconfigured.setter
    def reconfigured(self, is_reconfigured):
        assert type(is_reconfigured) == bool
        self._reconfigured = is_reconfigured
        return

    def reconfigure(self):
        """
        After an update to the ``avoid`` vector, ensure that the tile's
        output directions are not now invalid - and if so, modify them.
        This method calls the component's ``find_configuration`` method,
        to find an allowed configuration [of output directions in use]
        subject to the [tile-level] avoid/fixed constraints, by switching
        the output directions to a valid alternative if 'breached' by an
        update (from e.g. ``set_avoid``). Note that this is done before
        ``check_solvable``, and as such may result in using the only
        remaining degree of freedom (solving the tile), but no attempt is
        made to solve the tile within this method - instead it is left for
        ``check_solvable`` to 'discover'.
        """
        assert not self.solved # don't call this if already marked solved
        if not self.reconfigured:
        #    # annoyingly this doesn't seem to be being set?? so instead I'm
        #    # setting it in the call to rotation_grid once solved... :-/
        #    # also I'm setting start_config on init now, else I can't find
        #    # rotation state accurately, which is more important...
            self.reconfigured = True # first time it's been reconfigured
        #    self.component.start_config = self.component.directions.copy()
        self.component.find_configuration(self.avoid, self.fixed)
        return

    def check_solvable(self):
        """
        After an update to the ``avoid`` or ``fixed`` vector(s),
        check if conditions are met for the tile to be marked solved.
        """
        assert not self.solved # don't call this if already marked solved
        if self.component is None:
            self.solve()
            return
        if self.component.check_solved(self.avoid, self.fixed):
            self.solve()
        return

    def solve(self):
        """
        Declare a tile 'solved', and 'freeze' it on all sides (used edges are
        set to 'fixed', their other side is set to 'fixed' and that tile gets
        updated ('reconfigured') such that its outputs connect up to it;
        likewise, unused edges get set to 'avoid', their other side is set to
        'avoid' and that tile gets updated ('reconfigured') such that its
        outputs do not connect up to it). Note that while reconfiguring
        adjacent tiles in this way, it is possible to constrain them such that
        they become solved, and so this is checked for and done if possible.
        """
        self.solved = True
        if self.component is None:
            for a in np.arange(4):
                a_inv = (a + 2) % 4
                t_a = self.adjacent_tiles[a]
                if not t_a.solved and not a_inv in t_a.avoid:
                        t_a.set_avoid([a_inv])
            return
        ######################################################################
        #################### == FIXING USED DIRECTIONS == ####################
        ######################################################################
        out_fixable = np.where(self.component.directions)[0]
        out_unfixed = np.where(self.fixed == False)[0]
        out_to_fix = np.intersect1d(out_fixable, out_unfixed)
        if out_to_fix.size > 0:
            # N.B. may already be fixed but stll need to fix connections,
            # so even if out_to_fix.size == 0, don't return yet
            self.fixed[out_to_fix] = True
        # now ensure all connections [on adjacent tile/inverse side] get fixed
        out_fixed = np.union1d(out_fixable, out_to_fix)
        # get all used [out] directions (now set to fixed) reconfigure/fix inv
        for a in out_fixed:
            a_inv = (a + 2) % 4
            t_a = self.adjacent_tiles[a]
            if not a_inv in t_a.fixed and not t_a.solved:
                t_a.fix_connection([a_inv])
        ######################################################################
        ################# == AVOIDING UNUSED DIRECTIONS == ###################
        ######################################################################
        # get all unused edge directions, set_avoid t_a[a_inv] and reconfigure
        out_unused = np.where(self.component.directions == False)[0]
        unused_unavoided = np.where(self.avoid == False)[0]
        unused_to_avoid = np.intersect1d(out_unused, unused_unavoided)
        if unused_to_avoid.size > 0:
            # N.B. may be unused but need to set avoid vector explicitly,
            # so even if unused_to_avoid.size == 0, don't return yet
            self.avoid[unused_to_avoid] = True
        # ensure complementary edges [on adjacent tile/inverse side] avoid it
        voided = np.where(self.avoid)[0]
        # get all unused [empty edge] directions, set to avoid
        for a in voided:
            a_inv = (a + 2) % 4
            t_a = self.adjacent_tiles[a]
            if not a_inv in t_a.avoid and not t_a.solved:
                t_a.set_avoid([a_inv])
        return

    def get_adjacent_tiles(self, A=np.arange(4)):
        """
        Default is to get adjacent tiles in all directions. If fewer
        directions are specified, other values returned in the list
        ``adj`` will have ``None`` rather than a ``tile`` object in
        a tuple with the direction integer (i.e. ``(a, t_a)``.
        """
        adj = dict(zip(np.arange(4), [None]*4))
        # TODO: use n to get the tile from __parent__
        for a in A:
            t_a = self.get_adjacent_tile(a)
            adj[a] = t_a
        return adj

    def get_adjacent_tile(self, a: int, n: int = 1):
        """
        Get the ``n``th adjacent tile in the ``a`` direction,
        where ``a`` is a 0-based clockwise integer from top.
        """
        #assert n > 0
        assert 0 <= a < 4
        assert n == 1 # not ready to do multi-step yet
        adjacencies = self.__parent__.adjacencies.adjacencies
        #if n > 1:
        #    while n:
        #        t_a = adjacencies[a][n]
        #        n -= 1
        #    t_a = adjacencies[a][n]
        adjacency_index = (self.row*self.__parent__.shape[0])+self.col
        if a % 3 == 0:
            return adjacencies[adjacency_index].adj[a].interface.pre
        else:
            return adjacencies[adjacency_index].adj[a].interface.post

    def __repr__(self):
        return f"{self.row}:{self.col}"
