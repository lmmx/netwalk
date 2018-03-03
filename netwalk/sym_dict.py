# Define a symbol dictionary
import numpy as np

##########################################################################
#####                       State classes                            #####
##########################################################################

class out_1_state(object):
    """
    A class for the output direction of a single connection, abstracting
    handling of the full 4 directions into an internal property.
    """
    def __init__(self, out:int):
        self.out = out
        self.direction = self.out_to_direction(self.out)
        self.all_dirs = self.out_to_all_dirs(self.out)

    def __repr__(self):
        return self.direction

    def switch_direction(self, to_index):
        """
        Switch the direction of the output connection to the given one
        (used when a clash between a tile's 'avoid' edge list and the
        currently used output direction of a component is detected).
        """
        assert to_index in np.arange(4)
        # update self.out int, then self.direction and self.all_dirs
        self.out = to_index
        self.direction = self.out_to_direction(self.out)
        self.all_dirs = self.out_to_all_dirs(self.out)
        return

    @staticmethod
    def out_to_direction(out: int) -> str:
        dir_dict = dict(zip(np.arange(4), ["up","right","down","left"]))
        if out < 0 or out > 3:
            raise ValueError(f"``out`` must be 0-3 (got {out}).")
        else:
            return dir_dict[out]

    @staticmethod
    def out_to_all_dirs(out: int) -> list:
        all_dirs = np.zeros(4, dtype=bool)
        all_dirs[out] = True
        return all_dirs

class out_2h_state(object):
    """
    A class for the output direction of 2 planar connections, abstracting
    handling of the full 4 directions into an internal property.
    """
    def __init__(self, horizontal: bool):
        self.horizontal = horizontal
        self.direction = self.out_to_direction(self.horizontal)
        self.all_dirs = self.out_to_all_dirs(self.horizontal)

    def __repr__(self):
        return self.direction

    def switch_direction(self):
        self.horizontal = not self.horizontal
        self.direction = self.out_to_direction(self.horizontal)
        self.all_dirs = self.out_to_all_dirs(self.horizontal)
        return

    @staticmethod
    def out_to_direction(out: bool) -> str:
        if out:
            return "left and right"
        else:
            return "up and down"

    @staticmethod
    def out_to_all_dirs(out: bool) -> list:
        # int(out) gives 1 if horizontal, 0 if vertical, so using
        # a step size of 2 pulls out the horizontal/vertical axis
        all_dirs = np.zeros(4, dtype=bool)
        all_dirs[int(out):4:2] = True
        return all_dirs

class out_4_state(object):
    """
    A class for the output direction[s] of a single tile.
    """
    def __init__(self, out: np.ndarray):
        self.out = out
        self.direction = self.out_to_direction(self.out)

    def __repr__(self):
        return self.dirs_to_str(self.direction)

    def switch_directions(self, from_to_pair_list: list):
        """
        Switch the directions of the output connections - note that
        this method leaves the matter of whether the directions are
        valid to the code that calls it (i.e. it can, but should not
        be used to, switch a component to an invalid state, e.g. a
        corner wire must keep output directions 'beside' each other).
        Pass in a list of (from, to) integer pairs [list of 2-tuples]
        (used when a clash between a tile's 'avoid' edge list and the
        currently used output direction of a component is detected).
        """
        # switch the values at the given ``self.out`` array indices
        for (f, t) in from_to_pair_list:
            if not type(f) == type(t) == int:
                f, t = int(f), int(t)
            assert np.all(np.isin((f, t), np.arange(4)))
            self.out[f], self.out[t] = self.out[t], self.out[f]
        self.direction = self.out_to_direction(self.out)
        return

    @staticmethod
    def dirs_to_str(dir_vec: list) -> str:
        if len(dir_vec) == 1:
            return dir_vec[0]
        if len(dir_vec) > 1:
            start_dirs = ', '.join(dir_vec[0:-1])
            if len(dir_vec) > 2:
                start_dirs += ','
            end_dir = dir_vec[-1]
            return f"{' and '.join([start_dirs, end_dir])}"

    @staticmethod
    def out_to_direction(out: np.ndarray) -> list:
        dir_dict = dict(zip(np.arange(4), ["up","right","down","left"]))
        if type(out) != np.ndarray:
            raise TypeError(f"``out`` != ``np.ndarray`` (got {type(out)}.")
        elif out.dtype != bool:
            raise TypeError(f"``out.dtype`` != ``bool`` (got {out.dtype}.")
        elif len(out) != 4:
            raise ValueError(f"``len(out) != 4`` (got length {len(out)}.")
        else:
            return [dir_dict[i] for i, x in enumerate(out) if x]

# this feels like overkill but it's done and gives a different repr to out_2h
# so, use it as a wrapper for 2h_state to give horizontal/vertical repr
class h_state(object):
    """
    A class for the orientation of a single linear wire [class: ``l_wire``].
    """
    def __init__(self, horizontal: bool):
        self.horizontal = out_2h_state(horizontal)

    def switch_direction(self):
        self.horizontal.switch_direction()
        return

    def __repr__(self):
        return 'horizontal' if self.horizontal.horizontal else 'vertical'

class on_state(object):
    """
    A class for the on status of a single node.
    """
    def __init__(self, on: bool):
        self.on = on

    def __repr__(self):
        return 'on' if self.on else 'off'

##########################################################################
#####                      Component classes                         #####
##########################################################################

class server(object):
    def __init__(self, out):
        self.uni = type(out) == np.typeDict['int']
        if self.uni:
            self.out = out_1_state(out)
            self.directions = self.out.all_dirs
        else:
            self.out = out_4_state(out)
            self.direction_list = self.out.direction
            self.directions = self.out.out
        self.state = on_state(True)
        self._start_config = None

    @property
    def start_config(self):
        return self._start_config

    @start_config.setter
    def start_config(self, config):
        assert self._start_config is None
        self._start_config = config
        return

    def find_configuration(self, avoid, fixed, enforce: list = []):
        """
        Any overlapping directions between output and avoid are 'breaches',
        so must be changed in ``self.directions`` subject to any fixed
        directions. If no overlapping [breaching] directions then do nothing.
        Raise an error if the avoid-out overlap [breach] cannot be resolved.
        """
        if len(enforce) > 0:
            assert len(enforce) <= np.sum(self.directions)
            for a in enforce:
                used = np.where(self.directions)[0]
                available = np.intersect1d(np.where(fixed == False), used)
                if not a in used:
                    self.update_out_dir(available[0], a)
                if not fixed[a]:
                    fixed[a] = True
        breach = np.intersect1d(np.where(self.directions), np.where(avoid))
        if breach.size > 0:
            assert breach.size == 1
            unused = np.where(self.directions == False)
            available = np.intersect1d(np.where(avoid == False), unused)
            assert available.size > 0 # no tiles are unsolvable
            # could take a random position, but take the first available
            # ...though must ensure that the one you choose is legal!
            f_dir = np.intersect1d(np.where(fixed), np.where(self.directions))
            if self.uni:
                assert f_dir.size == 0 # can't have a breach AND a fixed dir
            else:
                assert f_dir.size < 3 # can't have a breach AND 3 fixed dirs
                # no constraints other than the available dir not being fixed
                available = np.setdiff1d(available, f_dir)
                assert available.size > 0
            self.update_out_dir(breach[0], available[0])
        return

    def update_out_dir(self, from_index, to_index):
        """
        Update the direction state from an output in the specified
        ``from_index`` direction to move to the ``to_index`` direction,
        where the index is an integer 0-3 (indicating clockwise from top).
        """
        if self.uni:
            # handle out_1_state
            self.out.switch_direction(from_index, to_index)
            self.directions = self.out.all_dirs
        else:
            # handle out_4_state - pass in single pair as list (even if
            # ``self.uni`` is False, can only have 1 avoid direction,
            # thus can only have 1 breach thus only 1 pair to switch)
            self.out.switch_directions([(from_index, to_index)])
            self.direction_list = self.out.direction
            self.directions = self.out.out
        return

    def check_solved(self, avoid: np.ndarray, fixed: np.ndarray):
        # if the fixed directions are the same as the output directions
        # then the outputs were found
        self.validate_edges(fixed, avoid)
        uni = type(self.out.out) == np.typeDict['int'] #self.out==out_1_state?
        out_fixed = np.intersect1d(np.where(self.directions), np.where(fixed))
        target = np.sum(self.directions)
        if out_fixed.size == target:
            # the output direction(s) is(/are) fixed, it's been solved
            return True
        else:
            # ==POSSIBILITY== 1 output, not fixed: only solve if av. all 3
            if uni:
                return np.sum(avoid) == (4 - target) # i.e. 3
        # ==ONLY POSSIBILITY== 3 outputs, 1 or 2 fixed: only solve if avoid 1
        assert target == 3 # not coded for 2 server outputs
        # there are not 3 out_fixed, so can only solve if avoid 1
        assert np.sum(avoid) < 2 # can't av. 2 for a 3-dir. component!
        return np.sum(avoid) == 1

    @staticmethod
    def validate_edges(fixed, avoid):
        # a fixed edge cannot also be an avoid edge
        mutual = np.intersect1d(np.where(fixed), np.where(avoid))
        assert mutual.size == 0
        return

    def __repr__(self):
        return f'A server (always on), with ports pointing {self.out!r}.'

class terminal(object):
    """
    A terminal with a single connection.
    
    - ``out`` indicates the direction of output connection.
    - ``state`` indicates the on status of the terminal.
    """
    def __init__(self, out: int, on: bool):
        self.out = out_1_state(out)
        self.directions = self.out.all_dirs
        self.state = on_state(on)
        self._start_config = None

    @property
    def start_config(self):
        return self._start_config

    @start_config.setter
    def start_config(self, config):
        assert self._start_config is None
        self._start_config = config
        return

    def find_configuration(self, avoid, fixed, enforce: list = []):
        """
        Any overlapping directions between output and avoid are 'breaches',
        so must be changed in ``self.directions`` subject to any fixed
        directions. If no overlapping [breaching] directions then do nothing.
        Raise an error if the avoid-out overlap [breach] cannot be resolved.
        """
        if len(enforce) > 0:
            assert len(enforce) <= np.sum(self.directions)
            for a in enforce:
                used = np.where(self.directions)[0]
                available = np.intersect1d(np.where(fixed == False), used)
                if not a in used:
                    self.update_out_dir(available[0], a)
                if not fixed[a]:
                    fixed[a] = True
        breach = np.intersect1d(np.where(self.directions), np.where(avoid))
        if breach.size > 0:
            assert breach.size == 1
            unused = np.where(self.directions == False)
            available = np.intersect1d(np.where(avoid == False), unused)
            assert available.size > 0 # no tiles are unsolvable
            # could take a random position, but take the first available
            f_dir = np.intersect1d(np.where(fixed), np.where(self.directions))
            assert f_dir.size == 0 # can't have a breach AND a fixed dir
            self.update_out_dir(breach[0], available[0])
        return

    def update_out_dir(self, from_index, to_index):
        """
        Update the direction state from an output in the specified
        ``from_index`` direction to move to the ``to_index`` direction,
        where the index is an integer 0-3 (indicating clockwise from top).
        """
        # handle out_1_state
        assert self.out.out == from_index
        self.out.switch_direction(to_index)
        self.directions = self.out.all_dirs
        return

    def check_solved(self, avoid: np.ndarray, fixed: np.ndarray):
        # if the fixed direction is the same as the output direction
        # then the output was found
        self.validate_edges(fixed, avoid)
        out_fixed = np.intersect1d(np.where(fixed), np.where(self.directions))
        if out_fixed.size > 0:
            # the output direction is fixed, it's been solved
            return True
        else:
            # if the sum of fixed + avoid directions is 3, this means the
            # current out direction is the only option, i.e. should be fixed
            return np.sum(avoid) == 3

    @staticmethod
    def validate_edges(fixed, avoid):
        # a fixed edge cannot also be an avoid edge
        mutual = np.intersect1d(np.where(fixed), np.where(avoid))
        assert mutual.size == 0
        return

    def __repr__(self):
        return f'A terminal pointing {self.out!r}, switched {self.state!r}.'

class l_wire(object):
    """
    A line wire with 2 outputs at opposite sides of the tile.
    
    - ``horizontal`` indicates whether output connections are horizontal.
    """
    def __init__(self, horizontal: bool, on: bool):
        self.horizontal = h_state(horizontal)
        self.directions = self.horizontal.horizontal.all_dirs
        self.state = on_state(on)
        self._start_config = None

    @property
    def start_config(self):
        return self._start_config

    @start_config.setter
    def start_config(self, config):
        assert self._start_config is None
        self._start_config = config
        return

    def find_configuration(self, avoid, fixed, enforce: list = []):
        """
        Any overlapping directions between output and avoid are 'breaches',
        so must be changed in ``self.directions`` subject to any fixed
        directions. If no overlapping [breaching] directions then do nothing.
        Raise an error if the avoid-out overlap [breach] cannot be resolved.
        """
        if len(enforce) > 0:
            assert len(enforce) <= np.sum(self.directions)
            for a in enforce:
                used = np.where(self.directions)[0]
                available = np.intersect1d(np.where(fixed == False), used)
                if not a in used:
                    self.switch_direction()
                if not fixed[a]:
                    fixed[a] = True
        breach = np.intersect1d(np.where(self.directions), np.where(avoid))
        if breach.size > 0:
            assert breach.size <= 2
            unused = np.where(self.directions == False)
            available = np.intersect1d(np.where(avoid == False), unused)
            assert unused[0].size == available.size == 2
            self.switch_direction() # toggle horizontal/vertical
        return

    def switch_direction(self):
        """
        Toggle the direction state, which is either horizontal or vertical.
        """
        # handle h_state
        self.horizontal.switch_direction()
        self.directions = self.horizontal.horizontal.all_dirs
        return

    def is_horizontal(self):
        return self.horizontal.horizontal.horizontal

    def check_solved(self, avoid: np.ndarray, fixed: np.ndarray):
        # if the fixed directions are the same as the output directions
        # then the outputs were found
        return np.any(fixed) or np.any(avoid)

    def __repr__(self):
        return f'A line wire pointing {self.horizontal!r}ly ({self.state!r}).'

class c_wire(object):
    """
    A corner wire with 2 outputs on diagonally adjacent tiles.
    
    - ``corner`` indicates the corner enclosed by the output connections
      - it counts 0 to 3 clockwise from top-left
    """
    def __init__(self, corner: int, on: bool):
        self.corner = c_wire.parse_corner(corner)
        self.out = out_4_state(self.corner)
        self.direction_list = self.out.direction
        self.directions = self.out.out
        self.state = on_state(on)
        self._start_config = None

    @property
    def start_config(self):
        return self._start_config

    @start_config.setter
    def start_config(self, config):
        assert self._start_config is None
        self._start_config = config
        return

    def find_configuration(self, avoid, fixed, enforce: list = []):
        """
        Any overlapping directions between output and avoid are 'breaches',
        so must be changed in ``self.directions`` subject to any fixed
        directions. If no overlapping [breaching] directions then do nothing.
        Raise an error if the avoid-out overlap [breach] cannot be resolved.
        """
        if len(enforce) > 0:
            assert len(enforce) <= np.sum(self.directions)
            for a in enforce:
                used = np.where(self.directions)[0]
                available = np.intersect1d(np.where(fixed == False), used)
                if not a in used:
                    self.update_out_dir([(available[0], a)])
                if not fixed[a]:
                    fixed[a] = True
        breach = np.intersect1d(np.where(self.directions), np.where(avoid))
        if breach.size == 2:
            unused = np.where(self.directions == False)
            available = np.intersect1d(np.where(avoid == False), unused)
            assert available.size == unused.size == 2
            # there is only one option: unused become new out directions
            self.update_out_dir(list(zip(breach, available)))
        elif breach.size == 1:
            unused = np.where(self.directions == False)
            available = np.intersect1d(np.where(avoid == False), unused)
            assert available.size > 0 # no tiles are unsolvable
            # could take a random position, but take the first available
            # ...though must ensure that the one you choose is legal!
            f_dir = np.intersect1d(np.where(fixed), np.where(self.directions))
            assert f_dir.size < 2 # can't have a breach AND 2 fixed out dirs
            if f_dir.size == 1:
                # only those adjacent to the fixed output are valid
                # so exclude the direction facing the fixed direction
                facing = (f_dir + 2) % 4
                available = np.setdiff1d(available, facing)
            self.update_out_dir([(breach[0], available[0])])
        return

    def update_out_dir(self, from_to_pair_list: list):
        """
        Update the direction state from an output in the specified
        ``from_index`` direction to move to the ``to_index`` direction,
        where the index is an integer 0-3 (indicating clockwise from top).
        """
        # handle out_4_state
        self.out.switch_directions(from_to_pair_list)
        self.direction_list = self.out.direction
        self.directions = self.out.out
        return

    def __repr__(self):
        return f'A corner wire pointing {self.out!r} ({self.state!r}).'

    def check_solved(self, avoid: np.ndarray, fixed: np.ndarray):
        # if the fixed directions are the same as the output directions
        # then the outputs were found
        self.validate_edges(fixed, avoid)
        out_dirs = self.out.out
        out_fixed = np.intersect1d(np.where(out_dirs), np.where(fixed))
        if out_fixed.size == 2:
            # the output direction(s) is(/are) fixed, it's been solved
            return True
        if out_fixed.size == 1:
            # the opposite should be avoid but I can't affect it here (can I?)
            out_dir = out_fixed[0]
            avoid_dir = (out_dir + 2) % 4
            avoid[avoid_dir] = True # updates the ``tile.avoid`` attribute
        if np.sum(avoid) == 2:
            return True
        # else still 2-3 degrees of freedom
        return False

    @staticmethod
    def validate_edges(fixed, avoid):
        # at most 2 edges can be avoided
        assert sum(avoid) <= 2
        # a fixed edge cannot also be an avoid edge
        mutual = np.intersect1d(np.where(fixed), np.where(avoid))
        assert mutual.size == 0
        return

    @staticmethod
    def parse_corner(corner: int):
        """
        Convert the corner int representation into a bool 4-tuple.
        """
        on = []
        if corner < 0 or corner > 3:
            raise ValueError(f"``corner`` must be 0-3 (got {corner}).")
        if corner in [0,1]: on += [0]
        if corner in [1,2]: on += [1]
        if corner in [2,3]: on += [2]
        if corner in [3,0]: on += [3]
        out = np.zeros(4, dtype=bool)
        out[on] = True
        return out

class t_wire(object):
    """
    A T-shaped wire with 3 outputs.
    
    - ``facing`` indicates the central output
      - it counts 0 to 3 clockwise from top
    - ``out`` indicates the output directions [explicitly]
      - it stores a clockwise Boolean 4-tuple, top = 0th
    """
    def __init__(self, facing: int, on: bool):
        self.facing = t_wire.parse_facing(facing)
        self.out = out_4_state(self.facing)
        self.direction_list = self.out.direction
        self.directions = self.out.out
        self.state = on_state(on)
        self._start_config = None

    @property
    def start_config(self):
        return self._start_config

    @start_config.setter
    def start_config(self, config):
        assert self._start_config is None
        self._start_config = config
        return

    def find_configuration(self, avoid, fixed, enforce: list = []):
        """
        Any overlapping directions between output and avoid are 'breaches',
        so must be changed in ``self.directions`` subject to any fixed
        directions. If no overlapping [breaching] directions then do nothing.
        Raise an error if the avoid-out overlap [breach] cannot be resolved.
        """
        if len(enforce) > 0:
            assert len(enforce) <= np.sum(self.directions)
            for a in enforce:
                used = np.where(self.directions)[0]
                available = np.intersect1d(np.where(fixed == False), used)
                if not a in used:
                    self.update_out_dir(available[0], a)
                if not fixed[a]:
                    fixed[a] = True
        breach = np.intersect1d(np.where(self.directions), np.where(avoid))
        if breach.size > 0:
            unused = np.where(self.directions == False)
            available = np.intersect1d(np.where(avoid == False), unused)
            assert available.size == breach.size == 1 # only one option
            self.update_out_dir(breach[0], available[0])
        return

    def update_out_dir(self, from_index, to_index):
        """
        Update the direction state from an output in the specified
        ``from_index`` direction to move to the ``to_index`` direction,
        where the index is an integer 0-3 (indicating clockwise from top).
        """
        # handle out_4_state - pass in single pair as list (can only
        # have 1 avoid direction, thus 1 breach, 1 pair to switch)
        self.out.switch_directions([(from_index, to_index)])
        self.direction_list = self.out.direction
        self.directions = self.out.out
        self.facing = type(self).out_to_facing(self.directions)
        return

    def __repr__(self):
        return f'A T wire pointing {self.out!r} ({self.state!r}).'

    def check_solved(self, avoid: np.ndarray, fixed: np.ndarray):
        # if the fixed directions are the same as the output directions
        # then the outputs were found
        self.validate_edges(fixed, avoid)
        out_dirs = self.out.out
        out_fixed = np.intersect1d(np.where(out_dirs), np.where(fixed))
        return (out_fixed.size == 3 or np.any(avoid))

    @staticmethod
    def validate_edges(fixed, avoid):
        # at most 1 edge can be avoided
        assert sum(avoid) <= 1
        # a fixed edge cannot also be an avoid edge
        mutual = np.intersect1d(np.where(fixed), np.where(avoid))
        assert mutual.size == 0
        return

    @staticmethod
    def parse_facing(facing: int):
        """
        Convert the facing int representation into a bool 4-tuple.
        """
        if facing < 0 or facing > 3:
            raise ValueError(f"``facing`` must be 0-3 (got {facing}).")
        excl = (facing + 2) % 4
        out = np.ones(4, dtype=bool)
        out[excl] = False
        return out

    @staticmethod
    def out_to_facing(out: np.ndarray):
        """
        Convert a bool 4-tuple into the facing int representation.
        """
        facing = np.where(out == False)[0]
        assert facing.size == 1
        return int(facing[0]) # turn from numpy int to regular int
