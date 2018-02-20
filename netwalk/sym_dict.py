# Define a symbol dictionary
import numpy as np

class out_1_state(object):
    """
    A class for the output direction of a single connection, abstracting
    handling of the full 4 directions into an internal property.
    """
    def __init__(self, out:int):
        self.out = out
        self.direction = out_1_state.out_to_direction(self.out)
        self.all_dirs = out_1_state.out_to_all_dirs(self.out)

    def __repr__(self):
        return self.direction

    @staticmethod
    def out_to_direction(out: int) -> str:
        dir_dict = dict(zip(range(0,4), ["up","right","down","left"]))
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
        self.direction = out_2h_state.out_to_direction(self.horizontal)
        self.all_dirs = out_2h_state.out_to_all_dirs(self.horizontal)

    def __repr__(self):
        return self.direction

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
        self.direction = out_4_state.out_to_direction(self.out)

    def __repr__(self):
        return out_4_state.dirs_to_str(self.direction)

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
        dir_dict = dict(zip(range(0,4), ["up","right","down","left"]))
        if out.dtype != np.dtype('bool'):
            raise TypeError(f"``out`` != ``np.ndarray`` (got {type(out)}.")
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

    def __repr__(self):
        return f'A terminal pointing {self.out!r}, {self.state!r}.'

# the h_state and out_2h_state feels bad... bleh TODO: refactor, maybe
class l_wire(object):
    """
    A line wire with 2 outputs at opposite sides of the tile.
    
    - ``horizontal`` indicates whether output connections are horizontal.
    """
    def __init__(self, horizontal: bool):
        self.horizontal = h_state(horizontal)
        self.directions = self.horizontal.horizontal.all_dirs

    def __repr__(self):
        return f'A line wire pointing {self.horizontal!r}ly.'

class c_wire(object):
    """
    A corner wire with 2 outputs on diagonally adjacent tiles.
    
    - ``corner`` indicates the corner enclosed by the output connections
      - it counts 0 to 3 clockwise from top-left
    """
    def __init__(self, corner: int):
        self.corner = c_wire.parse_corner(corner)
        self.out = out_4_state(self.corner)
        self.directions = self.out.direction

    def __repr__(self):
        return f'A corner wire pointing {self.out!r}.'

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
    def __init__(self, facing: int):
        self.facing = t_wire.parse_facing(facing)
        self.out = out_4_state(self.facing)
        self.directions = self.out.direction

    def __repr__(self):
        return f'A T-shaped wire pointing {self.out!r}.'

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

class server(object):
    def __init__(self):
        # TODO
        return
