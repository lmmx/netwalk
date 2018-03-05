# read in an image file
from sys import path as syspath
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from imageio.core.util import Image
from .colour_dict import game_colours
from .tiling import tileset, rotate_part_solved_image_segments

def detect_grid_border(img: Image) -> np.ndarray:
    """
    Detect the black game grid border and return it as a boolean
    mask numpy array (of the same size as the input image).
    """
    grid = np.all(img == game_colours['border_grid'], axis=-1)
    return grid

def segment_grid(grid: np.ndarray) -> np.ndarray:
    """
    Using the corners of the grid border, determine the coordinates of tiles
    and then segment these into sets of corner coordinates.
    """
    # ensure text is not in the image (tl/br = top left/bottom right)
    gr_tl, gr_br = list(np.argwhere(grid)[0]), list(np.argwhere(grid)[-1])
    grid_range_fill = grid.copy()
    grid_range_fill[gr_tl[0]:gr_br[0], gr_tl[1]:gr_br[1]] = True
    tile_segments = grid_range_fill - grid
    return tile_segments

def read_game_image():
    """
    Read an image file (.png) into a numpy array in which each entry is
    a row of pixels (i.e. ``len(game_img)`` is the image height in px.
    """
    # data_dir = Path(__file__) / '..' / '..' / 'data'
    data_dir = '/home/louis/rec/netwalk/data/'
    # - imread doesn't accept pathlib PosixPath objects?
    # TODO: have I forgotten to add an 'interpret path' call?
    # game_img = imread(data_dir / 'lgo_netwalk_example_game_easy.png')
    # game_img = imread(data_dir + 'lgo_netwalk_example_game_easy.png')
    # game_img = imread(data_dir + 'lgo_netwalk_example_game_easy_2.png')
    # game_img = imread(data_dir + 'lgo_netwalk_example_game_medium.png')
    game_img = imread(data_dir + 'lgo_netwalk_example_game_expert.png')
    # game_img = imread(data_dir + 'lgo_netwalk_example_game_expert_2.png')
    return game_img

def give_me_the_tiles():
    """
    Debugging/development: produce a tileset
    """
    game_img = read_game_image()
    grid = detect_grid_border(game_img)
    seg = segment_grid(grid)
    tiles = tileset(game_img, seg)
    return tiles

def show_me_the_tiles():
    """
    Debugging/development: produce and display a tileset
    """
    game_img = read_game_image()
    grid = detect_grid_border(game_img)
    seg = segment_grid(grid)
    tiles = tileset(game_img, seg)
    grid_img = game_img.copy()
    seg = np.transpose(seg)
    seg[::2] = False
    seg = np.transpose(seg)
    seg[::2] = False
    grid_img[seg] = [0,200,0]
    plt.imshow(grid_img)
    plt.show()
    return tiles

def show_me_the_solved_tiles():
    """
    Debugging/development: produce and display a solved [rotated] tileset
    """
    game_img = read_game_image()
    grid = detect_grid_border(game_img)
    seg = segment_grid(grid)
    tiles = tileset(game_img, seg)
    if tiles.rotated_image is None:
        print("(Not solved yet, displaying highlighted solved tiles...)")
        presolv = rotate_part_solved_image_segments(tiles)
        plt.imshow(presolv)
        plt.show()
        return tiles
    plt.imshow(tiles.rotated_image)
    plt.show()
    return tiles
