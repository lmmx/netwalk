# netwalk

Solver for netwalk, reading in image files of game state [under development]

- [x] Solver now works for the easier game format, where there are blank tiles on the grid periphery
- [x] Solver now works for the medium game format (for the one game I tried, uncertain if it will solve all)
- [ ] Solver for the expert level is under development...
- [x] Solver can now display the results by rotating the tiles of the input image

Example game states:

![](https://raw.githubusercontent.com/lmmx/netwalk/master/data/lgo_netwalk_example_game_state_easy.png)
![](https://raw.githubusercontent.com/lmmx/netwalk/master/data/lgo_netwalk_example_game_state_easy_2.png)
![](https://raw.githubusercontent.com/lmmx/netwalk/master/data/lgo_netwalk_example_game_state_medium.png)
![](https://raw.githubusercontent.com/lmmx/netwalk/master/data/lgo_netwalk_example_game_state_expert.png)

- terminology for components etc. as in [De Biasi 2012][debiasi12]\*

[debiasi12]: http://www.nearly42.org/vdisk/cstheory/netnpc.pdf "The complexity of the puzzle game Net: rotating wires can drive you crazy"

![Schematic of the NetWalk puzzle and its components, from De Biasi (2012) The complexity of the puzzle game Net: rotating wires can drive you crazy](https://raw.githubusercontent.com/lmmx/shots/master/2018/Feb/de-biasi12_figs1-%2B-2_netwalk-schematic.png)

\* A couple of exceptions:

- N.B. the power unit shown in the figure here [from a paper] has two outputs, whereas the one coded for in this library has from one to three, in line with the game [at this link](http://www.logicgamesonline.com/netwalk)

## Usage

## Interactive

- `python -im netwalk`
- `from .imaging import give_me_the_tiles` (or `show_me_the_solved_tiles` to open a viewer)

## Command line

- `python -c "from netwalk import imaging; imaging.show_me_the_solved_tiles()"`
- (There's no full [expert level] solving functionality yet but will add a proper CLI interface when there is)

## TODO

- [x] write tests for all components
  - [x] give all wire components `on` states
    - [ ] check this hasn't broken any tests
- [ ] use relative data path rather than hardcoded
- [x] read in image file and parse tiling/component state
- [x] detect output directions of component wiring and add to component state
- [ ] write tests for imaging and tiling
- [x] add to the tile initialisation:
  - [x] a 'solved' property per tile
  - [x] a 'fixed' vector per component output direction
  - [x] an 'avoid' vector per tile [maximum length = 4 - number of output directions]
- [ ] add methods that modify these attributes per tile over the tileset
- [x] implement an interface list with methods to access both tiles across an interface
- [x] implement tile level `check_solvable`, `solve`, `check_solved`, `set_fixed`, and `set_avoid` methods
- [x] add a series of solvers for a tile which can be run in succession as applicable
  - just implemented a single class in `solve_tile` in the end with logical checks
- [ ] add further solvers until the expert level can be solved
- [x] calculate and display tile rotations
  - [x] apply the rotations to the input image and display a solved form on screen
