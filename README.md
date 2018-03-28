# netwalk

Solver for netwalk, reading in image files of game state,

## Status - TBC

- Game currently unsolvable at the expert level
- Having exhausted all guidance in the tutorial, I'm open to adding new solver logic if I get suggestions or bright ideas
- Having tried to solve the partly solved game state (see [`data/lgo_netwalk_example_game_expert_part_solved.png`](https://github.com/lmmx/netwalk/blob/master/data/lgo_netwalk_example_game_expert_part_solved.png)) the only route I can see is to see whether taking arbitrary decisions sets off a 'chain reaction' of tile solving (as is ensured to happen at easy and intermediate difficulty levels).
- Having attempted this manually, I've noted that when setting `fix_connection(a)`, it's crucial to supply as many possible values to the `a` direction vector as there are connections possible on a tile (e.g. freezing the `0,0` [corner wire] tile in the above example using `a=[0,3]` will initiate a chain reaction whereas simply `a=[0]` or `a=[3]` will lead to errors, as it effectively puts the game 'out of sync').
- It's not immediately clear however whether this approach will lead to full solutions - however I suspect trying it with the tiles with the maximum number of connections (i.e. the T-wires) bordering an already solved region will be most likely to produce a full solution state...
- This however is dependent on already having a solved region from the initial solve step, which appears to be the minority of generated game states at expert level!

## Development progress

- [x] Solver now works for the easier game format, where there are blank tiles on the grid periphery
- [x] Solver now works for the medium game format (for the one game I tried, uncertain if it will solve all)
- [ ] Solver for the expert level is under development...
- [x] Solver can now display the results by rotating the tiles of the input image

## Example game states

### Beginner

#### Initial state 1:

![](https://raw.githubusercontent.com/lmmx/netwalk/master/data/lgo_netwalk_example_game_easy.png)

#### Solved:

![](https://raw.githubusercontent.com/lmmx/netwalk/master/data/lgo_netwalk_example_game_easy_solved.png)

#### Initial state 2:

![](https://raw.githubusercontent.com/lmmx/netwalk/master/data/lgo_netwalk_example_game_easy_2.png)

#### Solved:

![](https://raw.githubusercontent.com/lmmx/netwalk/master/data/lgo_netwalk_example_game_easy_2_solved.png)

### Medium

#### Initial state:

![](https://raw.githubusercontent.com/lmmx/netwalk/master/data/lgo_netwalk_example_game_medium.png)

#### Solved:

![](https://raw.githubusercontent.com/lmmx/netwalk/master/data/lgo_netwalk_example_game_medium_solved.png)

### Expert

#### Initial state 1:

![](https://raw.githubusercontent.com/lmmx/netwalk/master/data/lgo_netwalk_example_game_expert.png)

#### Part solved:

![](https://raw.githubusercontent.com/lmmx/netwalk/master/data/lgo_netwalk_example_game_expert_part_solved.png)

#### Initial state 2:

![](https://raw.githubusercontent.com/lmmx/netwalk/master/data/lgo_netwalk_example_game_expert_2.png)

#### Not solved:

![](https://raw.githubusercontent.com/lmmx/netwalk/master/data/lgo_netwalk_example_game_expert_2_not_solved.png)

## About

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
- [ ] add a command to automatically save a game solution to the data directory based on its input filename
