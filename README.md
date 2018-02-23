# netwalk

Solver for netwalk, reading in image files of game state [under development]

Example game state:

![](https://raw.githubusercontent.com/lmmx/netwalk/master/data/lgo_netwalk_example_game_state.png)

- terminology for components etc. as in [De Biasi 2012][debiasi12]\*

[debiasi12]: http://www.nearly42.org/vdisk/cstheory/netnpc.pdf "The complexity of the puzzle game Net: rotating wires can drive you crazy"

![Schematic of the NetWalk puzzle and its components, from De Biasi (2012) The complexity of the puzzle game Net: rotating wires can drive you crazy](https://raw.githubusercontent.com/lmmx/shots/master/2018/Feb/de-biasi12_figs1-%2B-2_netwalk-schematic.png)

\* A couple of exceptions:

- N.B. the power unit shown in the figure here [from a paper] has two outputs, whereas the one coded for in this library has only one, in line with the game [at this link](http://www.logicgamesonline.com/netwalk)

- In the component classes I call the connectors 'wires' but elsewhere I call them 'pipes' - it doesn't really matter

## TODO

- [x] write tests for all components
  - [ ] give all wire components `on` states
    - [ ] fix all the tests this breaks
- [x] read in image file and parse tiling/component state
- [ ] write tests for imaging and tiling
- [ ] use relative data path rather than hardcoded
