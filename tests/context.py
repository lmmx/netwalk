from sys import path as syspath
from pathlib import Path

inf = Path(__file__) / '..' / '..' / 'netwalk'
syspath.insert(1, inf.resolve())

import netwalk
