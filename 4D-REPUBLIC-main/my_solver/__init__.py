# The team's solver lives in the 4D-REPUBLIC/ folder (hyphenated, so it is
# loaded by name); this keeps `--solver my_solver` running the same code.
import importlib

solve = importlib.import_module('4D-REPUBLIC').solve
