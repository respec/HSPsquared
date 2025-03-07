"""The `hsp2tools` module contains supporting software modules such as the code
    to convert legacy WDM and UCI files to HDF5 files for HSP2, and to provide 
    additional new and legacy capabilities.
"""

from .clone import clone, removeClone
from .fetch import fetchtable
from .graph import (
    HDF5_isconnected,
    color_graph,
    component_list,
    graph_from_HDF5,
    make_opsequence,
)
from .readCSV import readCSV
from .readHBN import readHBN
from .readUCI import readUCI
from .readWDM import readWDM
from .restart import restart

from hsp2 import __version__
