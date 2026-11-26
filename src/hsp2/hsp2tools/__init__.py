"""The `hsp2tools` module contains supporting software modules such as the code
to convert legacy WDM and UCI files to HDF5 files for HSP2, and to provide
additional new and legacy capabilities.
"""

__all__ = [
    "HDF5_isconnected",
    "__version__",
    "clone",
    "color_graph",
    "component_list",
    "fetchtable",
    "graph_from_HDF5",
    "make_opsequence",
    "removeClone",
    "restart",
]

# First party imports
from hsp2 import __version__
from hsp2.hsp2tools.clone import clone, removeClone
from hsp2.hsp2tools.fetch import fetchtable
from hsp2.hsp2tools.graph import (
    HDF5_isconnected,
    color_graph,
    component_list,
    graph_from_HDF5,
    make_opsequence,
)

# Local folder imports
from .restart import restart
