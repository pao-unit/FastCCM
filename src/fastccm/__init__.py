# src/fastccm/__init__.py
from importlib import import_module
from importlib.metadata import version, PackageNotFoundError
from typing import TYPE_CHECKING
from ._version import __version__

try:
    __version__ = version("FastCCM")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ("__version__", "PairwiseCCM", "Visualizer", "Functions", "data", "utils")

if TYPE_CHECKING:
    from .ccm import PairwiseCCM as _PairwiseCCM
    from .ccm_utils import Visualizer as _Visualizer, Functions as _Functions
    from . import data as data
    from . import utils as utils

def __getattr__(name: str):
    if name == "PairwiseCCM":
        from .ccm import PairwiseCCM
        return PairwiseCCM
    if name == "Visualizer":
        from .ccm_utils import Visualizer
        return Visualizer
    if name == "Functions":
        from .ccm_utils import Functions
        return Functions
    if name in ("data", "utils"):
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module 'fastccm' has no attribute {name!r}")
