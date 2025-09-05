try:
    from beartype.claw import beartype_package
    from beartype import BeartypeConf

    beartype_package(
        "repeng",
        conf=BeartypeConf(violation_type=UserWarning),
    )
except ImportError:
    pass

from . import control, extract
from .extract import ControlVector, DatasetEntry
from .control import ControlModel

__VERSION__ = extract.__VERSION__

__all__ = ["control", "extract", "ControlVector", "DatasetEntry", "ControlModel", "__VERSION__"]
