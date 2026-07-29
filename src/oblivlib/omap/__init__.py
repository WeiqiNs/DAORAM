"""Public API for ``oblivlib.omap``: the OMAP schemes and their construction configs.

The configs are defined in ``oblivlib.dependency.config`` and re-exported here as a
convenience so a scheme and its config can be imported together.
"""

from oblivlib.dependency.config import (
    AvlOmapCachedConfig,
    AvlOmapConfig,
    BPlusOmapCachedConfig,
    BPlusOmapConfig,
    GroupOmapConfig,
    OmapConfig,
    OramOstOmapConfig,
)

from .avl_omap import AVLOmap
from .avl_omap_cache import AVLOmapCached
from .bplus_omap import BPlusOmap
from .bplus_omap_cache import BPlusOmapCached
from .group_omap import GroupOmap
from .oram_ost_omap import OramOstOmap

__all__ = [
    # schemes
    "AVLOmap",
    "AVLOmapCached",
    "BPlusOmap",
    "BPlusOmapCached",
    "GroupOmap",
    "OramOstOmap",
    # configs (re-exported from oblivlib.dependency.config)
    "AvlOmapCachedConfig",
    "AvlOmapConfig",
    "BPlusOmapCachedConfig",
    "BPlusOmapConfig",
    "GroupOmapConfig",
    "OmapConfig",
    "OramOstOmapConfig",
]
