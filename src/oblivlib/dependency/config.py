"""Construction-parameter configs for the ORAM and OMAP schemes.

Defaults live here — one place per scheme — instead of being repeated across every constructor
signature. The config hierarchy mirrors the scheme hierarchy: shared defaults are inherited and
scheme-specific defaults override them. All configs are frozen and keyword-only; derive a variant
with `dataclasses.replace(cfg, num_data=...)`.

Note: the internal recursion plumbing of the position-map schemes (`is_pos_map`, `last_oram_data`,
`last_oram_level`) is deliberately NOT here — those are not user knobs. They are passed as
keyword-only internal arguments when a scheme builds its own position-map children.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from oblivlib.dependency.crypto import Encryptor
    from oblivlib.dependency.interact_server import InteractServer


@dataclass(frozen=True, kw_only=True)
class OramConfig:
    """Shared construction parameters for every tree-based ORAM."""

    num_data: int
    data_size: int
    client: InteractServer | None = None
    name: str = "oram"
    filename: str | None = None
    bucket_size: int = 4
    stash_scale: int = 7
    encryptor: Encryptor | None = None

    def __post_init__(self) -> None:
        # Subclasses adding constraints must call super().__post_init__() (dataclasses don't chain it).
        if self.num_data < 1:
            raise ValueError(f"num_data must be >= 1, got {self.num_data}.")
        if self.data_size < 1:
            raise ValueError(f"data_size must be >= 1, got {self.data_size}.")
        if self.bucket_size < 1:
            raise ValueError(f"bucket_size must be >= 1, got {self.bucket_size}.")
        if self.stash_scale < 1:
            raise ValueError(f"stash_scale must be >= 1, got {self.stash_scale}.")


@dataclass(frozen=True, kw_only=True)
class PathOramConfig(OramConfig):
    name: str = "po"


@dataclass(frozen=True, kw_only=True)
class StaticOramConfig(OramConfig):
    name: str = "so"


@dataclass(frozen=True, kw_only=True)
class MulPathOramConfig(OramConfig):
    name: str = "mul_path_oram"
    stash_scale_multiplier: int = 1


@dataclass(frozen=True, kw_only=True)
class RecursiveOramConfig(OramConfig):
    name: str = "rc"
    on_chip_mem: int = 10
    compression_ratio: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.on_chip_mem < 1:
            raise ValueError(f"on_chip_mem must be >= 1, got {self.on_chip_mem}.")
        if self.compression_ratio < 2:
            raise ValueError(f"compression_ratio must be >= 2, got {self.compression_ratio}.")


@dataclass(frozen=True, kw_only=True)
class CounterOramConfig(OramConfig):
    """Shared structure for the counter-based recursive schemes (DA, Freecursive)."""

    num_ic: int = 64
    ic_length: int = 6
    gc_length: int = 64
    prf_key: bytes | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_ic < 1:
            raise ValueError(f"num_ic must be >= 1, got {self.num_ic}.")
        if self.ic_length < 1:
            raise ValueError(f"ic_length must be >= 1, got {self.ic_length}.")
        if self.gc_length < 1:
            raise ValueError(f"gc_length must be >= 1, got {self.gc_length}.")


@dataclass(frozen=True, kw_only=True)
class DaOramConfig(CounterOramConfig):
    name: str = "da"
    on_chip_mem: int = 10
    evict_path_obo: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.on_chip_mem < 1:
            raise ValueError(f"on_chip_mem must be >= 1, got {self.on_chip_mem}.")


@dataclass(frozen=True, kw_only=True)
class FreecursiveOramConfig(CounterOramConfig):
    name: str = "fc"
    num_ic: int = 48
    ic_length: int = 10
    gc_length: int = 32
    on_chip_size: int = 10
    reset_method: Literal["prob", "hard"] = "prob"
    reset_prob: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.on_chip_size < 1:
            raise ValueError(f"on_chip_size must be >= 1, got {self.on_chip_size}.")
        if self.reset_prob is not None and not 0.0 < self.reset_prob <= 1.0:
            raise ValueError(f"reset_prob must be in (0, 1], got {self.reset_prob}.")


@dataclass(frozen=True, kw_only=True)
class OmapConfig(OramConfig):
    """Base config for oblivious maps; adds the key size (kw_only lets it be required here)."""

    name: str = "omap"
    key_size: int
    # When False (default, fully oblivious) insert/search/delete all pad to the worst op's round count
    # so the operation type is hidden. When True, each op uses its own (smaller) bound, revealing the
    # op type but running faster.
    distinguishable: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.key_size < 1:
            raise ValueError(f"key_size must be >= 1, got {self.key_size}.")


@dataclass(frozen=True, kw_only=True)
class AvlOmapConfig(OmapConfig):
    name: str = "avl"


@dataclass(frozen=True, kw_only=True)
class AvlOmapCachedConfig(AvlOmapConfig):
    name: str = "avl_opt"


@dataclass(frozen=True, kw_only=True)
class BPlusOmapConfig(OmapConfig):
    name: str = "bplus"
    order: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.order < 3:
            raise ValueError(f"order must be >= 3, got {self.order}.")


@dataclass(frozen=True, kw_only=True)
class BPlusOmapCachedConfig(BPlusOmapConfig):
    name: str = "bplus_opt"


@dataclass(frozen=True, kw_only=True)
class GroupOmapConfig(OmapConfig):
    name: str = "group_omap"


@dataclass(frozen=True, kw_only=True)
class OramOstOmapConfig:
    """Construction parameters for OramOstOmap (the ODS-over-ORAM framework). The two sub-schemes (the
    ODS and the ORAM, each with its own config) are objects passed to the constructor; only the data
    count lives here -- hence this is a standalone config rather than an OmapConfig subclass."""

    num_data: int

    def __post_init__(self) -> None:
        if self.num_data < 1:
            raise ValueError(f"num_data must be >= 1, got {self.num_data}.")
