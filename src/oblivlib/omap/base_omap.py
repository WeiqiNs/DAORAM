"""The common base class for every OMAP scheme."""

from abc import ABC, abstractmethod
from typing import Any


class BaseOmap(ABC):
    """Public surface shared by both OMAP families: the ODS-tree maps (``OstBaseOmap``) and the
    composed maps (``GroupOmap``, ``OramOstOmap``). Schemes that support it add ``delete`` on top."""

    @abstractmethod
    def init_server_storage(self, data: Any = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def insert(self, key: Any, value: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, key: Any, value: Any = None) -> Any:
        """Return the current value for ``key``, writing ``value`` first when one is given."""
        raise NotImplementedError
