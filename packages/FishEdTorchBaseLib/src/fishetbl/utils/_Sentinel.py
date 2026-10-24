from enum import auto, Enum
from typing import override

class _Sentinel(Enum):
    SENTINEL = auto()
    @override
    def __repr__(self) -> str: return "<sentinel>"
    def __str__(self) -> str: return "<sentinel>"
    def __format__(self, _spec: str) -> str: return "<sentinel>"

SENTINEL = _Sentinel.SENTINEL