"""Payload compression plugin system.

Compression is transparent to the delta encoder and the hash chain.
The compressor wraps raw payload bytes; the hash is computed over the
wrapped bytes. This makes compression a storage-layer concern, not a
protocol-layer concern.
"""

import zlib
from typing import Dict, Optional


class Compressor:
    """Base compressor interface."""

    name: str = "null"

    def compress(self, data: bytes) -> bytes:
        raise NotImplementedError

    def decompress(self, data: bytes) -> bytes:
        raise NotImplementedError


class NullCompressor(Compressor):
    """No-op compressor — passes bytes through unchanged."""

    name = None

    def compress(self, data: bytes) -> bytes:
        return data

    def decompress(self, data: bytes) -> bytes:
        return data


class ZlibCompressor(Compressor):
    """zlib compression (standard library)."""

    name = "zlib"

    def __init__(self, level: int = 6):
        self.level = level

    def compress(self, data: bytes) -> bytes:
        return zlib.compress(data, self.level)

    def decompress(self, data: bytes) -> bytes:
        return zlib.decompress(data)


# Registry of available compressors.
# To add a new algorithm, implement Compressor and register here.
_REGISTRY: Dict[Optional[str], Compressor] = {
    None: NullCompressor(),
    "none": NullCompressor(),
    "zlib": ZlibCompressor(),
}


def get_compressor(name: Optional[str]) -> Compressor:
    """Look up a compressor by name.

    Args:
        name: Compressor name (e.g. 'zlib') or None for no compression.

    Returns:
        Compressor instance.

    Raises:
        ValueError: If the named compressor is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(
            repr(k) for k in _REGISTRY if k is not None
        ) or "none"
        raise ValueError(
            f"Unknown compression algorithm: {name!r}. "
            f"Available: {available}."
        )
    return _REGISTRY[name]


def compress_payload(data: bytes, name: Optional[str]) -> bytes:
    """Compress raw payload bytes."""
    return get_compressor(name).compress(data)


def decompress_payload(data: bytes, name: Optional[str]) -> bytes:
    """Decompress stored payload bytes."""
    return get_compressor(name).decompress(data)
