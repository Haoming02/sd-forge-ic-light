import importlib
from enum import Enum


class BackendType(Enum):
    A1111 = -1
    Forge = 0
    reForge = 1
    Classic = 2


def _import(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def detect_backend() -> BackendType:
    if _import("backend.shared"):
        return BackendType.Forge

    if _import("modules_forge.forge_version"):
        from modules_forge.forge_version import version

        if "1.10.1" in version:
            return BackendType.reForge
        else:
            return BackendType.Classic

    return BackendType.A1111
