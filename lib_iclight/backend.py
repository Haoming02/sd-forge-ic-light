from enum import Enum
from typing import Callable


class BackendType(Enum):
    A1111 = -1
    Forge = 0
    reForge = 1
    Classic = 2


def detect_backend() -> tuple["BackendType", Callable]:
    try:
        from .backends.forge import apply_ic_light

        return (BackendType.Forge, apply_ic_light)

    except ImportError:
        pass

    try:
        from modules_forge import forge_version

        from .backends.classic import apply_ic_light

        if "1.10.1" not in forge_version.version:
            return (BackendType.Classic, apply_ic_light)

        from lib_iclight.patch_weight import patch_channels

        patch_channels()

        return (BackendType.reForge, apply_ic_light)

    except ImportError:
        pass

    from .backends.a1111 import apply_ic_light

    return (BackendType.A1111, apply_ic_light)
