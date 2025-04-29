from .logging import logger


class ICModels:
    _init: bool = False

    fc: str = ""
    fbc: str = ""

    fc_path: str = None
    fbc_path: str = None

    @classmethod
    def detect_models(cls):
        if cls._init:
            return
        else:
            cls._init = True

        import os

        from modules.paths import models_path

        folder = os.path.join(models_path, "ic-light")
        os.makedirs(folder, exist_ok=True)

        fc, fbc = None, None

        for obj in os.listdir(folder):
            if not obj.endswith(".safetensors"):
                continue
            if "fc" in obj.lower():
                fc = os.path.join(folder, obj)
            if "fbc" in obj.lower():
                fbc = os.path.join(folder, obj)

        if fc is None or fbc is None:
            logger.error("Failed to locate IC-Light models! Download from Releases!")
            return

        cls.fc: str = os.path.basename(fc).rsplit(".", 1)[0]
        cls.fc_path: str = fc

        cls.fbc: str = os.path.basename(fbc).rsplit(".", 1)[0]
        cls.fbc_path: str = fbc

    @classmethod
    def get_path(cls, model: str) -> str:
        match model:
            case cls.fc:
                return cls.fc_path
            case cls.fbc:
                return cls.fbc_path
            case _:
                raise ValueError
