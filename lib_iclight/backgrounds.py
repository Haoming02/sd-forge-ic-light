from enum import Enum

import numpy as np


class BackgroundFC(Enum):
    """Background Source for FC Models"""

    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    GREY = "Ambient"
    CUSTOM = "Custom"

    def get_bg(self, width: int = 512, height: int = 512) -> np.ndarray:
        match self:
            case BackgroundFC.LEFT:
                gradient = np.linspace(255, 0, width)
                image = np.tile(gradient, (height, 1))
                input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

            case BackgroundFC.RIGHT:
                gradient = np.linspace(0, 255, width)
                image = np.tile(gradient, (height, 1))
                input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

            case BackgroundFC.TOP:
                gradient = np.linspace(255, 0, height)[:, None]
                image = np.tile(gradient, (1, width))
                input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

            case BackgroundFC.BOTTOM:
                gradient = np.linspace(0, 255, height)[:, None]
                image = np.tile(gradient, (1, width))
                input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

            case BackgroundFC.GREY:
                input_bg = np.zeros((height, width, 3), dtype=np.uint8) + 127

            case BackgroundFC.CUSTOM:
                return None

        return input_bg
