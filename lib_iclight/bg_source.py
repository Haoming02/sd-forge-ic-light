from enum import Enum
import numpy as np


class BGSourceFC(Enum):
    """BG Source for FC"""

    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    GREY = "Ambient"
    CUSTOM = "Custom LightMap"

    def get_bg(
        self,
        image_width: int,
        image_height: int,
    ) -> np.ndarray:

        match self:

            case BGSourceFC.LEFT:
                gradient = np.linspace(255, 0, image_width)
                image = np.tile(gradient, (image_height, 1))
                input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

            case BGSourceFC.RIGHT:
                gradient = np.linspace(0, 255, image_width)
                image = np.tile(gradient, (image_height, 1))
                input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

            case BGSourceFC.TOP:
                gradient = np.linspace(255, 0, image_height)[:, None]
                image = np.tile(gradient, (1, image_width))
                input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

            case BGSourceFC.BOTTOM:
                gradient = np.linspace(0, 255, image_height)[:, None]
                image = np.tile(gradient, (1, image_width))
                input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

            case BGSourceFC.GREY:
                input_bg = (
                    np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8) + 127
                )

            case _:
                raise SystemError

        return input_bg


class BGSourceFBC(Enum):
    """BG Source for FBC"""

    UPLOAD = "Use Background Image"
    UPLOAD_FLIP = "Use Flipped Background Image"
    # LEFT = "Left Light"
    # RIGHT = "Right Light"
    # TOP = "Top Light"
    # BOTTOM = "Bottom Light"
    # GREY = "Ambient"

    def get_bg(
        self,
        image_width: int,
        image_height: int,
        uploaded_bg: np.ndarray = None,
    ) -> np.ndarray:

        match self:

            case BGSourceFBC.UPLOAD:
                assert uploaded_bg is not None
                input_bg = uploaded_bg

            case BGSourceFBC.UPLOAD_FLIP:
                assert uploaded_bg is not None
                input_bg = np.fliplr(uploaded_bg)

            # case BGSourceFBC.GREY:
            #     input_bg = (
            #         np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8) + 64
            #     )

            # case BGSourceFBC.LEFT:
            #     gradient = np.linspace(224, 32, image_width)
            #     image = np.tile(gradient, (image_height, 1))
            #     input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

            # case BGSourceFBC.RIGHT:
            #     gradient = np.linspace(32, 224, image_width)
            #     image = np.tile(gradient, (image_height, 1))
            #     input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

            # case BGSourceFBC.TOP:
            #     gradient = np.linspace(224, 32, image_height)[:, None]
            #     image = np.tile(gradient, (1, image_width))
            #     input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

            # case BGSourceFBC.BOTTOM:
            #     gradient = np.linspace(32, 224, image_height)[:, None]
            #     image = np.tile(gradient, (1, image_width))
            #     input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)

            case _:
                raise SystemError

        return input_bg
