import logging
import sys


class ColorCode:
    RESET = "\033[0m"
    BLACK = "\033[0;90m"
    CYAN = "\033[0;36m"
    YELLOW = "\033[0;33m"
    RED = "\033[0;31m"

    MAP = {
        "DEBUG": BLACK,
        "INFO": CYAN,
        "WARNING": YELLOW,
        "ERROR": RED,
    }


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        levelname = record.levelname
        record.levelname = f"{ColorCode.MAP[levelname]}{levelname}{ColorCode.RESET}"
        return super().format(record)


logger = logging.getLogger("IC-Light")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter("[%(name)s] %(levelname)s - %(message)s"))
    logger.addHandler(handler)
