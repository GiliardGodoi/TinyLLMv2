import datasets
import logging
import sys
import transformers
import yaml
from pathlib import Path

# Define the root directory where datasets are stored.
CONFIGS = yaml.safe_load(
    (Path(__file__).resolve().parent.parent / 'config' / "configs.yaml").read_text()
)

logger = logging.getLogger('transformers')

if logger.hasHandlers():
    logger.handlers.clear()

formatter = logging.Formatter(
    fmt="[%(levelname)s|%(name)s|%(funcName)s] %(asctime)s >> %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)

file_handler = logging.FileHandler("debug.log", mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

transformers.utils.logging.set_verbosity_info()
datasets.utils.logging.set_verbosity_info()