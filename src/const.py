from pathlib import Path
from rich.console import Console
from src.utils.logging import Logger
from torch.cuda import is_available as torch_cuda_available
import yaml

with open("config.yaml", "r") as config_file:
    data = yaml.load(config_file, Loader=yaml.SafeLoader)
    paths = data["paths"]

ROOT_DIR = Path(__file__).parent.parent
RANDOM_SEED = data["random_seed"]
DATA_DIR = ROOT_DIR / Path(paths["data"])
MODEL_DIR = ROOT_DIR / Path(paths["model"])
RESULTS_DIR = ROOT_DIR / Path(paths["results"])
DEVICE = "cuda" if torch_cuda_available() else "cpu"
DEBUG = data["debug"]
CONSOLE = Console()
LOGGER = Logger("nlp_pipeline")
RETRAIN_MODEL = data["retrain_model"]
