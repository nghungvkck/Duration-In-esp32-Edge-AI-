#=========================
# PATH
#=========================
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BASE_PATH = PROJECT_ROOT / "data" / "processed" / "data_lan7_no_smooth"

N_MELS = 128
NOISE_FACTOR = 0.005
SHIFT_MAX = 10
FACTOR_RANGE = (0.8, 1.2)
