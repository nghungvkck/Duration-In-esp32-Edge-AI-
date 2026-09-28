#=========================
# PATH
#=========================
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BASE_PATH = PROJECT_ROOT / "data" / "processed" / "data_lan7_no_smooth"

N_MELS = 128