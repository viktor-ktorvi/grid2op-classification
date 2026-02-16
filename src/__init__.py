from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
DATA_PATH.mkdir(exist_ok=True, parents=True)
