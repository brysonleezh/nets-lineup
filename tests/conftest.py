import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline" / "ncaa_bridge"))
sys.path.insert(0, str(REPO_ROOT / "src" / "college_model"))
sys.path.insert(0, str(REPO_ROOT / "src" / "translator"))
sys.path.insert(0, str(REPO_ROOT / "src"))
