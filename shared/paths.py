"""Repository-relative locations shared by evaluation stages."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT

# Source checkouts keep the public configuration beside run.py. Wheels install
# the same files under a namespaced resource package (not site-packages/config).
CONFIG_ROOT = REPO_ROOT / "config"
if not (REPO_ROOT / "pyproject.toml").is_file():
    import vahbench_config
    CONFIG_ROOT = Path(vahbench_config.__file__).resolve().parent
