import os, sys, shutil
from pathlib import Path


def resource_base() -> Path:
    # When frozen by PyInstaller, bundled files are in sys._MEIPASS
    return Path(getattr(sys, "_MEIPASS", Path(__file__).parent)).resolve()

def user_data_root() -> Path:
    # Where we write config/outputs at runtime (writable)
    if sys.platform.startswith("win"):
        root = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData/Local")) / "loom_demo"
    elif sys.platform == "darwin":
        root = Path.home() / "Library/Application Support/loom_demo"
    else:
        root = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local/share")) / "loom_demo"
    root.mkdir(parents=True, exist_ok=True)
    return root

def ensure_seed_files():
    """On first run, copy default config and optional sample data into the writable root."""
    base = resource_base()
    target = user_data_root()

    # Copy config/setup/loom.json if missing
    src_cfg = base / "config" / "setup" / "loom.json"
    dst_cfg = target / "config" / "setup" / "loom.json"
    if src_cfg.exists() and not dst_cfg.exists():
        dst_cfg.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_cfg, dst_cfg)

    # Optional: copy sample patches if you bundle them
    for sub in ["data/patches/upper", "data/patches/lower"]:
        src = base / sub
        dst = target / sub
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)