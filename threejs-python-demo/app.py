# app.py
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
from pipeline import run_pipeline_once

from paths import resource_base, user_data_root, ensure_seed_files
ensure_seed_files()

ROOT = resource_base()                 # read-only bundled assets
STATIC_DIR = ROOT / "static"           # served from bundle
RUNTIME = user_data_root()             # writable

# The config the pipeline should read/write at runtime:
CONFIG_PATH = RUNTIME / "config" / "setup" / "loom.json"

# Where you look for patches at runtime:
GARMENTS_DIR_UPPER = RUNTIME / "data" / "patches" / "upper"
GARMENTS_DIR_LOWER = RUNTIME / "data" / "patches" / "lower"

# Body mesh location (runtime or bundled—your call):
BODY_MESH_PATH = RUNTIME / "outputs" / "latest" / "body.ply"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

'''
ROOT = Path(__file__).parent.resolve()
STATIC_DIR = ROOT / "static"

BODY_MESH_PATH = Path("data/meshes/body.ply")
GARMENTS_DIR_UPPER = Path("data/patches/upper")
GARMENTS_DIR_LOWER = Path("data/patches/lower")
CONFIG_PATH = Path("config/setup/loom_collapsed.json")
'''

RANGES = {
    "mid": (0.0, 1.0),
    "neck": (0.0, 1.0),
    "shoulder": (0.0, 1.0),
    "upper_side": (0.0, 1.0),
    "lower_side": (0.0, 1.0),
    "between": (0.0, 1.0),
    "sleeve": (0.1, 0.35),
    "upper_bottom": (0.15, 0.35),
    "lower_bottom": (0.4, 0.9),
    "upper_scale": (0.8, 2.0),
    "lower_scale": (0.8, 2.0),
}


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/config")
def get_config():
    if not CONFIG_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Config not found: {CONFIG_PATH}")
    try:
        return JSONResponse(json.loads(CONFIG_PATH.read_text()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON: {e}")
    
@app.put("/config")
async def put_config(request: Request):
    """Overwrite config/setup/loom.json with provided values (clamped to ranges)."""
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Body must be a JSON object.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad JSON: {e}")

    # Start from current config (if any), then apply provided keys
    current = {}
    if CONFIG_PATH.exists():
        try:
            current = json.loads(CONFIG_PATH.read_text())
        except Exception:
            current = {}

    updated = dict(current)
    for k, v in payload.items():
        if k not in RANGES:
            continue  # ignore unknown keys
        try:
            v = float(v)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Non-numeric value for '{k}'")
        lo, hi = RANGES[k]
        v = max(lo, min(hi, v))
        updated[k] = v

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(updated, indent=2))
    return JSONResponse({"status": "ok", "config": updated})

@app.get("/file")
async def download_file(path: str):
    p = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {p}")
    return FileResponse(p)

@app.get("/garments")
async def garments():
    """List all patch PLYs for upper/lower as relative paths the frontend can load."""
    def list_paths(base: Path):
        out = []
        if base.exists():
            for p in base.rglob("ref.ply"):  # matches .../patch_XX/ref.ply
                try:
                    out.append(str(p.resolve().relative_to(ROOT)).replace("\\", "/"))
                except Exception:
                    # If not under ROOT, fall back to absolute path (served via /file still ok)
                    out.append(str(p.resolve()).replace("\\", "/"))
        return sorted(out)

    data = {
        "upper": list_paths(GARMENTS_DIR_UPPER),
        "lower": list_paths(GARMENTS_DIR_LOWER),
    }
    if not data["upper"] and not data["lower"]:
        # Not fatal, but helpful feedback
        print("No garment patches found under data/patches/{upper|lower}/.../ref.ply")
    return JSONResponse(data)


@app.get("/body")
async def body():
    p = BODY_MESH_PATH.resolve()
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Body mesh not found: {p}")
    print(f"Serving body mesh: {p}")
    return FileResponse(p)


@app.post("/run")
def run():
    print("Received /run")
    try:
        result = run_pipeline_once()
        return JSONResponse(result)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(status_code=500, detail=f"{e}\n{tb}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
