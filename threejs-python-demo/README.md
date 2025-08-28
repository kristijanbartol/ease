# threejs-python-demo

A minimal, double-clickable demo that:
- Lets you paste a `loom.json`-compatible config in the browser
- Runs your Python pipeline (which calls pybind + a subprocessed C++ optimizer)
- Visualizes the generated garment meshes (PLY/OBJ/GLTF) using three.js

## Quick Start
1. (Optional) Create and populate a `venv` with `pip install -r requirements.txt` plus your own deps.
2. Double-click `start.bat` (Windows) or `start.command` (macOS/Linux).
3. Your browser opens at `http://127.0.0.1:8000/`. Paste your config JSON and click **Run**.

### Notes
- The wrapper writes your config to `config/setup/loom.json` so your existing code runs unchanged.
- Combined per-part meshes are saved into `outputs/latest/garment_upper.ply` and `garment_lower.ply` for visualization. Adjust `pipeline.py` if your outputs differ.
- If your optimizer produces additional meshes, copy/symlink them into `outputs/latest/` so the viewer can load them.

### Troubleshooting
- Ensure your pybind module and the precompiled C++ binary are discoverable on `PATH`/`PYTHONPATH` or referenced via absolute paths in your script.
- If you need environment variables (e.g., `LD_LIBRARY_PATH`), set them inside the start script(s).
- If you prefer a single-binary app, consider packaging with PyInstaller or bundling this inside an Electron shell later.
```

---

## How to integrate your current script
1) Rename your current single-file script to `loom.py`.
2) Expose the functions used above (`prepare_body_meshes`, `process_config`, `prepare_ref`, `export_*`, `create_latest_dir`, `run_optimization`, and a `GENDER` const). If some are not available, import/adjust accordingly in `pipeline.py`.
3) If your script already writes combined meshes for each part, remove `_maybe_export_garment_mesh` or point the frontend to your files.
4) If you need OBJ/GLTF support, place the corresponding three.js loaders and add a tiny switch in `main.js`.