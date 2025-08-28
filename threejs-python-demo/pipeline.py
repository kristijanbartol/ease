# pipeline.py
import importlib
import runpy

# adjust to your module filename
user_mod = importlib.import_module("loom")

def run_pipeline_once():
    if hasattr(user_mod, "main"):
        user_mod.main()
    else:
        runpy.run_module("loom", run_name="__main__")
    return {"status": "ok", "message": "Pipeline finished successfully"}
