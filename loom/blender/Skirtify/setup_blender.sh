#!/bin/sh

mkdir -p ~/.config/blender/4.5/scripts/startup
cat > ~/.config/blender/4.5/scripts/startup/site_extra_paths.py <<'PY'
import os, sys
extra = os.path.expanduser('~/Documents/blender-pydeps')
if os.path.isdir(extra) and extra not in sys.path:
    sys.path.insert(0, extra)
PY

