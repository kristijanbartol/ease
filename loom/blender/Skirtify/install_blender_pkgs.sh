#!/bin/bash
# install_blender_pkgs.sh <pkg_name(s)>

BLENDER_PY="/snap/blender/6514/4.5/python/bin/python3.11"
TARGET="$HOME/Documents/blender-pydeps"

mkdir -p "$TARGET"

"$BLENDER_PY" -m ensurepip --upgrade
"$BLENDER_PY" -m pip install --upgrade pip
"$BLENDER_PY" -m pip install --only-binary=:all: --target "$TARGET" "$@"
