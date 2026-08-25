#!/bin/sh
# Build the VerfCNN weight dumper.
#
# Expects the VerfCNN checkout (https://github.com/quwenjie/VerfCNN) as a
# sibling of the workspace root, i.e. ../../../VerfCNN relative to this script,
# which is where dump_weights.cpp looks for convnet_params.h.
#
# convnet_params.cpp is ~50 MB of array initializers; -O0 keeps the compile
# under a minute.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
verfcnn=${VERFCNN_DIR:-"$here/../../../VerfCNN"}

if [ ! -f "$verfcnn/convnet_params.cpp" ]; then
    echo "VerfCNN sources not found at $verfcnn" >&2
    echo "clone it there, or set VERFCNN_DIR" >&2
    exit 1
fi

"${CXX:-g++}" -O0 -o "$here/dump_weights" \
    "$here/dump_weights.cpp" "$verfcnn/convnet_params.cpp"

echo "built $here/dump_weights"
