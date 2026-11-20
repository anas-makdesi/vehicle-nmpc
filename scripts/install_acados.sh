#!/usr/bin/env bash
# Build acados from source at a pinned commit and install its Python interface.
#   scripts/install_acados.sh [install_dir]        (default: ./acados)
# Afterwards: export ACADOS_SOURCE_DIR=<install_dir>
#             export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<install_dir>/lib
set -euo pipefail
ACADOS_COMMIT=1d682b2deed23d0113b3356709c9e48341a3d90e  # 13 Nov 2026, tested
DIR=$(realpath -m "${1:-acados}")
PY=${PYTHON:-python3}
if [ ! -f "$DIR/lib/libacados.so" ]; then
  rm -rf "$DIR" && mkdir -p "$DIR" && cd "$DIR"
  git init -q && git remote add origin https://github.com/acados/acados.git
  git fetch -q --depth 1 origin "$ACADOS_COMMIT" && git checkout -q FETCH_HEAD
  git submodule update -q --init --recursive --depth 1
  mkdir -p build && cd build
  cmake -DACADOS_WITH_QPOASES=ON -DACADOS_WITH_OPENMP=OFF .. > /dev/null
  make install -j"$(nproc)" > /dev/null
fi
$PY -m pip install ${PIP_FLAGS:-} -q -e "$DIR/interfaces/acados_template"
# the code generator's template renderer
ACADOS_SOURCE_DIR="$DIR" $PY -c "from acados_template.utils import get_tera; get_tera(force_download=False)" > /dev/null
echo "acados ready in $DIR"
