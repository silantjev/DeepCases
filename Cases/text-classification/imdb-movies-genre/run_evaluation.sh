#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR"/src/shell/find_venv.sh

python3 ./evaluation.py "$@"
cd ..
