#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -z "$VIRTUAL_ENV" ]]; then
    VIRTUAL_ENV="$SCRIPT_DIR"/.venv
    if [[ -d "$VIRTUAL_ENV" ]];  then
        source "$VIRTUAL_ENV/bin/activate"
    else
        echo Warning: directory $VIRTUAL_ENV not found. Install virtual environment >&2
    fi
fi

python3 src/split_data.py "$@"
