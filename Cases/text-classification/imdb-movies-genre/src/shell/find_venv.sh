cd "$SCRIPT_DIR"

if [[ -z "$VIRTUAL_ENV" ]]; then
    VIRTUAL_ENV=".venv"
    while [[ $(pwd) != "/" && ! -d "$VIRTUAL_ENV" ]]; do
        cd ..
    done
    if [[ -d "$VIRTUAL_ENV" ]];  then
        source "$VIRTUAL_ENV/bin/activate"
        # Variable VIRTUAL_ENV substituted to absolute path
        echo [Info]: Virtual environment used: $VIRTUAL_ENV >&2
    else
        echo [Warning]: directory $VIRTUAL_ENV not found. Install virtual environment >&2
    fi
fi

cd "$SCRIPT_DIR"/src
