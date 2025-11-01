#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
command -v python3.11 -V &> /dev/null || sudo apt-get install python3.11 python3.11-venv
command -v curl -V &> /dev/null || sudo apt-get install curl
package_installed=$(dpkg-query -s python3.11-venv 2> /dev/null |grep -i installed)
[[ -z $package_installed ]] &&  sudo apt-get install python3.11-venv

if ! command -v uv -V &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    if ! command -v uv -V &> /dev/null; then
        echo "Failed to install uv. Using pip"
        ./pip_install_venv.sh
        exit 0
    fi
fi

set -e

uv venv --allow-existing --python /usr/bin/python3.11

source ".venv/bin/activate"

if [[ -f pyproject.toml ]]; then
    echo "Используем pyproject.toml"
    if [[ -n $1 ]]; then
        echo "Установка PyTorch \"$1\""
        uv sync --extra torch-$1
    elif lspci | grep -i nvidia > /dev/null; then
        echo "Установка GPU-версии PyTorch"
        uv sync --extra torch-cu128
    else
        echo "Установка CPU-версии PyTorch"
        uv sync --extra torch-cpu
    fi
else
    echo "Используем uv pip"
    uv pip install -r requirements.txt
    if [[ -n $1 ]]; then
        echo "Установка PyTorch \"$1\""
        if [[ "$1" == gpu ]]; then
            uv pip install --force-reinstall torch==2.9.0
        else
            uv pip install --force-reinstall torch==2.9.0 --index-url "https://download.pytorch.org/whl/$1"
        fi
    elif lspci | grep -i nvidia > /dev/null; then
        echo "Установка GPU-версии PyTorch"
        uv pip install torch==2.9.0
    else
        echo "Установка CPU-версии PyTorch"
        uv pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
    fi
fi

echo "Установка зависимостей прошла успешно"
