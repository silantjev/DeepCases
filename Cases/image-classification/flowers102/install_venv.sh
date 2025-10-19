#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
command -v python3.11 -V &> /dev/null || sudo apt-get install python3.11 python3.11-venv
package_installed=$(dpkg-query -s python3.11-venv 2> /dev/null |grep -i installed)
[[ -z $package_installed ]] &&  sudo apt-get install python3.11-venv

set -e

if [[ ! -d .venv ]]; then
    python3.11 -m venv ".venv"
fi
source ".venv/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

if [[ -n $1 ]]; then
    echo "Установка PyTorch \"$1\""
    pip install --upgrade --force-reinstall torch torchvision --index-url "https://download.pytorch.org/whl/$1"
    exit 0
fi

if lspci | grep -i nvidia > /dev/null; then
    echo "Установка GPU-версии PyTorch"
    pip install --upgrade torch torchvision
else
    echo "Установка CPU-версии PyTorch"
    pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "Установка зависимостей прошла успешно"
