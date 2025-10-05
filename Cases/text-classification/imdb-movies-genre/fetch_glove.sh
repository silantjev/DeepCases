#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/data"


URL="https://huggingface.co/spaces/Vrk/SkimLit/resolve/main/glove.6B.300d.txt"
DATASET=${URL##*/}
SITE=${URL##*://}
SITE=${SITE%%/*}
mkdir -p "$OUTPUT_DIR"
cd $OUTPUT_DIR

echo "Скачивание датасета: $DATASET c сайта $SITE по адресу $URL..."
wget "$URL"

if [ $? -eq 0 ]; then
    echo "[✓] Датет скачан успешно в: $OUTPUT_DIR"
else
    echo "[✗] Ошибка при скачивании"
    exit 1
fi

echo "Арховируем для экономии места на диске"
zip glove.6B.300d.txt.zip glove.6B.300d.txt
rm glove.6B.300d.txt
