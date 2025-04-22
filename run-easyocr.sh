#!/bin/bash

# Check if the virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Virtual environment already activated."
fi

python main.py --engine easyocr --langs ch_sim --debug input.pdf output.md
