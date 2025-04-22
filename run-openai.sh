#!/bin/bash

# Check if the virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Virtual environment already activated."
fi

export OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
export OPENAI_API_KEY=<Your Gemini API Key>

python main.py --engine openai --model gemini-2.0-flash --delay 4 --debug --prompt "Extract all readable text from this image." input.pdf output.md 
