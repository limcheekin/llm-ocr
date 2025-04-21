# llm-ocr: Image-to-Text for PDFs (via EasyOCR or LLM)

This project provides a command-line tool to extract text from PDF documents. It works by converting each page of the PDF into an image and then using either OpenAI's Vision API (like `gpt-4o`) or the EasyOCR library to recognize and extract the text from these images. The extracted text from all pages is then compiled into a single Markdown file.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yQYhcK75j1nSRExbGxQo9MoPr7b42K1W?usp=sharing)

## Features

*   **PDF to Text Conversion:** Processes multi-page PDF files.
*   **Multiple Engines:** Supports text extraction using either:
    *   OpenAI Vision API (requires API key and internet connection).
    *   EasyOCR library (can run offline, supports GPU acceleration).
*   **Image Conversion:** Renders PDF pages to images at a configurable DPI.
*   **Command-Line Interface (CLI):** Easy to integrate into scripts or run from the terminal.
*   **Configurable:** Customize extraction engine, OpenAI model/prompt/detail, EasyOCR languages/GPU usage, and image DPI via command-line arguments.
*   **Markdown Output:** Generates a single Markdown file with clear page separation.
*   **Retry Logic:** Built-in retries for transient OpenAI API errors.

## Requirements

*   Python 3.7 or higher
*   Required Python libraries (listed in `requirements-cpu.txt` or `requirements-gpu.txt`)
*   **For OpenAI engine:** An OpenAI API key (set as `OPENAI_API_KEY` environment variable). Costs are associated with API usage.
*   **For EasyOCR engine:** Depending on your installation choice:
    *   A CUDA-enabled GPU for faster processing (if using `requirements-gpu.txt` and `--gpu`).
    *   Sufficient CPU resources (if using `requirements-cpu.txt` or `--no-gpu`).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/limcheekin/llm-ocr.git
    cd llm-ocr
    ```

2.  **Install dependencies:**
    Choose either the CPU-only or the GPU requirements file.

    *   **For CPU-only:**
        ```bash
        pip install -r requirements-cpu.txt
        ```
    *   **For GPU acceleration (requires CUDA-enabled GPU and drivers):**
        ```bash
        pip install -r requirements-gpu.txt
        ```

3.  **Set up OpenAI API key (if using the OpenAI engine):**
    If you plan to use the OpenAI engine, set your API key as an environment variable:
    ```bash
    export OPENAI_API_KEY='YOUR_API_KEY'
    ```
    *(Replace `YOUR_API_KEY` with your actual key)*

## Usage

The script is run from the command line:

```bash
python main.py <input_pdf> <output_md> [options]
```

*   `<input_pdf>`: Path to the input PDF file. (Required positional argument)
*   `<output_md>`: Path where the output Markdown file will be saved. (Required positional argument)

### Options:

**Engine Selection:**

*   `--engine {openai,easyocr}`: Select the text extraction engine. Defaults to `openai`.

**OpenAI Engine Options (used when `--engine openai`):**

*   `--model MODEL`: OpenAI model to use (e.g., `gpt-4o`). Defaults to `gpt-4o`.
*   `--detail {low,high,auto}`: Image detail level for OpenAI Vision. Defaults to `auto`.
*   `--max-tokens MAX_TOKENS`: Maximum tokens for the OpenAI response per page. Defaults to `4000`.
*   `--prompt PROMPT`: The prompt to send to OpenAI for text extraction. Defaults to `"Extract all readable text from this image. Structure the output clearly, using Markdown formatting where appropriate (e.g., headings, lists)."`.
*   `--prompt-file PROMPT_FILE`: Path to a file containing the OpenAI prompt (overrides `--prompt`).
*   `--delay DELAY`: Delay in seconds between successive OpenAI API calls. Defaults to `1`.

**EasyOCR Engine Options (used when `--engine easyocr`):**

*   `--langs LANGS [LANGS ...]`: List of languages for EasyOCR (e.g., `en fr es`). Defaults to `['en']`.
*   `--gpu / --no-gpu`: Enable or disable GPU usage for EasyOCR. Defaults to `True` (use GPU if available). Use `--no-gpu` to force CPU.

**General Options:**

*   `--dpi DPI`: Resolution (DPI) for rendering PDF pages to images. Defaults to `200`. Higher DPI improves accuracy but increases image size and processing time/cost.
*   `--debug`: Enable debug logging for more verbose output.

### Examples:

1.  **Using OpenAI (default engine) with default options:**
    ```bash
    python main.py input.pdf output.md
    ```

2.  **Using EasyOCR with English and French, forcing CPU:**
    ```bash
    python main.py input.pdf output_easyocr.md --engine easyocr --langs en fr --no-gpu
    ```

3.  **Using OpenAI with a specific model and higher DPI:**
    ```bash
    python main.py document.pdf extracted_text.md --engine openai --model gpt-4-vision-preview --dpi 300
    ```

4.  **Using OpenAI with a prompt from a file:**
    *(First, create a file, e.g., `my_prompt.txt`, with your desired prompt text)*
    ```bash
    python main.py report.pdf report.md --engine openai --prompt-file my_prompt.txt
    ```

## Output Format

The output is a single Markdown file (`.md`). The text from each page is added sequentially. Each page's content is preceded by an `## Page X` heading and separated from the previous page's content by a horizontal rule (`---`).

```markdown
## Page 1

[Text extracted from page 1 goes here]

---

## Page 2

[Text extracted from page 2 goes here]

---

## Page 3

[Text extracted from page 3 goes here]
...
```

## Choosing an Engine

*   **OpenAI:** Generally provides better accuracy, especially for complex layouts or mixed content (text, images, tables), and supports a wide range of languages automatically. Requires an API key and incurs costs per usage. Requires an internet connection.
*   **EasyOCR:** Runs offline after initial model downloads. Can be very fast with GPU acceleration. Free to use (except for hardware costs). Accuracy might vary depending on the font, image quality, and languages compared to state-of-the-art online services. Requires specifying languages upfront.

Choose the engine that best suits your needs regarding accuracy, cost, privacy, and connectivity.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Dependencies

This project relies on the following open-source libraries:

*   [PyMuPDF (fitz)](https://github.com/pymupdf/PyMuPDF) for PDF processing.
*   [Pillow](https://python-pillow.org/) for image manipulation.
*   [openai](https://github.com/openai/openai-python) for interacting with the OpenAI API.
*   [EasyOCR](https://github.com/JaidedAI/EasyOCR) for offline OCR.
*   [OpenCV-Python-Headless](https://github.com/opencv/opencv-python#headless-builds) (a dependency of EasyOCR).
