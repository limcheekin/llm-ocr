import fitz  # PyMuPDF
import io
from PIL import Image
import base64
import openai
import os
import time
import argparse # For command-line arguments
import logging # For better logging
from typing import List, Optional, Tuple, Dict, Any

# --- Try importing easyocr and set a flag ---
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None # Define easyocr as None if not available

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
DEFAULT_ENGINE = "openai"
DEFAULT_MODEL = "gpt-4o"
DEFAULT_DPI = 200
DEFAULT_DELAY_SECONDS = 1 # Delay only relevant for OpenAI API calls
DEFAULT_MAX_TOKENS = 4000
DEFAULT_DETAIL_LEVEL = "auto" # options: low, high, auto
RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5
DEFAULT_EASYOCR_LANGS = ['en'] # Default languages for easyocr

# --- OpenAI Client Initialization ---
# Initialize OpenAI client only if needed later, handling potential errors.
openai_client: Optional[openai.OpenAI] = None

# --- EasyOCR Reader Initialization ---
# Initialize easyocr Reader only if needed later.
ocr_reader: Optional[easyocr.Reader] = None


def initialize_openai_client() -> Optional[openai.OpenAI]:
    """Initializes and returns the OpenAI client. Returns None on failure."""
    global openai_client
    if openai_client is None:
        try:
            openai_client = openai.OpenAI()
            # Optional: Test connection, e.g., list models (can add cost/time)
            # openai_client.models.list()
            logging.info("OpenAI client initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client: {e}. Ensure OPENAI_API_KEY is set.")
            openai_client = None # Ensure it's None on failure
    return openai_client

def initialize_easyocr_reader(languages: List[str], use_gpu: bool) -> Optional[easyocr.Reader]:
    """Initializes and returns the easyocr Reader. Returns None on failure."""
    global ocr_reader
    if not EASYOCR_AVAILABLE:
        logging.error("easyocr library is not installed. Cannot use easyocr engine.")
        return None
    if ocr_reader is None:
        try:
            logging.info(f"Initializing easyocr Reader for languages: {languages} (GPU: {use_gpu})...")
            # Explicitly check availability before attempting to use easyocr
            if easyocr:
                 ocr_reader = easyocr.Reader(languages, gpu=use_gpu)
                 logging.info("easyocr Reader initialized successfully.")
            else:
                 # This case should theoretically be caught by EASYOCR_AVAILABLE, but belt-and-suspenders
                 logging.error("easyocr module was not imported successfully earlier.")
                 ocr_reader = None

        except Exception as e:
            logging.error(f"Failed to initialize easyocr Reader: {e}. Ensure easyocr and dependencies (PyTorch) are installed correctly.")
            ocr_reader = None # Ensure it's None on failure
    return ocr_reader


def convert_pdf_to_images(pdf_path: str, dpi: int = DEFAULT_DPI) -> List[bytes]:
    """
    Converts each page of a PDF file into a list of PNG image bytes using a context manager.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering pages.

    Returns:
        A list of PNG image bytes, one per page. Returns empty list on error.
    """
    images: List[bytes] = []
    try:
        # Use context manager for automatic closing
        with fitz.open(pdf_path) as doc:
            logging.info(f"Opened PDF '{pdf_path}' with {len(doc)} pages.")
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(dpi=dpi)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format="PNG")
                    images.append(img_buffer.getvalue())
                    logging.debug(f"  - Converted page {page_num + 1} to image ({pix.width}x{pix.height} @{dpi} DPI).")
                except Exception as page_e:
                    logging.error(f"Error processing page {page_num + 1} in '{pdf_path}': {page_e}")
            logging.info(f"Successfully converted {len(images)} pages from '{pdf_path}'.")
    except fitz.fitz.FileNotFoundError:
        logging.error(f"PDF file not found at '{pdf_path}'")
        return []
    except Exception as e:
        logging.error(f"Failed to open or process PDF '{pdf_path}': {e}")
        return []
    return images

def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encodes image bytes to a base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')

def query_openai_with_single_image_with_retry(
    client: openai.OpenAI, # Pass initialized client
    image_bytes: bytes,
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    detail: str = DEFAULT_DETAIL_LEVEL,
    retry_attempts: int = RETRY_ATTEMPTS,
    retry_delay: int = RETRY_DELAY_SECONDS
) -> Optional[str]:
    """
    Sends a SINGLE image and prompt to OpenAI API with retry logic.
    (Function body remains largely the same as before, but takes client as arg)
    """
    base64_image = encode_image_to_base64(image_bytes)
    messages_payload: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "content": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": detail
                    }
                }
            ]
        }
    ]

    last_exception = None
    for attempt in range(retry_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages_payload,
                max_tokens=max_tokens,
                timeout=90
            )
            finish_reason = response.choices[0].finish_reason
            # (Finish reason checks remain the same...)
            if finish_reason == "content_filter":
                 logging.warning("OpenAI stopped generation due to content filtering.")
                 return "[Content Filtered by OpenAI Policy]"
            elif finish_reason == "length":
                 logging.warning(f"OpenAI stopped generation because max_tokens ({max_tokens}) was reached.")
                 return response.choices[0].message.content # Return partial
            elif finish_reason != "stop":
                 logging.warning(f"OpenAI generation stopped unexpectedly with reason: {finish_reason}")

            content = response.choices[0].message.content
            logging.debug(f"OpenAI API Usage: {response.usage}")
            return content # Success

        # (Error handling remains the same...)
        except openai.AuthenticationError as e:
            logging.error(f"OpenAI Authentication Error: {e}. Check API key. No retry.")
            return None
        except openai.NotFoundError as e:
             logging.error(f"OpenAI Model Not Found Error: {e}. Check model name '{model}'. No retry.")
             return None
        except openai.RateLimitError as e:
            logging.warning(f"OpenAI Rate Limit Error: {e}. Attempt {attempt + 1}/{retry_attempts}. Retrying in {retry_delay}s...")
            last_exception = e
            time.sleep(retry_delay)
        except (openai.APIError, openai.APITimeoutError, openai.APIConnectionError) as e:
            logging.warning(f"OpenAI API Error ({type(e).__name__}): {e}. Attempt {attempt + 1}/{retry_attempts}. Retrying in {retry_delay}s...")
            last_exception = e
            time.sleep(retry_delay)
        except Exception as e:
            logging.error(f"An unexpected error occurred during OpenAI API call: {e}")
            last_exception = e
            break # Break on truly unexpected errors

    logging.error(f"Failed to get response from OpenAI after {retry_attempts} attempts. Last error: {last_exception}")
    return None

def extract_text_with_easyocr(
    reader: easyocr.Reader, # Pass initialized reader
    image_bytes: bytes
) -> Optional[str]:
    """
    Extracts text from a single image using the initialized easyocr Reader.

    Args:
        reader: The initialized easyocr.Reader instance.
        image_bytes: Byte content of the PNG image.

    Returns:
        The extracted text joined by newlines, or None if an error occurs.
    """
    try:
        # detail=0 means only return the text, not coordinates or confidence
        results = reader.readtext(image_bytes, detail=0, paragraph=True)
        # paragraph=True tries to combine text into paragraphs
        # If paragraph=False (default), results is List[str] directly
        # If paragraph=True, results is List[str] representing paragraphs
        
        # Join the text segments/paragraphs with newlines
        extracted_text = "\n".join(results)
        logging.debug(f"easyocr extracted {len(extracted_text)} characters.")
        return extracted_text
    except Exception as e:
        logging.error(f"An error occurred during easyocr text extraction: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDF pages using OpenAI or easyocr, processing one page at a time.")
    parser.add_argument("input_pdf", help="Path to the input PDF file.")
    parser.add_argument("output_md", help="Path to the output Markdown file.")

    # Engine selection
    parser.add_argument("--engine", choices=["openai", "easyocr"], default=DEFAULT_ENGINE, help=f"Text extraction engine (default: {DEFAULT_ENGINE})")

    # OpenAI specific arguments
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (used only if engine is openai) (default: {DEFAULT_MODEL})")
    parser.add_argument("--detail", choices=["low", "high", "auto"], default=DEFAULT_DETAIL_LEVEL, help=f"OpenAI image detail level (default: {DEFAULT_DETAIL_LEVEL})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"OpenAI max tokens per response (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--prompt", default="Extract all readable text from this image. Structure the output clearly, using Markdown formatting where appropriate (e.g., headings, lists).", help="Prompt for OpenAI.")
    parser.add_argument("--prompt-file", help="Path to a file containing the OpenAI prompt (overrides --prompt).")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS, help=f"Delay (seconds) between OpenAI API calls (default: {DEFAULT_DELAY_SECONDS})")


    # EasyOCR specific arguments
    parser.add_argument("--langs", nargs='+', default=DEFAULT_EASYOCR_LANGS, help=f"Languages for easyocr (e.g., en fr es) (default: {' '.join(DEFAULT_EASYOCR_LANGS)})")
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True, help="Use GPU for easyocr if available (use --no-gpu to disable)") # Provides --gpu and --no-gpu

    # General arguments
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"Resolution (DPI) for rendering PDF pages (default: {DEFAULT_DPI})")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # --- Initialize selected engine ---
    if args.engine == "openai":
        if not initialize_openai_client():
             logging.error("Exiting due to OpenAI client initialization failure.")
             return # Exit if client fails
        # Determine Prompt
        prompt_to_use = args.prompt
        if args.prompt_file:
            try:
                with open(args.prompt_file, 'r', encoding='utf-8') as f:
                    prompt_to_use = f.read().strip()
                logging.info(f"Using OpenAI prompt from file: {args.prompt_file}")
            except Exception as e:
                logging.error(f"Failed to read prompt file '{args.prompt_file}': {e}. Using default/provided prompt.")
                # Fall back to args.prompt already assigned
        else:
            prompt_to_use = args.prompt # Ensure it's set if file isn't used

    elif args.engine == "easyocr":
        if not initialize_easyocr_reader(args.langs, args.gpu):
             logging.error("Exiting due to easyocr Reader initialization failure.")
             return # Exit if reader fails
    else:
        # Should not happen due to argparse choices, but good practice
        logging.error(f"Invalid engine selected: {args.engine}")
        return

    # --- 1. Convert PDF to Images ---
    images = convert_pdf_to_images(args.input_pdf, dpi=args.dpi)

    if not images:
        logging.error(f"No images were extracted from '{args.input_pdf}'. Exiting.")
        return # Exit if no images

    # --- 2. Process Images One by One using Selected Engine ---
    logging.info(f"Starting text extraction using engine '{args.engine}' for {len(images)} pages...")
    all_pages_text: List[str] = []
    total_pages = len(images)

    for i, image_bytes in enumerate(images):
        page_num = i + 1
        logging.info(f"--- Processing Page {page_num}/{total_pages} ---")
        page_text: Optional[str] = None # Initialize page_text for this iteration

        # Call the appropriate engine function
        if args.engine == "openai":
            # Ensure client is initialized (should be, but check is safe)
            if openai_client:
                page_text = query_openai_with_single_image_with_retry(
                    client=openai_client, # Pass the client instance
                    image_bytes=image_bytes,
                    prompt=prompt_to_use,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    detail=args.detail,
                    retry_attempts=RETRY_ATTEMPTS,
                    retry_delay=RETRY_DELAY_SECONDS
                )
                # Apply delay only for OpenAI and if not the last page
                if args.delay > 0 and i < total_pages - 1:
                    logging.debug(f"Waiting {args.delay}s before next request...")
                    time.sleep(args.delay)
            else:
                 logging.error("OpenAI client not available for processing.")
                 # page_text remains None

        elif args.engine == "easyocr":
             # Ensure reader is initialized
             if ocr_reader:
                 page_text = extract_text_with_easyocr(
                     reader=ocr_reader, # Pass the reader instance
                     image_bytes=image_bytes
                 )
             else:
                  logging.error("easyocr Reader not available for processing.")
                  # page_text remains None


        # Handle result (common to both engines)
        if page_text is not None: # Check for None explicitly
            logging.info(f"Page {page_num}: Text extracted successfully using {args.engine}.")
            all_pages_text.append(f"## Page {page_num}\n\n{page_text}")
        else:
            logging.warning(f"Page {page_num}: Failed to extract text using {args.engine}.")
            all_pages_text.append(f"## Page {page_num}\n\n[Error extracting text for this page using {args.engine}]")


    # --- 3. Combine results and save ---
    if all_pages_text:
        combined_text = "\n\n---\n\n".join(all_pages_text)
        try:
            with open(args.output_md, "w", encoding="utf-8") as md_file:
                md_file.write(combined_text)
            logging.info(f"Markdown Conversion Complete! Combined output saved to '{args.output_md}'.")
        except IOError as e:
            logging.error(f"Error writing output file '{args.output_md}': {e}")
    else:
        logging.warning("No text content was generated or collected.")


if __name__ == '__main__':
    main()