import re
from PIL import Image
import io
import os
from utils.question_filter import clean_and_extract_questions

def extract_all_text_from_image(image_file):
    """
    Extract ALL text from PNG/JPG image using OCR without filtering.
    Returns the full text as a string.
    
    Args:
        image_file: Streamlit UploadedFile or file-like object
    """
    # Read image bytes - handle both file objects and bytes
    if hasattr(image_file, 'read'):
        image_bytes = image_file.read()
        # Reset file pointer if possible
        if hasattr(image_file, 'seek'):
            image_file.seek(0)
    else:
        image_bytes = image_file
    
    # Open image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Perform OCR - try pytesseract first, then easyocr
    text = ""
    try:
        import pytesseract
        text = pytesseract.image_to_string(image)
    except Exception:
        # Fallback to easyocr if pytesseract fails
        try:
            import easyocr
            import numpy as np
            reader = easyocr.Reader(['en'], gpu=False)
            np_image = np.array(image)
            result = reader.readtext(np_image)
            text = '\n'.join([item[1] for item in result])
        except Exception as e:
            raise Exception(f"OCR failed. Please install tesseract (brew install tesseract) or ensure easyocr is properly installed. Error: {e}")
    
    return text

def extract_questions_from_image(image_file):
    """
    Extract questions from PNG/JPG image using OCR, filtering out headers, footers, and non-question content.
    Returns a list of strings, each representing a question.
    
    Args:
        image_file: Streamlit UploadedFile or file-like object
    """
    # Read image bytes - handle both file objects and bytes
    if hasattr(image_file, 'read'):
        image_bytes = image_file.read()
        # Reset file pointer if possible
        if hasattr(image_file, 'seek'):
            image_file.seek(0)
    else:
        image_bytes = image_file
    
    # Open image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Perform OCR - try pytesseract first, then easyocr
    text = ""
    try:
        import pytesseract
        text = pytesseract.image_to_string(image)
    except Exception:
        # Fallback to easyocr if pytesseract fails
        try:
            import easyocr
            import numpy as np
            reader = easyocr.Reader(['en'], gpu=False)
            np_image = np.array(image)
            result = reader.readtext(np_image)
            text = '\n'.join([item[1] for item in result])
        except Exception as e:
            raise Exception(f"OCR failed. Please install tesseract (brew install tesseract) or ensure easyocr is properly installed. Error: {e}")
    
    # Use smart filtering to extract only questions
    questions = clean_and_extract_questions(text)
    
    return questions
