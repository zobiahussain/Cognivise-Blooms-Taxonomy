import pdfplumber
import os
from utils.question_filter import clean_and_extract_questions

def extract_all_text_from_pdf(pdf_file):
    """
    Extract ALL text from a PDF file without filtering.
    Returns the full text as a string.
    """
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_questions_from_pdf(pdf_file):
    """
    Extract questions from a PDF file, filtering out headers, footers, and non-question content.
    Returns a list of strings, each representing a question.
    """
    text = extract_all_text_from_pdf(pdf_file)
    
    # Use smart filtering to extract only questions
    questions = clean_and_extract_questions(text)
    
    return questions
