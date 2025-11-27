"""
Content extraction and chunking utilities for RAG.
Extracts text from various sources and chunks it for vector storage.
"""

import os
import re
import json
from typing import List, Dict, Optional, Any


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except ImportError:
        raise ImportError("pdfplumber is required for PDF extraction. Install with: pip install pdfplumber")
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def extract_text_from_file(file_path: str) -> str:
    """Extract text from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")


def extract_text_from_json(json_path: str) -> str:
    """
    Extract text from a JSON file.
    Handles structured JSON books with chapters, sections, content fields, etc.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Recursively extract all text values from JSON
        text_parts = []
        
        def extract_text_recursive(obj, path=""):
            """Recursively extract text from JSON structure."""
            if isinstance(obj, dict):
                # Check for common content fields first
                for key in ['content', 'text', 'body', 'description', 'chapter', 'section', 'paragraph']:
                    if key in obj and isinstance(obj[key], str):
                        text_parts.append(obj[key])
                
                # Recursively process all values
                for key, value in obj.items():
                    extract_text_recursive(value, f"{path}.{key}" if path else key)
            
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_text_recursive(item, f"{path}[{i}]" if path else f"[{i}]")
            
            elif isinstance(obj, str):
                # Only add if it's substantial text (not just IDs or short labels)
                if len(obj.strip()) > 10 and not obj.strip().startswith(('http://', 'https://')):
                    # Avoid duplicates
                    if obj not in text_parts:
                        text_parts.append(obj)
        
        extract_text_recursive(data)
        
        # Join all text parts
        if text_parts:
            return "\n\n".join(text_parts)
        else:
            # Fallback: convert entire JSON to string representation
            return json.dumps(data, indent=2, ensure_ascii=False)
    
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON file: {str(e)}")
    except Exception as e:
        raise Exception(f"Error reading JSON file: {str(e)}")


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Uses sentence-based chunking by default (prevents mid-sentence fragments).
    Falls back to character-based if sentence-based fails.
    
    Args:
        text: Text to chunk
        chunk_size: Target size of each chunk (in characters)
        chunk_overlap: Number of characters to overlap between chunks
    
    Returns:
        List of dictionaries with 'text' and 'index' keys
    """
    if not text or not text.strip():
        return []
    
    # Clean up text
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    
    if len(text) <= chunk_size:
        return [{'text': text, 'index': 0}]
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        # Calculate end position
        end = start + chunk_size
        
        # If not the last chunk, try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 200 chars
            sentence_end = max(
                text.rfind('.', start, end),
                text.rfind('!', start, end),
                text.rfind('?', start, end),
                text.rfind('\n', start, end)
            )
            
            # If found a sentence boundary, use it
            if sentence_end > start + chunk_size * 0.5:  # At least 50% of chunk size
                end = sentence_end + 1
        
        # Extract chunk
        chunk_text = text[start:end].strip()
        
        if chunk_text:
            chunks.append({
                'text': chunk_text,
                'index': chunk_index
            })
            chunk_index += 1
        
        # Move start position with overlap
        start = end - chunk_overlap
        if start < 0:
            start = end
    
    return chunks


def extract_and_chunk_content(
    content_source: str,
    source_type: str = "text",
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Extract content from a source and chunk it for RAG.
    
    Args:
        content_source: Path to file (for 'pdf', 'text_file', 'json') or text content (for 'text')
        source_type: 'pdf', 'text_file', 'json', or 'text'
        chunk_size: Target size of each chunk (in characters)
        chunk_overlap: Number of characters to overlap between chunks
    
    Returns:
        List of dictionaries with 'text' and 'index' keys
    """
    # Extract text based on source type
    if source_type == "pdf":
        if not os.path.exists(content_source):
            raise FileNotFoundError(f"PDF file not found: {content_source}")
        text = extract_text_from_pdf(content_source)
    elif source_type == "text_file":
        if not os.path.exists(content_source):
            raise FileNotFoundError(f"Text file not found: {content_source}")
        text = extract_text_from_file(content_source)
    elif source_type == "json":
        if not os.path.exists(content_source):
            raise FileNotFoundError(f"JSON file not found: {content_source}")
        text = extract_text_from_json(content_source)
    elif source_type == "text":
        # content_source is already text
        text = content_source
    else:
        raise ValueError(f"Unknown source_type: {source_type}. Must be 'pdf', 'text_file', 'json', or 'text'")
    
    if not text or not text.strip():
        return []
    
    # Chunk the text
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    return chunks


def get_chunk_summary(chunk: Dict[str, Any], max_length: int = 100) -> str:
    """
    Get a summary of a chunk (first few sentences or characters).
    
    Args:
        chunk: Dictionary with 'text' key
        max_length: Maximum length of summary
    
    Returns:
        Summary string
    """
    text = chunk.get('text', '')
    if not text:
        return ""
    
    # Try to get first sentence
    sentences = re.split(r'[.!?]+\s+', text)
    if sentences and len(sentences[0]) <= max_length:
        return sentences[0].strip()
    
    # Otherwise, just return first max_length characters
    return text[:max_length].strip() + "..." if len(text) > max_length else text.strip()

