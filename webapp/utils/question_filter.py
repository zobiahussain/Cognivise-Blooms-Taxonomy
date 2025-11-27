import re

def clean_and_extract_questions(text):
    """
    Extract ALL numbered questions from text.
    Only excludes: headers, footers, and instructions.
    """
    text = text.replace('\r', '\n').replace('\u2028', '\n')
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    # ONLY exclude headers, footers, and instructions - be very specific
    exclude_patterns = [
        r'^COMPUTER SCIENCE\s+EXAM',  # Header
        r'^Introduction\s+to\s+',  # Header
        r'^Instructions?:',  # Instructions line
        r'^Answer\s+all\s+questions',  # Instructions
        r'^Each\s+question\s+is\s+worth',  # Instructions
        r'^Time\s+Allowed',  # Footer/Header
        r'^Maximum\s+Marks',  # Footer/Header
        r'^PAPER',  # Header
        r'^SECTION',  # Header
        r'^\d+-\d+-\d+$',  # Page numbers (standalone)
        r'^\(?\d+\)?$',  # Standalone page numbers
    ]
    
    questions = []
    current_question = None
    current_num = None
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped or len(line_stripped) < 2:
            continue
        
        line_lower = line_stripped.lower()
        
        # Check if this line should be excluded (headers/footers/instructions)
        should_exclude = False
        for pattern in exclude_patterns:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                should_exclude = True
                break
        
        # Also exclude very long all-caps lines (likely headers)
        if line_stripped.isupper() and len(line_stripped) > 25:
            should_exclude = True
        
        if should_exclude:
            # Save current question before excluding
            if current_question and current_num:
                questions.append(current_question.strip())
                current_question = None
                current_num = None
            continue
        
        # Check if line starts with a number (question pattern)
        # Match: "1.", "1)", "1 ", "Q1", "Question 1", "(i)", "i)", "(a)", "a)"
        number_match = re.match(r'^(\d+)\.\s*(.+)$', line_stripped)  # "1. text"
        if not number_match:
            number_match = re.match(r'^(\d+)\)\s*(.+)$', line_stripped)  # "1) text"
        if not number_match:
            number_match = re.match(r'^(\d+)\s+(.+)$', line_stripped)  # "1 text"
        if not number_match:
            number_match = re.match(r'^Q(\d+)\.?\s*(.+)$', line_stripped, re.IGNORECASE)  # "Q1 text"
        if not number_match:
            number_match = re.match(r'^Question\s+(\d+)\.?\s*(.+)$', line_stripped, re.IGNORECASE)  # "Question 1 text"
        if not number_match:
            number_match = re.match(r'^\(([ivx]+)\)\s*(.+)$', line_stripped, re.IGNORECASE)  # "(i) text"
        if not number_match:
            number_match = re.match(r'^([ivx]+)\)\s*(.+)$', line_stripped, re.IGNORECASE)  # "i) text"
        if not number_match:
            number_match = re.match(r'^\(([a-z])\)\s*(.+)$', line_stripped, re.IGNORECASE)  # "(a) text"
        if not number_match:
            number_match = re.match(r'^([a-z])\)\s*(.+)$', line_stripped, re.IGNORECASE)  # "a) text"
        
        if number_match:
            # Save previous question
            if current_question and current_num:
                questions.append(current_question.strip())
            
            # Start new question
            current_num = number_match.group(1)
            question_text = number_match.group(2) if len(number_match.groups()) > 1 else ""
            current_question = question_text.strip()
        else:
            # This line might continue the current question
            if current_question is not None:
                # Check if next line starts a new question
                next_starts_question = False
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if re.match(r'^\d+[\.\)\s]', next_line) or re.match(r'^Q\d+', next_line, re.IGNORECASE):
                        next_starts_question = True
                
                # If next line doesn't start a question, append current line
                if not next_starts_question and len(line_stripped) > 0:
                    # Don't append standalone numbers or very short formatting
                    if not re.match(r'^\d+$', line_stripped) and len(line_stripped) > 1:
                        current_question += " " + line_stripped
    
    # Save last question
    if current_question and current_num:
        questions.append(current_question.strip())
    
    # Final pass: remove duplicates and very short items
    filtered = []
    seen = set()
    
    for q in questions:
        q = q.strip()
        if len(q) < 3:  # Very permissive - only skip if less than 3 chars
            continue
        
        # Skip duplicates
        q_key = q[:80].lower()
        if q_key in seen:
            continue
        seen.add(q_key)
        
        filtered.append(q)
    
    return filtered




