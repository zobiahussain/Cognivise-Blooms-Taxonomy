import streamlit as st
import hashlib
import json
import os
from datetime import datetime

USERS_FILE = "users.json"
HISTORY_DIR = "user_history"

def init_auth():
    """Initialize authentication system"""
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump({}, f)
    os.makedirs(HISTORY_DIR, exist_ok=True)

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email=""):
    """Register a new user"""
    init_auth()
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
    
    if username in users:
        return False, "Username already exists"
    
    users[username] = {
        'password_hash': hash_password(password),
        'email': email,
        'created_at': datetime.now().isoformat()
    }
    
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)
    
    # Create user history directory
    os.makedirs(os.path.join(HISTORY_DIR, username), exist_ok=True)
    
    return True, "Registration successful!"

def verify_user(username, password):
    """Verify user credentials"""
    init_auth()
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
    
    if username not in users:
        return False, "Username not found"
    
    if users[username]['password_hash'] != hash_password(password):
        return False, "Incorrect password"
    
    return True, "Login successful!"

def save_user_history(username, exam_data):
    """Save user's exam analysis history"""
    if not username or username == "guest":
        return
    
    history_file = os.path.join(HISTORY_DIR, username, "history.json")
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    exam_data['timestamp'] = datetime.now().isoformat()
    history.append(exam_data)
    
    # Keep only last 50 analyses
    history = history[-50:]
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

def get_user_history(username):
    """Get user's exam analysis history"""
    if not username or username == "guest":
        return []
    
    history_file = os.path.join(HISTORY_DIR, username, "history.json")
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return []


def save_user_book(username, book_name, book_path, book_type="json", metadata=None):
    """Save user's uploaded book/content for later use"""
    if not username or username == "guest":
        return False
    
    books_dir = os.path.join(HISTORY_DIR, username, "books")
    os.makedirs(books_dir, exist_ok=True)
    
    # Save book info
    books_info_file = os.path.join(HISTORY_DIR, username, "books_info.json")
    if os.path.exists(books_info_file):
        with open(books_info_file, 'r') as f:
            books_info = json.load(f)
    else:
        books_info = []
    
    # Create unique book ID
    import hashlib
    book_id_hash = hashlib.md5(f"{book_name}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
    book_id = f"{book_name.replace(' ', '_')}_{book_id_hash}"
    
    # Check if similar book already exists (by name)
    existing = [b for b in books_info if b.get('name') == book_name]
    
    if existing:
        # Update existing
        book_info = existing[0]
        book_info['last_accessed'] = datetime.now().isoformat()
        book_info['path'] = book_path  # Update path in case file was re-uploaded
    else:
        # Add new book
        book_info = {
            'id': book_id,
            'name': book_name,
            'type': book_type,
            'path': book_path,
            'uploaded_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        books_info.append(book_info)
    
    with open(books_info_file, 'w') as f:
        json.dump(books_info, f, indent=2)
    
    return True


def get_user_books(username):
    """Get list of user's uploaded books"""
    if not username or username == "guest":
        return []
    
    books_info_file = os.path.join(HISTORY_DIR, username, "books_info.json")
    if os.path.exists(books_info_file):
        with open(books_info_file, 'r') as f:
            return json.load(f)
    return []


def get_user_book_path(username, book_id):
    """Get the file path for a user's book"""
    if not username or username == "guest":
        return None
    
    books_info = get_user_books(username)
    for book in books_info:
        if book.get('id') == book_id:
            return book.get('path')
    return None



