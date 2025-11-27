# Installation and Setup Guide

This guide will walk you through setting up the Bloom's Taxonomy Exam Analyzer from scratch.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Model Setup](#model-setup)
4. [API Configuration](#api-configuration)
5. [Running the Application](#running-the-application)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: At least 2GB free space
- **Internet**: Required for initial setup and API usage

### Software Installation

1. **Install Python**
   - Download from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"
   - Verify installation:
     ```bash
     python --version
     pip --version
     ```

2. **Install Git** (if cloning from repository)
   - Download from [git-scm.com](https://git-scm.com/downloads)
   - Verify installation:
     ```bash
     git --version
     ```

## Installation

### Step 1: Get the Code

**Option A: Clone from Repository**
```bash
git clone <repository-url>
cd Blooms-Taxonomy-Project
```

**Option B: Download ZIP**
- Download and extract the ZIP file
- Navigate to the extracted folder

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Navigate to webapp directory
cd webapp

# Install all required packages
pip install -r requirements.txt

# Install additional dependencies if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Note**: If you have a GPU, install the CUDA version of PyTorch instead:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Verify Installation

```bash
# Check if all packages are installed
pip list

# Test imports
python -c "import streamlit; import torch; import transformers; print('All imports successful!')"
```

## Model Setup

The application requires a trained model to analyze questions. You have two options:

### Option 1: Use Pre-trained Model (Recommended)

1. **Download Model from GitHub Releases**
   - Go to: https://github.com/zobiahussain/cognivise-blooms-taxonomy/releases
   - Download `blooms-taxonomy-model-v1.0.zip`
   - See [MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md) for detailed instructions

2. **Extract and Place Model Files**
   ```bash
   # Extract the zip file
   unzip blooms-taxonomy-model-v1.0.zip
   
   # Move to correct location
   mv final_model webapp/models/final_model
   ```

3. **Verify Model Files**
   ```
   webapp/models/final_model/
   ├── adapter_config.json
   ├── adapter_model.safetensors
   ├── tokenizer_config.json
   ├── tokenizer.json
   ├── tokenizer.model
   ├── special_tokens_map.json
   ├── training_args.bin
   └── README.md
   ```

4. **Verify Model Location**
   - The app will automatically detect the model in `webapp/models/final_model/`
   - All 8 files should be present

### Option 2: Train Your Own Model

**Note**: Training your own model requires advanced setup, training data, and computational resources. This is not covered in this repository. The pre-trained model provided in releases is recommended for most users.

## API Configuration

### Gemini API Setup (Recommended)

1. **Get API Key**
   - Visit: https://aistudio.google.com/apikey
   - Sign in with your Google account
   - Click "Create API Key"
   - Copy the key

2. **Create .env File**
   ```bash
   # In project root directory
   touch .env
   ```

3. **Add API Key**
   ```bash
   # Edit .env file
   GEMINI_API_KEY=your-actual-api-key-here
   ```

4. **Verify .env File**
   - Make sure `.env` is in the project root (same level as `webapp/`)
   - Never commit `.env` to git (it's in `.gitignore`)

### Local Model Mode (No API Required)

If you don't want to use Gemini API:
- The app will automatically fall back to local Qwen model
- No API key needed
- Slower but works offline

## Running the Application

### Step 1: Navigate to Project Directory

```bash
cd Blooms-Taxonomy-Project
```

### Step 2: Start the Application

```bash
streamlit run webapp/app.py
```

### Step 3: Access the Application

- The app will open automatically in your browser
- Default URL: `http://localhost:8501`
- If it doesn't open, manually navigate to the URL shown in terminal

### Step 4: First Run

1. **Register an Account**
   - Click "Register" on the login page
   - Create your account

2. **Test Analysis**
   - Go to "Analyze Exam" page
   - Enter a few sample questions
   - Click "Analyze Exam"
   - Verify the analysis works

## Verification

### Test 1: Model Loading

1. Go to "Analyze Exam" page
2. Enter a test question: "What is photosynthesis?"
3. Click "Analyze Exam"
4. **Expected**: Should show analysis without errors

### Test 2: Question Generation

1. Go to "Generate Questions" page
2. Select "Understanding" level
3. Enter topic: "Computer Science"
4. Click "Generate"
5. **Expected**: Should generate a question

### Test 3: API Connection (if using Gemini)

1. Check terminal output for:
   ```
   ✓ Loaded .env file from ...
   ✓ GEMINI_API_KEY found: ...
   ```
2. **Expected**: Should see API key confirmation

## Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
# Make sure you're in the correct directory
cd webapp

# Reinstall dependencies
pip install -r requirements.txt

# If using virtual environment, make sure it's activated
```

### Issue: "Model not found"

**Solution:**
1. Check if model files exist in `webapp/models/final_model/`
2. Verify file names match expected names
3. Check file permissions (should be readable)
4. If missing, download model from releases

### Issue: "API key not found"

**Solution:**
1. Verify `.env` file exists in project root
2. Check file content: `GEMINI_API_KEY=your-key`
3. No spaces around `=`
4. Restart the application after creating `.env`

### Issue: "Port 8501 already in use"

**Solution:**
```bash
# Use a different port
streamlit run webapp/app.py --server.port 8502

# Or kill the process using port 8501
# On Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# On macOS/Linux:
lsof -ti:8501 | xargs kill
```

### Issue: "CUDA out of memory" or slow performance

**Solution:**
1. The app uses CPU by default (safer)
2. If you have GPU, ensure CUDA is properly installed
3. Reduce batch size in code if needed
4. Close other applications to free memory

### Issue: "Permission denied" errors

**Solution:**
```bash
# On macOS/Linux, you might need:
chmod +x webapp/app.py

# Or run with sudo (not recommended)
sudo streamlit run webapp/app.py
```

### Issue: Import errors for specific packages

**Solution:**
```bash
# Install missing package individually
pip install <package-name>

# Common missing packages:
pip install sentence-transformers
pip install chromadb
pip install duckduckgo-search
```

## Advanced Configuration

### Custom Model Path

Edit `webapp/utils/bloom_analyzer_complete.py`:
```python
MODEL_PATH = "path/to/your/model"
```

### Custom Port

Create `webapp/.streamlit/config.toml`:
```toml
[server]
port = 8502
```

### Enable GPU

The app automatically detects GPU. To force CPU:
```python
# In bloom_analyzer_complete.py
use_cpu_model = True
```

## Next Steps

After successful setup:

1. **Read the README.md** for feature overview
2. **Explore the web interface** to understand all features
3. **Check the API Configuration section above** if you need to set up Gemini API

## Getting Help

If you encounter issues:

1. Check this troubleshooting section
2. Review error messages in terminal
3. Check GitHub issues (if repository is public)
4. Verify all prerequisites are met
5. Ensure you followed all setup steps

## System Requirements Details

### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 2GB
- **Python**: 3.8+

### Recommended Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 5GB+ (for models)
- **Python**: 3.9+
- **GPU**: Optional but recommended for faster inference

---

**Setup Complete!** You're ready to use the Bloom's Taxonomy Exam Analyzer.

