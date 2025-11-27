# Bloom's Taxonomy Exam Analyzer & Question Generator

A comprehensive AI-powered system for analyzing educational exams using Bloom's Taxonomy and generating high-quality questions to improve exam distribution.

## Features

### Exam Analysis
- **Automatic Classification**: Classifies questions into 6 Bloom's Taxonomy levels (Remembering, Understanding, Applying, Analyzing, Evaluating, Creating)
- **Distribution Analysis**: Compares exam distribution with ideal educational standards
- **Quality Scoring**: Provides quality scores (0-100) with detailed ratings
- **Visual Reports**: Interactive charts and visualizations
- **Multiple Input Formats**: Supports PDF, images, and text input

### Question Generation
- **Smart Generation**: Uses RAG (Retrieval-Augmented Generation) with Gemini API or local models
- **Bloom-Level Specific**: Generate questions for specific cognitive levels
- **Exam Improvement**: Automatically generates questions to fill distribution gaps
- **Content-Aware**: Uses uploaded content or web search for context-aware generation

### Web Application
- **User-Friendly Interface**: Modern Streamlit-based web app
- **User Authentication**: Secure login and registration system
- **History Tracking**: Save and manage your exams
- **Export Options**: Download exams as CSV or text files

## Model Performance

- **Test Accuracy**: 82.42%
- **Model**: TinyLlama-1.1B fine-tuned with LoRA
- **Training Data**: ~2,948 questions
- **Base Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0

## Ideal Bloom Distribution

Based on educational research standards:
- **Remembering**: 15%
- **Understanding**: 20%
- **Applying**: 25%
- **Analyzing**: 20%
- **Evaluating**: 10%
- **Creating**: 10%

## Installation and Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/zobiahussain/cognivise-blooms-taxonomy.git
   cd cognivise-blooms-taxonomy
   ```

2. **Download the trained model** (Required)
   - Go to [GitHub Releases](https://github.com/zobiahussain/cognivise-blooms-taxonomy/releases)
   - Download `blooms-taxonomy-model-v1.0.zip`
   - Extract and place in `webapp/models/final_model/`
   - See [MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md) for detailed instructions

3. **Install dependencies**
   ```bash
   cd webapp
   pip install -r requirements.txt
   ```

4. **Set up API keys** (Optional but recommended)
   - Create a `.env` file in the project root
   - Add your Gemini API key:
     ```
     GEMINI_API_KEY=your-api-key-here
     ```
   - Get your API key from: https://aistudio.google.com/apikey
   - See [SETUP.md](SETUP.md) section "API Configuration" for detailed instructions

5. **Run the application**
   ```bash
   streamlit run webapp/app.py
   ```

6. **Access the app**
   - Open your browser to `http://localhost:8501`

## Detailed Setup Instructions

For comprehensive installation and setup instructions, see [SETUP.md](SETUP.md)

## Project Structure

```
cognivise-blooms-taxonomy/
├── webapp/                    # Web application
│   ├── app.py                # Main Streamlit application
│   ├── requirements.txt      # Python dependencies
│   ├── utils/                # Utility modules
│   │   ├── bloom_analyzer_complete.py    # Bloom taxonomy analyzer
│   │   ├── question_generation.py        # Question generation logic
│   │   ├── rag_exam_generator.py         # RAG-based exam generator
│   │   ├── pdf_extract.py                # PDF text extraction
│   │   ├── image_extract.py              # Image text extraction
│   │   └── auth.py                       # Authentication system
│   ├── static/               # Static assets (images, fonts)
│   └── exports/              # User-generated exports (gitignored)
├── README.md                  # Project overview and quick start
├── SETUP.md                   # Complete installation and setup guide
├── MODEL_DOWNLOAD.md          # Model download instructions
└── .gitignore                 # Git ignore rules
```

## Usage

### Analyze an Exam

1. **Upload a PDF or Image**
   - Go to "Analyze Exam" page
   - Upload your exam file
   - Click "Analyze Exam"

2. **Or Enter Questions Manually**
   - Paste questions in the text area
   - Click "Analyze Exam"

3. **View Results**
   - See quality score and distribution
   - Review recommendations
   - Download analysis report

### Generate Questions

1. **Single Question Generation**
   - Go to "Generate Questions" page
   - Select Bloom level and topic
   - Click "Generate"

2. **Complete Exam Generation**
   - Go to "Generate Exam Paper" page
   - Upload content or enter text
   - Specify number of questions
   - Click "Generate Exam Paper"

### Improve an Exam

1. **Analyze your exam first** (see above)
2. **Click "Generate Improved Exam"**
3. **Review the improved version**
4. **Download the improved exam**

## Configuration

### API Configuration

The system supports two modes:

1. **Gemini API** (Recommended)
   - Fast and high-quality generation
   - Free tier available
   - Requires API key in `.env` file

2. **Local Model** (Fallback)
   - Uses Qwen2.5-1.5B-Instruct model
   - No API key required
   - Slower but works offline

### Model Files

**Required**: The trained model is required for the application to work.

**Download the Model:**
1. Go to [GitHub Releases](https://github.com/zobiahussain/cognivise-blooms-taxonomy/releases)
2. Download `blooms-taxonomy-model-v1.0.zip`
3. Extract and place in `webapp/models/final_model/`
4. See [MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md) for detailed instructions

**Model Details:**
- Size: ~21 MB
- Accuracy: 82.42% on test set
- Base: TinyLlama-1.1B fine-tuned with LoRA

**Note**: The model is pre-trained and ready to use. Training your own model requires advanced setup and is not covered in this repository.

## Development

This repository contains the production-ready application. For development and training, you would need to set up your own training environment and data.

## API Reference

### Core Functions

```python
from utils.bloom_analyzer_complete import analyze_exam, load_model

# Load model
model, tokenizer = load_model(use_cpu_model=True)

# Analyze exam
result = analyze_exam(
    questions=["Question 1", "Question 2"],
    model=model,
    tokenizer=tokenizer,
    exam_name="My Exam"
)
```

### Question Generation

```python
from utils.question_generation import generate_questions_improved

# Generate questions
questions = generate_questions_improved(
    bloom_level="Understanding",
    topic="Computer Science",
    num_questions=5
)
```

## Troubleshooting

### Common Issues

**"Model not found" error**
- Download the model files from [GitHub Releases](https://github.com/zobiahussain/cognivise-blooms-taxonomy/releases)
- Place in `webapp/models/final_model/`
- See [MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md) for detailed instructions

**"API key not found" error**
- Create `.env` file in project root
- Add `GEMINI_API_KEY=your-key`
- Or use local model mode (no API key needed)

**"Module not found" error**
- Install all dependencies: `pip install -r webapp/requirements.txt`
- Make sure you're in the correct directory

**Port already in use**
- Change port: `streamlit run webapp/app.py --server.port 8502`
- Or kill the process using port 8501

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

**Zobia Hussain**

- Email: zobiahussain25@gmail.com
- GitHub: [@zobiahussain](https://github.com/zobiahussain)
- Repository: [cognivise-blooms-taxonomy](https://github.com/zobiahussain/cognivise-blooms-taxonomy)

For questions, issues, or contributions, please open an issue on GitHub or contact via email.

## Acknowledgments

- TinyLlama model by PygmalionAI
- Bloom's Taxonomy framework
- Streamlit for the web framework
- Google Gemini API for question generation

## Additional Resources

- [SETUP.md](SETUP.md) - Detailed setup instructions including API configuration
- [MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md) - Model download instructions

---

**Note**: This project requires model files to function. See the "Model Files" section above for instructions on obtaining or training the model.
