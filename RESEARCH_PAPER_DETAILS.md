# Bloom's Taxonomy Exam Analyzer & Question Generator
## Research Paper Details & Technical Documentation

---

## 1. PROJECT OVERVIEW

### 1.1 Title
**Automatic Classification and Generation of Educational Questions Using Bloom's Taxonomy with Fine-Tuned Large Language Models**

### 1.2 Abstract Summary
This project presents an AI-powered system for automatically classifying educational questions according to Bloom's Taxonomy and generating contextually appropriate questions to improve exam quality. The system uses a fine-tuned TinyLlama-1.1B model with LoRA (Low-Rank Adaptation) to achieve 82.42% classification accuracy on a test set of 91 questions, trained on 2,948 educational questions across six Bloom's Taxonomy levels.

### 1.3 Key Contributions
- **Automated Classification**: Fine-tuned LLM for accurate Bloom's Taxonomy classification
- **Exam Quality Analysis**: Compares actual exam distributions against educational research standards
- **Intelligent Question Generation**: RAG-based system for generating contextually relevant questions
- **Web Application**: Full-featured Streamlit interface with PDF/image extraction capabilities
- **Cost-Effective Solution**: Uses parameter-efficient fine-tuning (LoRA) requiring only 1-2% trainable parameters

---

## 2. METHODOLOGY

### 2.1 Base Model
- **Model**: TinyLlama-1.1B-Chat-v1.0
- **Parameters**: 1.1 billion
- **Architecture**: Causal Language Model (Decoder-only Transformer)
- **Rationale**: Lightweight, efficient, suitable for fine-tuning on consumer hardware

### 2.2 Fine-Tuning Approach
- **Method**: LoRA (Low-Rank Adaptation)
- **LoRA Configuration**:
  - Rank (r): 16
  - Alpha: 32
  - Dropout: 0.05
  - Target Modules: Query, Key, Value, Output projections
- **Trainable Parameters**: ~1-2% of total model parameters (~22-44M parameters)
- **Training Time**: 1-2 hours on free Google Colab (T4 GPU)
- **Epochs**: 3
- **Framework**: Hugging Face Transformers + PEFT (Parameter-Efficient Fine-Tuning)

### 2.3 Training Data
- **Total Training Samples**: 2,948 questions
- **Validation Samples**: Included in split
- **Test Samples**: 91 questions
- **Data Format**: CSV with columns: `Questions`, `bloom_level`
- **Bloom Levels**: 6 categories (Remembering, Understanding, Applying, Analyzing, Evaluating, Creating)
- **Data Sources**: Educational questions from various domains (Computer Science, General Science, etc.)

### 2.4 Training Procedure
1. **Data Preprocessing**: Questions formatted with classification prompts
2. **Model Setup**: Base TinyLlama loaded with LoRA adapters
3. **Training Configuration**:
   - Learning rate: Standard for LoRA fine-tuning
   - Batch size: Optimized for available GPU memory
   - Mixed precision: FP16 for efficiency
4. **Evaluation**: Test accuracy calculated on held-out test set

---

## 3. SYSTEM ARCHITECTURE

### 3.1 Core Components

#### Module 1: Classification & Analysis
- **Function**: Classifies questions into Bloom's Taxonomy levels
- **Input**: Raw question text
- **Output**: Bloom level classification + detailed analysis
- **Features**:
  - Batch processing for efficiency
  - Fallback keyword-based classification
  - Bloom Correction Layer (BCL) for post-processing refinement

#### Module 2: Question Generation & Exam Improvement
- **Function**: Generates questions at specific Bloom levels and improves exam papers
- **Architecture**: RAG (Retrieval-Augmented Generation)
- **Components**:
  - Vector store (embeddings-based similarity search)
  - LLM integration (local model or API-based: Grok/Gemini)
  - Content extraction from PDFs/images
  - Web content search capabilities

### 3.2 Web Application Architecture
- **Framework**: Streamlit
- **Features**:
  - User authentication system
  - PDF question extraction
  - Image-based question extraction (OCR)
  - Interactive visualizations (Plotly)
  - Export capabilities (CSV, TXT)
  - User history tracking
  - Book/content management

---

## 4. TECHNICAL SPECIFICATIONS

### 4.1 Classification Pipeline

#### Prompt Format
```
System: "Classify this educational question into ONE Bloom's Taxonomy level: 
         Remembering, Understanding, Applying, Analyzing, Evaluating, or Creating."

Question: {question_text}

Bloom Level:
```

#### Inference Parameters
- **Max New Tokens**: 6-10 (for classification)
- **Temperature**: 0.1 (low for deterministic classification)
- **Sampling**: Greedy decoding (do_sample=False)
- **Batch Processing**: Enabled for efficiency

#### Post-Processing
- **Bloom Correction Layer (BCL)**: Heuristic-based refinement
  - Pattern matching for common misclassifications
  - Keyword-based corrections
  - Context-aware adjustments

### 4.2 Ideal Distribution Standards
Based on educational research, the system uses the following ideal distribution:
- **Remembering**: 15%
- **Understanding**: 20%
- **Applying**: 25%
- **Analyzing**: 20%
- **Evaluating**: 10%
- **Creating**: 10%

**Rationale**:
- Lower levels (15-20%): Foundation building
- Middle levels (20-25%): Core competency development
- Higher levels (10%): Critical thinking and creativity

### 4.3 Quality Scoring System
- **Formula**: `Quality Score = max(0, 100 - (total_deviation / 2))`
- **Rating Scale**:
  - 90-100: Excellent
  - 80-89: Good
  - 70-79: Fair
  - 60-69: Needs Improvement
  - <60: Poor

### 4.4 Question Generation System

#### RAG Architecture
- **Embedding Model**: sentence-transformers/paraphrase-MiniLM-L3-v2
- **Vector Store**: In-memory similarity search (cosine similarity)
- **Chunking Strategy**: Content-based chunking with metadata
- **Retrieval**: Top-k similarity search (k=5)

#### Generation Models
- **Local**: Optimized quantized models (Qwen2.5-7B-Instruct as default)
- **API Options**: 
  - Grok API (default, fast)
  - Gemini API (alternative)

#### Generation Process
1. Content extraction from PDF/image/text
2. Chunking and embedding
3. Similarity search for relevant context
4. Prompt construction with context
5. LLM generation with specific Bloom level targeting
6. Post-processing and validation

---

## 5. PERFORMANCE METRICS

### 5.1 Classification Accuracy
- **Test Accuracy**: 82.42% (82.42/100)
- **Test Set Size**: 91 questions
- **Model**: TinyLlama-1.1B fine-tuned with LoRA
- **Evaluation Date**: 2025-11-13

### 5.2 Performance Characteristics
- **Handles Balanced Test Sets**: ✓ (all classes present)
- **Handles Imbalanced Test Sets**: ✓ (real-world BISE exams)
- **Handles Missing Categories**: ✓ (graceful degradation)
- **Batch Processing**: ✓ (efficient inference)

### 5.3 Real-World Test Results
**BISE Test Set Analysis** (91 questions):
- **Quality Score**: 79.18/100 (Fair)
- **Total Deviation**: 41.65%
- **Distribution Analysis**:
  - **Deficient Categories**: Applying (-10.71%), Analyzing (-10.11%)
  - **Excessive Categories**: Remembering (+6.98%), Evaluating (+6.48%), Creating (+6.48%)
  - **Balanced Categories**: Understanding (+0.88%)

---

## 6. DATASET INFORMATION

### 6.1 Training Data Structure
- **Format**: CSV
- **Columns**: `Questions`, `bloom_level`
- **Sample Questions**:
  - "Compare preemptive and non-preemptive scheduling algorithms" → Analyzing
  - "Write a SQL query to select students with marks above average" → Applying
  - "What is the chemical formula for water?" → Remembering
  - "Develop a Python program to simulate a basic traffic signal system" → Creating

### 6.2 Data Distribution
- **Total Training Samples**: 2,948
- **Test Samples**: 91
- **Validation Samples**: Included in train/val split
- **Domains**: Computer Science, General Science, Mathematics, etc.

### 6.3 Data Preprocessing
- Question text normalization
- Prompt formatting for classification
- Label encoding for Bloom levels

---

## 7. IMPLEMENTATION DETAILS

### 7.1 Technology Stack

#### Core Libraries
- **PyTorch**: Deep learning framework
- **Transformers (Hugging Face)**: Model loading and training
- **PEFT**: Parameter-efficient fine-tuning
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data processing
- **NumPy**: Numerical operations

#### Additional Libraries
- **pdfplumber/PyPDF2**: PDF text extraction
- **sentence-transformers**: Embedding generation
- **python-dotenv**: Environment variable management
- **keybert**: Keyword extraction
- **nltk**: Natural language processing

### 7.2 Model Files
- **Final Model Path**: `models/final/` or `webapp/models/final_model/`
- **Checkpoints**: `models/checkpoints/checkpoint-1100/`, `checkpoint-1104/`
- **Adapter Files**: LoRA adapter weights (safetensors format)
- **Tokenizer**: Preserved from base model

### 7.3 Code Structure
```
Blooms-Taxonomy-Project/
├── code/
│   ├── analyzer.py              # Core classification module
│   └── module2_generator.py     # Question generation module
├── webapp/
│   ├── app.py                   # Main Streamlit application (3972 lines)
│   ├── utils/
│   │   ├── bloom_analyzer_complete.py    # Complete analyzer with BCL
│   │   ├── rag_exam_generator.py         # RAG-based generator
│   │   ├── pdf_extract.py                # PDF extraction
│   │   ├── image_extract.py              # Image/OCR extraction
│   │   └── content_extractor.py           # Content chunking
│   └── models/final_model/      # Deployed model
├── data/
│   ├── processed/               # Train/test/val splits
│   └── raw/                     # Original data
├── results/
│   ├── metrics/                  # Performance metrics
│   ├── reports/                 # Analysis reports
│   └── plots/                   # Visualizations
└── models/                      # Training checkpoints and final model
```

---

## 8. KEY FEATURES & CAPABILITIES

### 8.1 Classification Features
- ✓ Automatic classification into 6 Bloom levels
- ✓ Batch processing for multiple questions
- ✓ Fallback keyword-based classification
- ✓ Post-processing correction layer
- ✓ Confidence handling for uncertain predictions

### 8.2 Analysis Features
- ✓ Distribution comparison (Actual vs. Ideal)
- ✓ Missing category identification
- ✓ Deficient category detection (<70% of ideal)
- ✓ Excessive category detection (>130% of ideal)
- ✓ Quality score calculation
- ✓ Actionable recommendations
- ✓ Visual reports and charts

### 8.3 Generation Features
- ✓ Generate questions at specific Bloom levels
- ✓ Context-aware generation (RAG)
- ✓ PDF content extraction
- ✓ Image/OCR text extraction
- ✓ Web content search integration
- ✓ Exam improvement suggestions
- ✓ Multi-format export (CSV, TXT)

### 8.4 Web Application Features
- ✓ User authentication
- ✓ PDF upload and processing
- ✓ Image upload and OCR
- ✓ Text input (paste questions)
- ✓ Interactive visualizations
- ✓ Export functionality
- ✓ User history tracking
- ✓ Book/content management

---

## 9. EVALUATION & VALIDATION

### 9.1 Test Set Performance
- **Accuracy**: 82.42%
- **Test Size**: 91 questions
- **Evaluation Method**: Standard classification accuracy

### 9.2 Real-World Validation
- **BISE Exam Analysis**: Successfully analyzed real-world exam papers
- **Imbalanced Data Handling**: System handles exams with missing or excessive categories
- **Quality Assessment**: Provides meaningful quality scores and recommendations

### 9.3 Limitations
- **Accuracy**: 82.42% leaves room for improvement
- **Model Size**: 1.1B parameters may limit complex reasoning
- **Domain Generalization**: Performance may vary across different subject domains
- **Question Format**: Performance may depend on question phrasing

---

## 10. COMPARATIVE ANALYSIS

### 10.1 Advantages of Approach
1. **Parameter Efficiency**: LoRA requires only 1-2% trainable parameters
2. **Cost-Effective**: Can train on free Colab resources
3. **Fast Inference**: Lightweight model enables real-time classification
4. **Extensible**: Easy to add new features (generation, RAG, etc.)
5. **Practical**: Full web application for end-user deployment

### 10.2 Comparison with Alternatives
- **vs. Rule-Based Systems**: More flexible, learns from data
- **vs. Larger Models**: More efficient, deployable on consumer hardware
- **vs. Traditional ML**: Better semantic understanding, transfer learning
- **vs. Manual Classification**: Automated, consistent, scalable

---

## 11. FUTURE WORK & IMPROVEMENTS

### 11.1 Model Improvements
- Increase training data size
- Experiment with larger base models
- Fine-tune hyperparameters (LoRA rank, learning rate)
- Multi-task learning (classification + generation)
- Domain-specific fine-tuning

### 11.2 System Enhancements
- Multi-language support
- Advanced question filtering
- Question difficulty estimation
- Answer key generation
- Integration with LMS platforms
- Collaborative exam creation

### 11.3 Research Directions
- Few-shot learning for new domains
- Active learning for data collection
- Explainable AI for classification decisions
- Adversarial robustness testing
- Cross-domain transfer learning

---

## 12. DEPLOYMENT & USAGE

### 12.1 System Requirements
- **Python**: 3.8+
- **GPU**: Optional (CPU inference supported)
- **RAM**: 8GB+ recommended
- **Storage**: ~5GB for models and dependencies

### 12.2 Installation
```bash
pip install -r webapp/requirements.txt
```

### 12.3 API Keys (Optional)
- **Grok API**: For question generation (default)
- **Gemini API**: Alternative for generation
- Configure via `.env` file or environment variables

### 12.4 Running the Application
```bash
streamlit run webapp/app.py
```

---

## 13. CITATIONS & REFERENCES

### 13.1 Models & Libraries
- **TinyLlama**: Zhang et al. (2024) - TinyLlama: An Open-Source Small Language Model
- **LoRA**: Hu et al. (2021) - LoRA: Low-Rank Adaptation of Large Language Models
- **PEFT**: Hugging Face Parameter-Efficient Fine-Tuning library
- **Transformers**: Wolf et al. (2020) - Transformers: State-of-the-Art Natural Language Processing

### 13.2 Educational Framework
- **Bloom's Taxonomy**: Bloom, B. S. (1956) - Taxonomy of Educational Objectives
- **Ideal Distribution**: Based on educational research standards for balanced assessment

### 13.3 Related Work
- Educational question classification using NLP
- Automated exam quality assessment
- RAG for educational content generation

---

## 14. PROJECT METADATA

### 14.1 Project Information
- **Project Name**: Bloom Taxonomy Exam Analyzer & Question Generator
- **Completion Date**: 2025-11-13
- **Status**: Complete and Ready for Deployment
- **Version**: 1.0

### 14.2 Key Files
- **Main Application**: `webapp/app.py` (3972 lines)
- **Core Analyzer**: `webapp/utils/bloom_analyzer_complete.py`
- **RAG Generator**: `webapp/utils/rag_exam_generator.py`
- **Training Notebook**: `blooms_taxonomy_final.ipynb`
- **Model**: `models/final/` or `webapp/models/final_model/`

### 14.3 Results & Reports
- **Metrics**: `results/metrics/results.json`
- **Test Analysis**: `results/reports/test_set_analysis.json`
- **Visualizations**: `results/plots/`
- **Sample Reports**: `results/reports/sample_improved_exam.*`

---

## 15. TECHNICAL NOTES FOR RESEARCH PAPER

### 15.1 Methodology Section
- Emphasize parameter-efficient fine-tuning (LoRA) as a cost-effective approach
- Highlight the use of educational research standards for ideal distribution
- Discuss the combination of classification and generation in a unified system

### 15.2 Results Section
- Present 82.42% accuracy with context (test set size, model size)
- Include real-world validation (BISE exam analysis)
- Discuss quality scoring and recommendation system

### 15.3 Discussion Section
- Address limitations (82.42% accuracy, domain generalization)
- Discuss practical deployment considerations
- Compare with alternative approaches
- Suggest future improvements

### 15.4 Figures & Tables to Include
1. **Table**: Training hyperparameters
2. **Table**: Test set performance metrics
3. **Figure**: Architecture diagram (Classification + Generation)
4. **Figure**: Distribution comparison (Actual vs. Ideal)
5. **Figure**: Quality score visualization
6. **Table**: Real-world exam analysis results

---

## 16. ADDITIONAL RESEARCH QUESTIONS

### 16.1 Research Questions to Address
1. How does parameter-efficient fine-tuning (LoRA) compare to full fine-tuning for educational question classification?
2. What is the optimal distribution of Bloom's Taxonomy levels for different educational contexts?
3. Can RAG-based generation produce pedagogically sound questions?
4. How does model size affect classification accuracy in this domain?
5. What are the challenges in generalizing across different subject domains?

### 16.2 Hypotheses
- LoRA fine-tuning provides sufficient accuracy while maintaining efficiency
- Educational research standards provide valid benchmarks for exam quality
- RAG-based generation improves question relevance compared to zero-shot generation
- Smaller models can achieve reasonable performance with proper fine-tuning

---

## 17. CONTACT & ACKNOWLEDGMENTS

### 17.1 Project Completion
- **Date**: November 13, 2025
- **Training Environment**: Google Colab (Free Tier)
- **Deployment**: Streamlit Web Application

### 17.2 Key Technologies
- Hugging Face Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- Streamlit
- PyTorch
- Sentence Transformers

---

## END OF DOCUMENT

This document provides comprehensive technical details for writing a research paper on the Bloom's Taxonomy Exam Analyzer & Question Generator project. Use these details to structure your paper with appropriate sections on methodology, results, discussion, and future work.











