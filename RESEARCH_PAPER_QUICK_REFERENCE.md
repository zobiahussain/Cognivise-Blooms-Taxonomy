# Research Paper Quick Reference
## Bloom's Taxonomy Exam Analyzer & Question Generator

---

## KEY METRICS (For Abstract/Results)

- **Test Accuracy**: 82.42%
- **Model**: TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning Method**: LoRA (r=16, alpha=32)
- **Trainable Parameters**: ~1-2% of total (~22-44M)
- **Training Data**: 2,948 questions
- **Test Set**: 91 questions
- **Training Time**: 1-2 hours (free Colab)
- **Epochs**: 3

---

## IDEAL DISTRIBUTION (Educational Standards)

| Bloom Level | Ideal % |
|------------|---------|
| Remembering | 15% |
| Understanding | 20% |
| Applying | 25% |
| Analyzing | 20% |
| Evaluating | 10% |
| Creating | 10% |

---

## METHODOLOGY SUMMARY

1. **Base Model**: TinyLlama-1.1B (1.1B parameters)
2. **Fine-tuning**: LoRA (Parameter-Efficient Fine-Tuning)
3. **Training**: 3 epochs, FP16 mixed precision
4. **Evaluation**: Standard classification accuracy
5. **Post-processing**: Bloom Correction Layer (BCL)

---

## SYSTEM ARCHITECTURE

### Module 1: Classification
- Input: Question text
- Output: Bloom level + analysis
- Features: Batch processing, fallback classification, BCL

### Module 2: Generation
- Architecture: RAG (Retrieval-Augmented Generation)
- Embeddings: sentence-transformers/paraphrase-MiniLM-L3-v2
- LLM: Local (Qwen2.5-7B) or API (Grok/Gemini)

---

## 📈 PERFORMANCE HIGHLIGHTS

- ✓ Handles balanced test sets
- ✓ Handles imbalanced real-world exams
- ✓ Graceful degradation for missing categories
- ✓ Real-world validation: BISE exam analysis (79.18 quality score)

---

## TECHNOLOGY STACK

- **Core**: PyTorch, Transformers, PEFT
- **Web**: Streamlit
- **Visualization**: Plotly
- **NLP**: sentence-transformers, NLTK
- **Extraction**: pdfplumber, OCR

---

## PAPER STRUCTURE SUGGESTIONS

1. **Abstract**: 82.42% accuracy, LoRA fine-tuning, 2,948 training samples
2. **Introduction**: Bloom's Taxonomy, automated assessment, LLM fine-tuning
3. **Related Work**: Educational NLP, question classification, parameter-efficient fine-tuning
4. **Methodology**: 
   - Model selection (TinyLlama)
   - LoRA configuration
   - Training procedure
   - Evaluation metrics
5. **Results**: 
   - Test accuracy: 82.42%
   - Real-world validation (BISE exams)
   - Quality scoring system
6. **Discussion**: 
   - Limitations (82.42% accuracy)
   - Advantages (efficiency, cost-effectiveness)
   - Future work
7. **Conclusion**: Summary of contributions and impact

---

## KEY POINTS FOR DISCUSSION

### Strengths
- Parameter-efficient (1-2% trainable parameters)
- Cost-effective (free Colab training)
- Practical deployment (web application)
- Real-world validation

### Limitations
- 82.42% accuracy (room for improvement)
- Domain generalization concerns
- Model size limitations (1.1B parameters)

### Future Work
- Larger training datasets
- Experiment with larger base models
- Multi-language support
- Domain-specific fine-tuning

---

## CITATIONS TO INCLUDE

1. **TinyLlama**: Zhang et al. (2024)
2. **LoRA**: Hu et al. (2021) - "LoRA: Low-Rank Adaptation of Large Language Models"
3. **Bloom's Taxonomy**: Bloom, B. S. (1956)
4. **Transformers**: Wolf et al. (2020)
5. **PEFT**: Hugging Face library

---

## FIGURES & TABLES NEEDED

1. **Table 1**: Training hyperparameters
2. **Table 2**: Test set performance metrics
3. **Figure 1**: System architecture diagram
4. **Figure 2**: Distribution comparison (Actual vs. Ideal)
5. **Figure 3**: Quality score visualization
6. **Table 3**: Real-world exam analysis (BISE)

---

## RESEARCH QUESTIONS

1. How does LoRA compare to full fine-tuning for this task?
2. What is the optimal Bloom's Taxonomy distribution?
3. Can RAG-based generation produce pedagogically sound questions?
4. How does model size affect accuracy?
5. What are cross-domain generalization challenges?

---

## 📅 PROJECT TIMELINE

- **Completion**: November 13, 2025
- **Training**: 1-2 hours on free Colab
- **Status**: Complete and deployed

---

**For detailed information, see: `RESEARCH_PAPER_DETAILS.md`**











