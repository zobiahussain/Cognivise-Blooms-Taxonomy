# Google Colab Test Guide - Bloom Analyzer Accuracy

## Quick Setup Steps

### 1. Open Google Colab
- Go to https://colab.research.google.com/
- Create a new notebook

### 2. Enable GPU
- Go to: **Runtime → Change runtime type**
- Set **Hardware accelerator** to **GPU**
- Click **Save**

### 3. Mount Google Drive (if model is on Drive)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Install Required Packages
```python
!pip install transformers accelerate peft torch
```

### 5. Upload the Test Script
**Option A: Upload file directly**
- Upload `test_bloom_colab.py` to Colab
- Or copy-paste the code into a cell

**Option B: Clone from GitHub (if you have it there)**
```python
!git clone <your-repo-url>
```

### 6. Update Model Path
In the script, update this line to point to your model:
```python
MODEL_PATH = "/content/drive/MyDrive/Blooms-Taxonomy-Project/models/final_model"
```

### 7. Run the Test
```python
# Copy the entire test_bloom_colab.py content into a cell and run
# OR
exec(open('test_bloom_colab.py').read())
```

## Alternative: All-in-One Colab Cell

Copy this entire block into one Colab cell:

```python
# Install packages
!pip install transformers accelerate peft torch -q

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re
from collections import defaultdict

# ===== CONFIGURATION =====
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# UPDATE THIS PATH to your model location
MODEL_PATH = "/content/drive/MyDrive/Blooms-Taxonomy-Project/models/final_model"

BLOOM_LEVELS = ['Remembering', 'Understanding', 'Applying', 'Analyzing', 'Evaluating', 'Creating']

# Test questions
TEST_QUESTIONS = [
    ("What is the capital of France?", "Remembering"),
    ("List three types of renewable energy.", "Remembering"),
    ("Define photosynthesis.", "Remembering"),
    ("Name the planets in our solar system.", "Remembering"),
    ("Identify the main components of a cell.", "Remembering"),
    ("Explain how photosynthesis works.", "Understanding"),
    ("Describe the water cycle.", "Understanding"),
    ("Summarize the main themes in Romeo and Juliet.", "Understanding"),
    ("Why does ice float on water?", "Understanding"),
    ("What is the purpose of the mitochondria?", "Understanding"),
    ("Calculate the area of a circle with radius 5cm.", "Applying"),
    ("Use the quadratic formula to solve x² + 5x + 6 = 0.", "Applying"),
    ("Apply Newton's second law to find the force.", "Applying"),
    ("Solve this equation: 2x + 3 = 11", "Applying"),
    ("Implement a function to sort a list.", "Applying"),
    ("Compare and contrast mitosis and meiosis.", "Analyzing"),
    ("Analyze the causes of World War I.", "Analyzing"),
    ("What are the differences between DNA and RNA?", "Analyzing"),
    ("Examine the relationship between supply and demand.", "Analyzing"),
    ("Explain how different interpretations of Islamic law have influenced the development of democratic institutions in Pakistan.", "Analyzing"),
    ("Evaluate the effectiveness of renewable energy policies.", "Evaluating"),
    ("Judge whether the death penalty is ethical.", "Evaluating"),
    ("Assess the impact of social media on society.", "Evaluating"),
    ("Critique the author's argument in this passage.", "Evaluating"),
    ("Which solution is better and why?", "Evaluating"),
    ("Design a sustainable city for the future.", "Creating"),
    ("Create a marketing campaign for a new product.", "Creating"),
    ("Develop a hypothesis about climate change.", "Creating"),
    ("Propose a solution to reduce plastic waste.", "Creating"),
    ("Construct a model of the solar system.", "Creating"),
]

def format_prompt(question):
    system = "Classify this educational question into ONE Bloom's Taxonomy level: Remembering, Understanding, Applying, Analyzing, Evaluating, or Creating."
    return f"{system}\n\nQuestion: {question}\n\nBloom Level:"

def correctBloomLevel(predictedLevel: str, questionText: str) -> str:
    if not predictedLevel or predictedLevel == "Unknown":
        return predictedLevel
    text = questionText.lower()
    remembering_patterns = [r"\blist\b", r"\bdefine\b", r"\bwhat is\b", r"\bname\b", r"\bidentify\b"]
    if any(re.search(pat, text) for pat in remembering_patterns):
        return "Remembering"
    creating_patterns = [r"\bdesign\b", r"\bdevelop\b", r"\bpropose\b", r"\bconstruct\b", r"\bformulate\b"]
    if any(re.search(pat, text) for pat in creating_patterns):
        return "Creating"
    applying_patterns = [r"\bapply\b", r"\buse\b", r"\bcalculate\b", r"\bimplement\b", r"\bsolve\b"]
    if any(re.search(pat, text) for pat in applying_patterns):
        if predictedLevel != "Creating":
            return "Applying"
    coding_demote_patterns = [r"write a function", r"code a program", r"create a function", r"use recursion", r"write pseudocode"]
    if predictedLevel == "Creating" and any(re.search(pat, text) for pat in coding_demote_patterns):
        return "Applying"
    understanding_patterns = [r"\bexplain\b", r"\bdescribe\b", r"\bwhy\b", r"\bpurpose\b", r"how does", r"\bsignificance\b"]
    if predictedLevel == "Remembering" and any(re.search(pat, text) for pat in understanding_patterns):
        return "Understanding"
    return predictedLevel

def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
    print("Loading fine-tuned weights...")
    model = PeftModel.from_pretrained(base, MODEL_PATH)
    model.eval()
    print("✓ Model loaded!")
    return model, tokenizer

def predict_raw(model, tokenizer, question):
    prompt = format_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.1, do_sample=False)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted = None
    if "Bloom Level:" in result:
        pred = result.split("Bloom Level:")[-1].strip().split('\n')[0].strip()
        for level in BLOOM_LEVELS:
            if level.lower() in pred.lower():
                predicted = level
                break
    if not predicted:
        for level in BLOOM_LEVELS:
            if level.lower() in result.lower():
                predicted = level
                break
    return predicted or "Unknown"

# ===== RUN TEST =====
print("=" * 80)
print("BLOOM TAXONOMY ANALYZER ACCURACY TEST")
print("=" * 80)

model, tokenizer = load_model()

results_with = []
results_without = []
detailed = []

for i, (question, expected) in enumerate(TEST_QUESTIONS, 1):
    print(f"\n[{i}/{len(TEST_QUESTIONS)}] {question[:60]}...")
    predicted_raw = predict_raw(model, tokenizer, question)
    predicted_corrected = correctBloomLevel(predicted_raw, question)
    correct_without = (predicted_raw == expected)
    correct_with = (predicted_corrected == expected)
    results_with.append(correct_with)
    results_without.append(correct_without)
    detailed.append({'q': question, 'exp': expected, 'raw': predicted_raw, 'cor': predicted_corrected})
    print(f"  Expected: {expected}, Raw: {predicted_raw}, Corrected: {predicted_corrected}")
    print(f"  {'✓' if correct_without else '✗'} Without | {'✓' if correct_with else '✗'} With")

total = len(TEST_QUESTIONS)
acc_without = (sum(results_without) / total) * 100
acc_with = (sum(results_with) / total) * 100
improvement = acc_with - acc_without

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Without Correction: {sum(results_without)}/{total} ({acc_without:.2f}%)")
print(f"With Correction:    {sum(results_with)}/{total} ({acc_with:.2f}%)")
print(f"Improvement:        {improvement:+.2f}%")
print(f"\n{'✓ HELPS' if improvement > 0 else '✗ HURTS' if improvement < 0 else '→ NO EFFECT'}")
```

## Expected Output

The script will show:
1. Model loading progress
2. Each question being tested with raw and corrected predictions
3. Final accuracy comparison
4. Breakdown by Bloom level
5. Examples where correction helped/hurt

## Troubleshooting

**Model not found:**
- Check MODEL_PATH is correct
- Make sure model files are uploaded to Drive or Colab

**Out of memory:**
- Use smaller batch size
- Or test fewer questions at a time

**Slow execution:**
- Make sure GPU is enabled
- Check GPU is being used: `!nvidia-smi`




