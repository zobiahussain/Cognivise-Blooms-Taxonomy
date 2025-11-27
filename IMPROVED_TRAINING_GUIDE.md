# Improved Context-Aware Training Guide

## Overview

This guide helps you train an improved Bloom Taxonomy classifier with:
- **Better prompt engineering** - Emphasizes context understanding over keyword matching
- **Enhanced training data** - Includes edge cases we've learned about
- **Quantized models** - Fast inference while maintaining accuracy
- **Keeps current system intact** - This is experimental, doesn't replace existing models

## What's Different

### 1. Improved Prompt
- **Before**: Simple "Classify this question into Bloom level"
- **After**: Detailed explanation of each level + guidelines for context understanding

### 2. Enhanced Training Data
- Uses your existing `data/processed/train.csv` (2,948 questions)
- Adds ~20 edge case examples we've identified:
  - "What is the use of..." = Understanding (not Remembering)
  - "Write steps..." = Applying (not Creating)
  - "Compare... Which is better?" = Evaluating (not Analyzing)
  - Multi-level questions → Highest level

### 3. Better Model Options
- **Qwen2.5-1.5B-Instruct**: Better context understanding, quantized-friendly
- **TinyLlama-1.1B**: Original (for comparison)
- Both use 4-bit quantization for speed

## Setup in Google Colab

### Step 1: Mount Drive and Install

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install transformers accelerate peft bitsandbytes datasets trl -q
!pip install torch --index-url https://download.pytorch.org/whl/cu118 -q

# Set paths
import os
BASE_PATH = "/content/drive/MyDrive/Blooms-Taxonomy-Project"
os.makedirs(f"{BASE_PATH}/models/improved_model", exist_ok=True)
```

### Step 2: Upload Training Script

Upload `train_improved_context_aware.py` to your Colab or copy-paste it into a cell.

### Step 3: Run Training

```python
# Run the training script
exec(open('train_improved_context_aware.py').read())
```

Or run it cell by cell if you prefer.

## Expected Results

### Training Time
- **Qwen2.5-1.5B**: ~2-3 hours on free Colab T4
- **TinyLlama-1.1B**: ~1-2 hours on free Colab T4

### Expected Accuracy
- **Baseline**: 82.42% (current model)
- **Target**: 85-88% (with improved prompts and edge cases)

### Model Size
- **LoRA adapters**: ~5-10 MB (saves to `models/improved_model/`)
- **Base model**: Loaded from HuggingFace (not saved)

## Testing the New Model

After training, test it:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = "Qwen/Qwen2.5-1.5B-Instruct"  # or TinyLlama
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(model, f"{BASE_PATH}/models/improved_model")

# Test
def format_prompt_improved(question):
    system = """You are an expert educational assessment classifier..."""  # (full prompt from script)
    return f"{system}\n\nQuestion: {question}\n\nBloom Level:"

question = "What is the use of image processing system?"
prompt = format_prompt_improved(question)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.1)
    
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## Comparison with Current Model

### Current Model
- Location: `models/final_model/`
- Prompt: Simple classification instruction
- Accuracy: 82.42%
- Issues: Confuses Analyzing/Evaluating, Understanding/Remembering

### Improved Model
- Location: `models/improved_model/` (NEW - doesn't replace old one)
- Prompt: Detailed context-aware instruction
- Expected Accuracy: 85-88%
- Should handle: Edge cases, multi-level questions, ambiguous questions

## If Results Are Better

If the improved model performs better:

1. **Test thoroughly** on your test set
2. **Compare side-by-side** with current model
3. **Update webapp** to use new model (optional - can keep both)
4. **Update model path** in `bloom_analyzer.py` and `bloom_analyzer_complete.py`

## If Results Are Not Better

- Keep current model (nothing deleted)
- Analyze what went wrong
- Try different prompt variations
- Try different model architectures

## Files Created

- `train_improved_context_aware.py`: Training script
- `models/improved_model/`: New model directory (if training succeeds)
- This guide: `IMPROVED_TRAINING_GUIDE.md`

## Notes

- **Current models untouched**: All existing code and models remain intact
- **Experimental**: This is a test to see if better prompts help
- **Reversible**: Can always go back to current model
- **Incremental**: Can add more edge cases to training data as needed

## Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE` to 2
- Reduce `MAX_LENGTH` to 256
- Use smaller model (TinyLlama instead of Qwen)

### Slow Training
- Reduce `NUM_EPOCHS` to 2 for testing
- Use `gradient_checkpointing=True` in training args
- Reduce `GRADIENT_ACCUMULATION` to 2

### Import Errors
- Make sure `trl` is installed: `!pip install trl -q`
- Restart runtime after installing packages

## Next Steps

1. Run training in Colab
2. Evaluate on test set
3. Compare with current model
4. Decide if worth upgrading
5. If yes, integrate into webapp (optional)




