# Techniques to Improve Bloom Taxonomy Model Accuracy (82% → 90%+)

## Current Status
- **Current Accuracy**: 82.42%
- **Model**: TinyLlama-1.1B-Chat-v1.0
- **Training Data**: 2,948 questions
- **Test Set**: 91 questions
- **Fine-tuning**: LoRA (r=16, alpha=32, 3 epochs)

## Key Problem Areas (From Test Results)
1. **Analyzing vs Evaluating confusion** - Model predicts Evaluating when it should be Analyzing
2. **Understanding vs Remembering confusion** - "What is the significance..." questions
3. **Applying vs Creating confusion** - "create a function" vs "create a framework"

---

## Technique 1: **Targeted Data Augmentation** (Highest Impact)

### Focus on Confusion Pairs
Create more training examples for the pairs that confuse the model:

#### Analyzing vs Evaluating
```python
# Add more examples that clearly distinguish:
Analyzing examples:
- "Compare X and Y" (not judging, just comparing)
- "Differentiate between X and Y" (identifying differences)
- "Examine the relationship between X and Y" (exploring connections)
- "What are the differences between X and Y?" (identifying distinctions)
- "Analyze how X affects Y" (understanding relationships)

Evaluating examples:
- "Evaluate the effectiveness of X" (making a judgment)
- "Judge whether X is better than Y" (making a decision)
- "Assess the quality of X" (determining value)
- "Which approach is more suitable?" (making a choice)
- "Critique the validity of X" (determining worth)
```

#### Understanding vs Remembering
```python
# Add examples that show the difference:
Understanding examples:
- "What is the significance of X?" (comprehending meaning)
- "What is the purpose of X?" (understanding function)
- "Why does X happen?" (understanding cause)
- "How does X relate to Y?" (understanding connections)

Remembering examples:
- "What is X?" (recalling definition)
- "List the components of X" (recalling facts)
- "Name the steps in X" (recalling sequence)
- "Identify X" (recognizing from memory)
```

### Implementation
1. Collect 200-300 more examples for each confusion pair
2. Use Gemini to generate synthetic examples:
   ```
   "Generate 50 Analyzing questions that use 'differentiate', 'compare', 'examine'"
   "Generate 50 Evaluating questions that use 'evaluate', 'judge', 'assess'"
   ```

---

## Technique 2: **Better Training Data Balance**

### Current Issue
- Test showed Analyzing: 50% accuracy (worst performing)
- Need more Analyzing examples in training

### Solution
1. **Analyze current distribution**:
   ```python
   # Check training data distribution
   df = pd.read_csv('data/processed/train.csv')
   print(df['bloom_level'].value_counts())
   ```

2. **Balance the dataset**:
   - Ensure each level has at least 500 examples
   - Use oversampling for underrepresented levels (Analyzing, Evaluating)
   - Use undersampling for overrepresented levels if needed

3. **Stratified sampling**:
   - Ensure test/val sets have balanced representation
   - Current test set might be too small (91 questions)

---

## Technique 3: **Improved Prompt Engineering**

### Current Prompt
```
"Classify this educational question into ONE Bloom's Taxonomy level: 
Remembering, Understanding, Applying, Analyzing, Evaluating, or Creating."
```

### Enhanced Prompt (More Descriptive)
```
"Classify this educational question into ONE Bloom's Taxonomy level:

- Remembering: Recalling facts, definitions, lists, names
- Understanding: Explaining concepts, describing processes, summarizing
- Applying: Using information in new situations, solving problems, implementing
- Analyzing: Comparing, contrasting, examining relationships, breaking down
- Evaluating: Judging, critiquing, assessing, making decisions
- Creating: Designing, developing, constructing, formulating new ideas

Question: {question}
Bloom Level:"
```

### Why This Helps
- Gives model clearer definitions
- Helps distinguish Analyzing (relationships) from Evaluating (judgments)
- More context for the model to learn from

---

## Technique 4: **Hard Negative Mining**

### Strategy
1. **Identify model's mistakes** from test results
2. **Create hard negatives** - questions that are similar but different level
3. **Add to training** with correct labels

### Example
```python
# Model predicted "Evaluating" but should be "Analyzing"
Hard negative pair:
- "Differentiate the impacts of X vs Y" → Analyzing (correct)
- "Evaluate which is better: X or Y" → Evaluating (similar but different)

Add both to training with correct labels
```

---

## Technique 5: **Longer Training / More Epochs**

### Current: 3 epochs
### Try: 5-7 epochs with early stopping

```python
# Add early stopping
from transformers import EarlyStoppingCallback

training_args = TrainingArguments(
    # ... existing args ...
    num_train_epochs=7,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Trainer(
    # ... existing args ...
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)
```

---

## Technique 6: **Better LoRA Configuration**

### Current: r=16, alpha=32
### Try Different Configurations:

```python
# Option 1: Higher rank (more capacity)
LoRAConfig(r=32, alpha=64, dropout=0.1)

# Option 2: Target more layers
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Option 3: Different alpha/rank ratio
LoRAConfig(r=16, alpha=48)  # 3:1 ratio instead of 2:1
```

---

## Technique 7: **Learning Rate Schedule**

### Current: Likely fixed or cosine
### Try: Warmup + Cosine Annealing

```python
training_args = TrainingArguments(
    learning_rate=2e-4,
    warmup_steps=100,  # Gradual warmup
    lr_scheduler_type="cosine",  # Smooth decay
    # ...
)
```

---

## Technique 8: **Larger Test Set**

### Current: 91 questions (too small)
### Target: 200-300 questions

- More reliable accuracy measurement
- Better identification of failure patterns
- More confidence in improvements

---

## Technique 9: **Ensemble Methods** (Advanced)

### Combine Multiple Models
1. Train 3-5 models with different seeds
2. Use majority voting for predictions
3. Can improve accuracy by 2-5%

```python
def ensemble_predict(question, models, tokenizers):
    predictions = []
    for model, tokenizer in zip(models, tokenizers):
        pred = predict(question, model, tokenizer)
        predictions.append(pred)
    
    # Majority vote
    from collections import Counter
    return Counter(predictions).most_common(1)[0][0]
```

---

## Technique 10: **Domain-Specific Fine-Tuning**

### If Most Questions Are CS-Related
1. Pre-train on CS educational content
2. Then fine-tune on Bloom classification
3. Better understanding of technical terms

---

## Recommended Action Plan (Priority Order)

### Phase 1: Quick Wins (1-2 days)
1. ✅ **Add 200-300 examples for Analyzing vs Evaluating** (highest impact)
2. ✅ **Improve prompt with level descriptions**
3. ✅ **Balance training data distribution**

### Phase 2: Training Improvements (2-3 days)
4. ✅ **Increase epochs to 5-7 with early stopping**
5. ✅ **Try different LoRA configs** (r=32, alpha=64)
6. ✅ **Add learning rate warmup**

### Phase 3: Advanced (1 week)
7. ✅ **Hard negative mining** from test failures
8. ✅ **Expand test set to 200+ questions**
9. ✅ **Ensemble methods** (if needed)

---

## Expected Results

- **Phase 1**: 82% → 85-87%
- **Phase 2**: 85-87% → 88-90%
- **Phase 3**: 88-90% → 90-92%

---

## Data Collection Strategy

### Use Gemini to Generate Training Examples
```python
# Prompt for Gemini:
"""
Generate 50 educational questions for Bloom level: Analyzing

Requirements:
- Use verbs: compare, contrast, differentiate, examine, analyze, distinguish
- Focus on relationships, differences, connections
- NOT judgments or evaluations
- Computer Science domain preferred

Format: One question per line
"""
```

### Manual Review
- Review generated examples
- Ensure they're truly Analyzing (not Evaluating)
- Add to training data

---

## Monitoring Improvements

### Track Per-Level Accuracy
```python
# After each training run, test on:
1. Overall accuracy
2. Per-level accuracy (especially Analyzing)
3. Confusion matrix (Analyzing vs Evaluating)
4. Specific failure cases
```

### Create Test Suite
- Keep a fixed set of 50-100 challenging questions
- Test after each improvement
- Track accuracy progression

---

## Quick Implementation Script

```python
# collect_more_training_data.py
"""
Use Gemini to generate more training examples for confusion pairs
"""

import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

confusion_pairs = [
    {
        "level": "Analyzing",
        "verbs": ["compare", "contrast", "differentiate", "examine", "analyze", "distinguish"],
        "description": "Focus on relationships, differences, connections - NOT judgments",
        "count": 100
    },
    {
        "level": "Evaluating", 
        "verbs": ["evaluate", "judge", "assess", "critique", "determine"],
        "description": "Focus on making judgments, decisions, determining value",
        "count": 100
    },
    {
        "level": "Understanding",
        "verbs": ["explain", "describe", "why", "purpose", "significance"],
        "description": "Focus on comprehension, meaning, understanding concepts",
        "count": 100
    }
]

for pair in confusion_pairs:
    prompt = f"""
    Generate {pair['count']} educational questions for Bloom level: {pair['level']}
    
    Requirements:
    - Use verbs: {', '.join(pair['verbs'])}
    - {pair['description']}
    - Computer Science domain preferred
    - One question per line
    - No numbering
    
    Generate the questions:
    """
    
    response = model.generate_content(prompt)
    # Save to file
    with open(f'data/augmented/{pair["level"]}_examples.txt', 'w') as f:
        f.write(response.text)
```

---

## Summary

**Best ROI Techniques** (in order):
1. **Targeted data augmentation** for confusion pairs (Analyzing vs Evaluating)
2. **Better prompt** with level descriptions
3. **More epochs** with early stopping
4. **Balanced training data**
5. **Hard negative mining**

Start with #1 and #2 - they'll give you the biggest improvement with least effort!




