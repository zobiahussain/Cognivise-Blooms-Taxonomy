"""
Improved Context-Aware Bloom Taxonomy Classifier Training
==========================================================

This script trains a model with better prompt engineering that emphasizes:
1. Context understanding over keyword matching
2. Handling ambiguous/multi-level questions
3. Semantic analysis of entire sentences

Uses quantized/distilled models for speed while maintaining accuracy.

Run this in Google Colab with GPU runtime.
"""

# ============================================================================
# SETUP - Run this first in Colab
# ============================================================================
"""
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
"""

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths (adjust for your Colab setup)
BASE_PATH = "/content/drive/MyDrive/Blooms-Taxonomy-Project"
DATA_PATH = f"{BASE_PATH}/data/processed"
MODEL_SAVE_PATH = f"{BASE_PATH}/models/improved_model"

# Model choice: Use quantized/distilled model for speed
# Option 1: Qwen2.5-1.5B (better than TinyLlama, still small)
# Option 2: TinyLlama-1.1B (original, for comparison)
# Option 3: Phi-2 (Microsoft's small model, very efficient)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Better context understanding, quantized-friendly
# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Original for comparison

# Training config
SEED = 42
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_LENGTH = 512

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ============================================================================
# IMPROVED PROMPT ENGINEERING
# ============================================================================

def format_prompt_improved(question: str, label: str = None) -> str:
    """
    Improved prompt that emphasizes context understanding.
    
    Key improvements:
    1. Explains what each Bloom level means (not just names)
    2. Emphasizes understanding context, not just keywords
    3. Handles multi-level questions (use highest level)
    4. Provides examples of what to look for
    """
    
    system_prompt = """You are an expert educational assessment classifier. Your task is to classify educational questions into Bloom's Taxonomy levels by understanding the FULL CONTEXT and SEMANTIC MEANING of the question, not just individual words or verbs.

Bloom's Taxonomy Levels (from lowest to highest cognitive complexity):

1. **Remembering**: Recalling facts, definitions, lists, or basic information. Questions ask "What is...?", "Define...", "List...", "Name...". These are simple fact recall questions.

2. **Understanding**: Explaining concepts, purposes, meanings, or how/why things work. Questions ask "Explain...", "Why...", "How does...", "What is the purpose/use/significance of...". These require comprehension, not just recall.

3. **Applying**: Using knowledge in new situations, implementing procedures, solving problems. Questions ask "Implement...", "Use...", "Apply...", "Calculate...", "Write steps to...", "How to...". These require using existing knowledge practically.

4. **Analyzing**: Examining relationships, comparing, contrasting, distinguishing between concepts. Questions ask "Compare...", "Differentiate...", "Distinguish...", "Analyze the relationship...". These examine parts and relationships WITHOUT making judgments.

5. **Evaluating**: Making judgments, assessing value, deciding between options, critiquing. Questions ask "Evaluate...", "Judge...", "Which is better...", "Assess...", "Determine which...". These require making value judgments or decisions.

6. **Creating**: Designing new things, developing novel approaches, constructing original work. Questions ask "Design...", "Develop...", "Create...", "Propose...", "Construct...". These require producing something new and original.

**IMPORTANT GUIDELINES:**
- Understand the ENTIRE SENTENCE CONTEXT, not just keywords
- If a question spans multiple levels, classify it as the HIGHEST level present
- "What is the use of X?" = Understanding (asking about purpose/function)
- "Write steps to..." = Applying (procedural, not creative)
- "Compare X and Y. Which is better?" = Evaluating (has judgment, not just comparison)
- Focus on what the question is ASKING FOR, not just what words it contains

Now classify this question:"""

    if label:
        return f"{system_prompt}\n\nQuestion: {question}\n\nBloom Level: {label}"
    else:
        return f"{system_prompt}\n\nQuestion: {question}\n\nBloom Level:"


# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

def load_training_data():
    """Load training data from your existing CSV files."""
    print("Loading training data...")
    
    # Load your existing training data
    df_train = pd.read_csv(f"{DATA_PATH}/train.csv")
    df_val = pd.read_csv(f"{DATA_PATH}/val.csv")
    
    print(f"Original train: {len(df_train)} questions")
    print(f"Original val: {len(df_val)} questions")
    
    # Add additional training examples for edge cases we've learned about
    additional_examples = [
        # Edge case: "What is the use of..." = Understanding (not Remembering)
        ("What is the use of image processing system?", "Understanding"),
        ("What is the purpose of the mitochondria?", "Understanding"),
        ("What is the significance of the Magna Carta?", "Understanding"),
        
        # Edge case: "Write steps..." = Applying (not Creating)
        ("Write the steps to insert a header in MS Word.", "Applying"),
        ("Write the procedure to add two numbers in Excel.", "Applying"),
        ("Write the steps to copy and paste text.", "Applying"),
        
        # Edge case: "Compare... Which is better?" = Evaluating (not Analyzing)
        ("Compare manual and computerized systems. Which one is better and why?", "Evaluating"),
        ("Compare X and Y. Determine which is more effective.", "Evaluating"),
        
        # Edge case: "Why is X used?" can be Remembering (simple fact)
        ("Why is RAM used in computers?", "Remembering"),
        
        # Edge case: "What are impact printers?" = Remembering (technical term, not Analyzing)
        ("What are impact printers?", "Remembering"),
        
        # Edge case: Multi-level questions (use highest)
        ("Differentiate between X and Y. Which approach is more suitable?", "Evaluating"),
        ("Compare A and B. Evaluate their effectiveness.", "Evaluating"),
        
        # Edge case: "Create steps..." = Applying (not Creating)
        ("Create a simple menu-driven program in pseudocode.", "Creating"),  # This IS Creating
        ("Create steps to solve this problem.", "Applying"),  # This is NOT Creating
        
        # Edge case: "Write two functions..." = Remembering (list, not Creating)
        ("Write two functions of Network Layer.", "Remembering"),
        
        # Edge case: "What is standard code module?" = Remembering (definition)
        ("What is standard code module?", "Remembering"),
        
        # Edge case: Procedural vs Creative
        ("Write a formula to calculate simple interest.", "Applying"),
        ("Design a comprehensive framework for evaluation.", "Creating"),
    ]
    
    # Create DataFrame for additional examples
    df_additional = pd.DataFrame(additional_examples, columns=['Questions', 'bloom_level'])
    
    # Combine with original training data
    df_train_enhanced = pd.concat([df_train, df_additional], ignore_index=True)
    df_train_enhanced = df_train_enhanced.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    print(f"Enhanced train: {len(df_train_enhanced)} questions (added {len(additional_examples)} edge cases)")
    
    return df_train_enhanced, df_val


def prepare_datasets(df_train, df_val, tokenizer):
    """Prepare datasets with improved prompts and tokenize them."""
    print("Formatting data with improved prompts...")
    
    train_data = [
        {'text': format_prompt_improved(row['Questions'], row['bloom_level'])}
        for _, row in df_train.iterrows()
    ]
    
    val_data = [
        {'text': format_prompt_improved(row['Questions'], row['bloom_level'])}
        for _, row in df_val.iterrows()
    ]
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Tokenize datasets (SFTTrainer might need pre-tokenized data in some versions)
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
    
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    print(f"Train dataset: {len(train_dataset)}")
    print(f"Val dataset: {len(val_dataset)}")
    
    return train_dataset, val_dataset


# ============================================================================
# MODEL SETUP
# ============================================================================

def load_model_and_tokenizer():
    """Load model with quantization for efficiency."""
    print(f"Loading model: {MODEL_NAME}...")
    
    # Configure quantization (4-bit for efficiency)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare for LoRA training
    model = prepare_model_for_kbit_training(model)
    
    print(f"Model loaded: {MODEL_NAME}")
    print(f"Parameters: {model.num_parameters() / 1e9:.2f}B")
    
    return model, tokenizer


def setup_lora(model):
    """Configure LoRA adapters."""
    print("Configuring LoRA...")
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable parameters: {trainable:,} ({100*trainable/total:.2f}%)")
    
    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, tokenizer, train_dataset, val_dataset):
    """Train the model."""
    print("Setting up training...")
    
    training_args = TrainingArguments(
        output_dir=f"{MODEL_SAVE_PATH}/checkpoints",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        save_steps=100,
        eval_steps=100,
        save_total_limit=2,
        eval_strategy="steps",  # Changed from evaluation_strategy to eval_strategy
        load_best_model_at_end=True,
        report_to="none",
        save_strategy="steps",
        warmup_steps=50,
        max_steps=-1,
    )
    
    # Use SFTTrainer for supervised fine-tuning (better for instruction following)
    # Since datasets are pre-tokenized, we can use standard Trainer or SFTTrainer with minimal params
    try:
        # Try SFTTrainer with minimal parameters (for pre-tokenized data)
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
    except Exception as e:
        # Fallback to standard Trainer if SFTTrainer fails
        print(f"SFTTrainer failed: {e}")
        print("Falling back to standard Trainer...")
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
    
    # Set tokenizer for evaluation/inference
    trainer.tokenizer = tokenizer
    
    print("Starting training...")
    trainer.train()
    
    print("Training complete!")
    
    # Save final model
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    return trainer


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, tokenizer, test_path):
    """Evaluate on test set."""
    print("Evaluating on test set...")
    
    df_test = pd.read_csv(test_path)
    
    correct = 0
    total = 0
    
    for _, row in df_test.iterrows():
        question = row['Questions']
        true_label = row['bloom_level']
        
        prompt = format_prompt_improved(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract prediction
        predicted = None
        if "Bloom Level:" in result:
            pred_text = result.split("Bloom Level:")[-1].strip().split('\n')[0].strip()
            bloom_levels = ['Remembering', 'Understanding', 'Applying', 'Analyzing', 'Evaluating', 'Creating']
            for level in bloom_levels:
                if level.lower() in pred_text.lower():
                    predicted = level
                    break
        
        if predicted == true_label:
            correct += 1
        total += 1
        
        if total % 10 == 0:
            print(f"Processed {total}/{len(df_test)}... Accuracy: {correct/total:.2%}")
    
    accuracy = correct / total
    print(f"\nFinal Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline."""
    print("="*80)
    print("IMPROVED CONTEXT-AWARE BLOOM TAXONOMY CLASSIFIER TRAINING")
    print("="*80)
    print(f"Model: {MODEL_NAME}")
    print(f"Save path: {MODEL_SAVE_PATH}")
    print("="*80)
    
    # Set random seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Load data
    df_train, df_val = load_training_data()
    
    # Load model first (needed for tokenizer)
    model, tokenizer = load_model_and_tokenizer()
    
    # Prepare datasets (with tokenization)
    train_dataset, val_dataset = prepare_datasets(df_train, df_val, tokenizer)
    
    # Setup LoRA
    model = setup_lora(model)
    
    # Train (SFTTrainer handles tokenization internally)
    trainer = train_model(model, tokenizer, train_dataset, val_dataset)
    
    # Evaluate
    test_path = f"{DATA_PATH}/test.csv"
    if os.path.exists(test_path):
        accuracy = evaluate_model(model, tokenizer, test_path)
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS")
        print(f"{'='*80}")
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Model saved to: {MODEL_SAVE_PATH}")
        print(f"{'='*80}")
    else:
        print(f"Test file not found at {test_path}, skipping evaluation.")
    
    print("\nTraining complete! Model ready for testing.")


if __name__ == "__main__":
    main()

