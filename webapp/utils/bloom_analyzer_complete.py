"""
Complete Bloom Taxonomy Analyzer Module
Includes all functionality from the notebook: classification, analysis, and visualization
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import numpy as np
import re
from collections import Counter

# Using enhanced correctBloomLevel with comprehensive rules

# Bloom Taxonomy Constants
BLOOM_LEVELS = ['Remembering', 'Understanding', 'Applying', 'Analyzing', 'Evaluating', 'Creating']

IDEAL_DISTRIBUTION = {
    'Remembering': 0.15,    # 15%
    'Understanding': 0.20,  # 20%
    'Applying': 0.25,       # 25%
    'Analyzing': 0.20,      # 20%
    'Evaluating': 0.10,     # 10%
    'Creating': 0.10        # 10%
}

# Model paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/final_model')
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Global model cache
_model = None
_tokenizer = None


def load_model(use_cpu_model=False):
    """
    Load the fine-tuned Bloom Taxonomy model.
    
    Args:
        use_cpu_model: If True, force CPU usage (for systems without GPU)
    
    Returns:
        model, tokenizer tuple
    """
    global _model, _tokenizer
    
    # Return cached model if available
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.padding_side = "right"
    
    # Determine device and dtype
    if use_cpu_model or not torch.cuda.is_available():
        device = "cpu"
        dtype = torch.float32
        device_map = None
    else:
        device = "cuda"
        dtype = torch.float16
        device_map = None  # Changed from "auto" to None to avoid meta device issues
    
    # Load base model
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=device_map
    )
    
    # Load fine-tuned LoRA weights
    if os.path.exists(MODEL_PATH):
        _model = PeftModel.from_pretrained(base, MODEL_PATH)
    else:
        # If fine-tuned model doesn't exist, use base model
        print(f"Warning: Fine-tuned model not found at {MODEL_PATH}. Using base model.")
        _model = base
    
    _model.eval()
    
    # Explicitly move to the target device
    if use_cpu_model or not torch.cuda.is_available():
        _model = _model.to("cpu")
    elif torch.cuda.is_available():
        _model = _model.to("cuda")
    
    return _model, _tokenizer


def format_prompt(question):
    """Format question for classification prompt."""
    system = "Classify this educational question into ONE Bloom's Taxonomy level: Remembering, Understanding, Applying, Analyzing, Evaluating, or Creating."
    return f"{system}\n\nQuestion: {question}\n\nBloom Level:"


def correctBloomLevel(predictedLevel: str, questionText: str) -> str:
    """
    Bloom Correction Layer (BCL) — Secondary verb-based correction
    Only overrides if model prediction doesn't match verb dictionary.
    """
    if not predictedLevel or predictedLevel == "Unknown":
        return predictedLevel
    
    text = questionText.lower()
    
    # --- Special case: "how" at the START of question (after removing numbering/prefixes) is Understanding ---
    # Strip common prefixes like "(i)", "(ii)", "7.", etc., then check if it starts with "how"
    text_clean = re.sub(r'^(\([ivx]+\)\s*|\d+\.?\s*|\([a-z]\)\s*)', '', text).strip()
    if re.search(r'^how\b', text_clean):
        # Only override if model didn't predict Understanding
        if predictedLevel != "Understanding":
            return "Understanding"
    
    # --- Verb Dictionary: Secondary override with multi-verb edge case handling ---
    # Edge case: If multiple verbs from different levels are detected, always choose the HIGHEST level
    # Otherwise: Only override if model prediction doesn't match (secondary correction)
    
    # Define verb lists in order from highest to lowest
    verb_lists = [
        ("Creating", [r"\bdesign\b", r"\bpropose\b", r"\bformulate\b", r"\bdevelop\b", r"\bconstruct\b", r"\binvent\b"]),
        ("Evaluating", [r"\bcritique\b", r"\bjustify\b", r"\bassess\b", r"\bappraise\b", r"\bdefend\b", r"\bjudge\b"]),
        ("Analyzing", [r"\bdistinguish\b", r"\bdifferentiate\b", r"\bcompare\b", r"\bcategorize\b", r"\bseparate\b", r"\bdeconstruct\b"]),
        ("Applying", [r"\bimplement\b", r"\bcompute\b", r"\bsolve\b", r"\bdemonstrate\b", r"\bsimulate\b", r"\bexecute\b"]),
        ("Understanding", [r"\bexplain\b", r"\bparaphrase\b", r"\bsummarize\b", r"\bdescribe\b", r"\billustrate\b", r"\binfer\b"]),
        ("Remembering", [r"\bdefine\b", r"\blist\b", r"\bstate\b", r"\brecall\b", r"\bname\b", r"\bidentify\b"]),
    ]
    
    # First pass: Collect all levels that have verbs present
    detected_levels = []
    for level, verbs in verb_lists:
        if any(re.search(verb, text) for verb in verbs):
            detected_levels.append(level)
    
    # Edge case: Multiple verbs detected → always return HIGHEST level
    if len(detected_levels) > 1:
        # Return the first (highest) level since verb_lists is ordered from highest to lowest
        return detected_levels[0]
    
    # Single verb detected → use secondary logic (only override if model doesn't match)
    elif len(detected_levels) == 1:
        detected_level = detected_levels[0]
        if predictedLevel != detected_level:
            return detected_level
    
    # --- Default: return original model prediction (no override) ---
    return predictedLevel

def predict(question, model, tokenizer):
    """Predict Bloom level for a single question."""
    # Use local fine-tuned TinyLlama model (from notebook training)
    if model is None or tokenizer is None:
        model, tokenizer = load_model(use_cpu_model=True)
    
    prompt = format_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move to model's device - handle various model types
    try:
        if hasattr(model, 'device'):
            device = model.device
        elif hasattr(model, 'hf_device_map'):
            # For models with device_map, get the device of the first parameter
            device = next(model.parameters()).device
        else:
            # Try to get device from model parameters
            device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except (StopIteration, AttributeError):
        # Fallback to CPU if device detection fails
        device = torch.device("cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    matched = False
    matched_level = None
    
    # First, try to extract from text immediately after "Bloom Level:"
    if "Bloom Level:" in result:
        pred = result.split("Bloom Level:")[-1].strip().split('\n')[0].strip()
        # Filter out single digits that might be question numbers
        if not (pred.isdigit() and len(pred) == 1):
            for level in BLOOM_LEVELS:
                if level.lower() in pred.lower():
                    matched_level = level
                    matched = True
                    break
    
    # If no match found, search the entire result string for Bloom level names
    if not matched:
        for level in BLOOM_LEVELS:
            if level.lower() in result.lower():
                matched_level = level
                matched = True
                break
    
    # If still no match, use fallback with correction layer (ensures no "Unknown")
    if matched:
        # Apply enhanced correction layer
        corrected_level = correctBloomLevel(matched_level, question)
        return corrected_level
    else:
        # Fallback - use correction layer to assign based on question patterns
        # This ensures questions are not marked as Unknown
        fallback_level = correctBloomLevel("Understanding", question)  # Default to Understanding, let correction layer fix it
        return fallback_level


def analyze_exam(questions, model, tokenizer, exam_name="Exam", progress_callback=None):
    """
    Complete exam analysis with ideal distribution comparison.
    Returns detailed analysis including missing, deficient, excessive, and balanced categories.
    
    Args:
        questions: List of question strings
        model: Loaded model (fine-tuned TinyLlama from notebook)
        tokenizer: Loaded tokenizer
        exam_name: Name of the exam (optional)
        progress_callback: Optional callback function(progress: float, status: str) for progress updates
    
    Returns:
        Dictionary with complete analysis results
    """
    import sys
    sys.stdout.flush()  # Ensure output is visible immediately
    
    if not questions:
        empty_result = {
            'exam_name': exam_name,
            'total_questions': 0,
            'predictions': [],
            'comparison': {
                level: dict(
                    actual=0,
                    ideal=IDEAL_DISTRIBUTION[level] * 100,
                    difference=-IDEAL_DISTRIBUTION[level] * 100,
                    count=0,
                )
                for level in BLOOM_LEVELS
            },
            'actual_pct': {level: 0 for level in BLOOM_LEVELS},
            'quality_score': 0,
            'quality_rating': "Poor",
            'missing': [],
            'deficient': [],
            'excessive': [],
            'balanced': [],
        }
        return empty_result
    
    # Use local fine-tuned TinyLlama model (from notebook training)
    if model is None or tokenizer is None:
        model, tokenizer = load_model(use_cpu_model=True)
    
    if progress_callback:
        progress_callback(0.1, "Preparing questions for analysis...")
    
    # Process in batches for large question sets to show progress
    # Use larger batches on GPU, smaller on CPU
    if torch.cuda.is_available() and device.type == "cuda":
        batch_size = 200  # Larger batches on GPU for faster processing
    else:
        batch_size = 100  # Smaller batches on CPU to avoid memory issues
    total_questions = len(questions)
    all_decoded = []
    
    # Get device - handle various model types
    try:
        if hasattr(model, 'device'):
            device = model.device
        elif hasattr(model, 'hf_device_map'):
            # For models with device_map, get the device of the first parameter
            device = next(model.parameters()).device
        else:
            # Try to get device from model parameters
            device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        # Fallback to CPU if device detection fails
        device = torch.device("cpu")
    
    # Process in batches for better progress feedback
    for batch_start in range(0, total_questions, batch_size):
        batch_end = min(batch_start + batch_size, total_questions)
        batch_questions = questions[batch_start:batch_end]
        
        if progress_callback:
            progress = 0.1 + (batch_start / total_questions) * 0.7
            progress_callback(progress, f"Analyzing questions {batch_start + 1}-{batch_end} of {total_questions}...")
        
        # Classify batch of questions
        prompts = [format_prompt(q) for q in batch_questions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,  # Increased to 8 to ensure complete level names (was causing "Under"/"App" truncation)
                temperature=0.1,
                do_sample=False,
            )
        
        batch_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_decoded.extend(batch_decoded)
    
    decoded = all_decoded
    
    if progress_callback:
        progress_callback(0.8, "Processing classification results...")
    predictions = []
    
    # Pre-compile level names in lowercase for faster matching
    level_lower_map = {level.lower(): level for level in BLOOM_LEVELS}
    
    for i, result in enumerate(decoded):
        question = questions[i] if i < len(questions) else f"Question {i+1}"
        matched = False
        matched_level = None
        
        # First, try to extract from text immediately after "Bloom Level:"
        if "Bloom Level:" in result:
            pred = result.split("Bloom Level:")[-1].strip().split('\n')[0].strip()
            # Filter out single digits that might be question numbers
            if not (pred.isdigit() and len(pred) == 1):
                pred_lower = pred.lower()
                # Original substring matching logic (was working before)
                for level_lower, level in level_lower_map.items():
                    if level_lower in pred_lower:  # Original substring matching
                        matched_level = level
                        matched = True
                        break
        
        # If no match found, search the entire result string for Bloom level names (original substring logic)
        if not matched:
            result_lower = result.lower()
            # Original logic: substring matching (this was working before)
            for level_lower, level in level_lower_map.items():
                if level_lower in result_lower:
                    matched_level = level
                    matched = True
                    break
        
        # If still no match, use fallback with correction layer (ensures no "Unknown")
        if matched:
            # Apply enhanced correction layer
            corrected_level = correctBloomLevel(matched_level, question)
            predictions.append(corrected_level)
        else:
            # Fallback - use correction layer to assign based on question patterns
            # This ensures questions are not marked as Unknown
            fallback_level = correctBloomLevel("Understanding", question)  # Default to Understanding, let correction layer fix it
            predictions.append(fallback_level)
    
    if progress_callback:
        progress_callback(0.9, "Calculating distribution and quality scores...")
    
    # Use Counter for O(n) counting instead of O(n*m) with .count()
    prediction_counter = Counter(predictions)
    actual_dist = {level: prediction_counter.get(level, 0) for level in BLOOM_LEVELS}
    unknown_count = prediction_counter.get("Unknown", 0)
    actual_pct = {
        level: 100 * actual_dist[level] / len(questions) if questions else 0
        for level in BLOOM_LEVELS
    }
    
    # Compare with ideal
    comparison = {}
    missing = []
    deficient = []
    excessive = []
    balanced = []
    total_deviation = 0
    
    for level in BLOOM_LEVELS:
        actual = actual_pct[level]
        ideal = IDEAL_DISTRIBUTION[level] * 100
        difference = actual - ideal
        total_deviation += abs(difference)
        
        comparison[level] = dict(
            actual=actual,
            ideal=ideal,
            difference=difference,
            count=actual_dist[level],
        )
        
        # Categorize levels
        if actual == 0:
            missing.append({
                'level': level,
                'ideal': ideal,
                'needed': int(len(questions) * IDEAL_DISTRIBUTION[level]),
                'action': f"ADD {int(len(questions) * IDEAL_DISTRIBUTION[level])} {level} questions"
            })
        elif actual < ideal * 0.7:  # Less than 70% of ideal
            deficient.append({
                'level': level,
                'actual': actual,
                'ideal': ideal,
                'gap': ideal - actual,
                'action': f"INCREASE {level} by {int((ideal - actual) / 100 * len(questions))} questions"
            })
        elif actual > ideal * 1.3:  # More than 130% of ideal
            excessive.append({
                'level': level,
                'actual': actual,
                'ideal': ideal,
                'excess': actual - ideal,
                'action': f"REDUCE {level} by {int((actual - ideal) / 100 * len(questions))} questions"
            })
        else:
            balanced.append({
                'level': level,
                'actual': actual,
                'ideal': ideal
            })
    
    # Calculate quality score
    quality_score = max(0, 100 - (total_deviation / 2))
    
    if quality_score >= 90:
        quality_rating = "Excellent"
    elif quality_score >= 80:
        quality_rating = "Good"
    elif quality_score >= 70:
        quality_rating = "Fair"
    elif quality_score >= 60:
        quality_rating = "Needs Improvement"
    else:
        quality_rating = "Poor"
    
    # Add Unknown to comparison if present
    if unknown_count > 0:
        comparison["Unknown"] = dict(
            actual=100 * unknown_count / len(questions) if questions else 0,
            ideal=0,  # Unknown has no ideal
            difference=100 * unknown_count / len(questions) if questions else 0,
            count=unknown_count,
        )
    
    return {
        'exam_name': exam_name,
        'total_questions': len(questions),
        'predictions': predictions,
        'questions': questions,  # Include original questions for preview
        'comparison': comparison,
        'actual_pct': actual_pct,
        'quality_score': quality_score,
        'quality_rating': quality_rating,
        'missing': missing,
        'deficient': deficient,
        'excessive': excessive,
        'balanced': balanced,
        'total_deviation': total_deviation,
    }
