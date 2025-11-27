"""
Optimized model loader for question generation.
Uses quantized, lightweight models for fast inference.
Separate from analysis models - this is only for generation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Optimized models for generation (separate from analysis)
GENERATION_MODEL_OPTIONS = {
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",  # Best quality, ~1.5B params, better instruction following
    "qwen2-0.5b": "Qwen/Qwen2-0.5B-Instruct",  # Fastest, ~0.5B params
    "gemma-2b": "google/gemma-2b-it",  # Fast, ~2B params, good quality
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",  # ~3.8B params (old default)
    "phi2": "microsoft/phi-2",  # Efficient, ~2.7B params
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Fallback, same as analysis
}

# Default to Qwen2.5-1.5B-Instruct (better for instruction following and question generation)
DEFAULT_GENERATION_MODEL = "qwen2.5-1.5b"

_generation_model = None
_generation_tokenizer = None
_generation_model_name = None


def load_generation_model(
    model_name: str = None,
    use_quantization: bool = False,  # Changed default: Qwen2.5-7B performs MUCH better unquantized
    quantization_bits: int = 4,
    use_cpu: bool = False,
    device_map: str = "auto"
):
    """
    Load a model for question generation.
    
    Args:
        model_name: Model name from GENERATION_MODEL_OPTIONS, or None for default (Qwen2.5-7B)
        use_quantization: Whether to use quantization (default: False for better quality)
        quantization_bits: 4 or 8 (4-bit is faster, 8-bit is better quality)
        use_cpu: Force CPU usage
        device_map: Device mapping strategy ("auto", "cpu", etc.)
    
    Returns:
        model, tokenizer tuple
    """
    global _generation_model, _generation_tokenizer, _generation_model_name
    
    # Use cached model if same model is requested
    selected_model = model_name or DEFAULT_GENERATION_MODEL
    if _generation_model is not None and _generation_model_name == selected_model:
        return _generation_model, _generation_tokenizer
    
    model_id = GENERATION_MODEL_OPTIONS.get(selected_model, GENERATION_MODEL_OPTIONS[DEFAULT_GENERATION_MODEL])
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # For Qwen2.5-7B, load without quantization initially (performs MUCH better)
        # Only use quantization if explicitly requested and GPU available
        quantization_config = None
        if use_quantization and not use_cpu and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                if quantization_bits == 4:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                    quantization_config = bnb_config
                elif quantization_bits == 8:
                    bnb_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    quantization_config = bnb_config
            except ImportError:
                quantization_config = None
            except Exception as e:
                quantization_config = None
        
        # Load model (prefer unquantized for Qwen2.5-7B)
        if quantization_config and not use_cpu:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map=device_map if not use_cpu else None,
                torch_dtype=torch.float16 if not use_cpu else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        else:
            # Load without quantization (recommended for Qwen2.5-7B)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map if not use_cpu else None,
                torch_dtype=torch.float16 if not use_cpu else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            if use_cpu:
                model = model.to("cpu")
        
        model.eval()
        
        # Cache the model
        _generation_model = model
        _generation_tokenizer = tokenizer
        _generation_model_name = selected_model
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading generation model {model_id}: {e}")
        print("Falling back to TinyLlama without quantization...")
        
        # Fallback to TinyLlama
        try:
            fallback_id = GENERATION_MODEL_OPTIONS["tinyllama"]
            tokenizer = AutoTokenizer.from_pretrained(fallback_id)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                fallback_id,
                torch_dtype=torch.float32 if use_cpu else torch.float16,
                device_map=device_map if not use_cpu else None,
                low_cpu_mem_usage=True,
            )
            if use_cpu:
                model = model.to("cpu")
            model.eval()
            
            _generation_model = model
            _generation_tokenizer = tokenizer
            _generation_model_name = "tinyllama"
            
            return model, tokenizer
        except Exception as fallback_error:
            raise Exception(f"Failed to load generation model and fallback: {str(e)}, {str(fallback_error)}")


def get_current_generation_model_name():
    """Return the name/ID of currently loaded generation model."""
    global _generation_model_name, _generation_model
    if _generation_model is not None and _generation_model_name:
        return _generation_model_name
    return None

