
# Bloom Taxonomy Exam Analyzer
# Generated: 2025-11-13T13:26:03.587694

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_PATH = "/content/drive/My Drive/Blooms-Taxonomy-Project/models/final"
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

BLOOM_LEVELS = ['Remembering', 'Understanding', 'Applying', 'Analyzing', 'Evaluating', 'Creating']

IDEAL_DISTRIBUTION = {
    'Remembering': 0.15,
    'Understanding': 0.20,
    'Applying': 0.25,
    'Analyzing': 0.20,
    'Evaluating': 0.10,
    'Creating': 0.10
}

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base, MODEL_PATH)
    model.eval()
    
    return model, tokenizer

print("Bloom Taxonomy Analyzer Module Ready")
print("Load with: model, tokenizer = load_model()")
