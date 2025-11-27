"""
RAG-based exam generator using vector stores and LLM.
Supports content-based and web search-based exam generation.
"""

import os
import json
import re
from typing import List, Dict, Optional, Any, Callable
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import torch
import time
from collections import deque, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.content_extractor import extract_and_chunk_content, get_chunk_summary

# Performance: Create a session with connection pooling for faster API calls
_session = None
def get_session():
    """Get or create a requests session with connection pooling."""
    global _session
    if _session is None:
        _session = requests.Session()
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=25, pool_maxsize=25)
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
    return _session

# Generation logging for debugging
_generation_logs = deque(maxlen=10)  # Keep last 10 generation logs


class RAGExamGenerator:
    """RAG-based exam generator using embeddings and LLM."""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/paraphrase-MiniLM-L3-v2",  # Faster embedding model
        llm_api: str = "gemini",  # "local" or "gemini" (free API)
        api_key: Optional[str] = None,
        use_local_vector_store: bool = True,
        local_model=None,
        local_tokenizer=None,
        use_optimized_generation: bool = True,  # Use optimized quantized model for generation
        generation_model_name: Optional[str] = None  # Specific model to use (None = use default)
    ):
        """
        Initialize RAG exam generator.
        
        Args:
            embedding_model: Hugging Face model name for embeddings (default: faster paraphrase-MiniLM-L3-v2)
            llm_api: Which LLM API to use ("local" uses optimized generation model, "gemini" - free API)
            api_key: API key for Gemini (free tier available)
            use_local_vector_store: Whether to use local ChromaDB or simple in-memory store
            local_model: Pre-loaded model for local generation (if None and use_optimized_generation=True, will load optimized model)
            local_tokenizer: Pre-loaded tokenizer for local generation
            use_optimized_generation: If True and local_model is None, load optimized quantized generation model
            generation_model_name: Specific model name to use (e.g., "phi3-mini", "qwen2-0.5b", "gemma-2b"). None = use default.
        """
        self.embedding_model = embedding_model
        self.llm_api = llm_api
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.use_local_vector_store = use_local_vector_store
        self.use_optimized_generation = use_optimized_generation
        self.generation_logs = []  # Instance-level logs
        self._analysis_model = None  # Cached analysis model for fallback
        self._analysis_tokenizer = None  # Cached tokenizer for fallback
        self._analysis_model_attached = False
        # Gemini model fallback chain - if one model hits rate limit, try next
        self._gemini_models = [
            "gemini-2.0-flash",
            "gemini-2.5-flash", 
            "gemini-2.0-flash-lite"
        ]
        self._current_gemini_model_index = 0  # Start with first model
        # Enable local fallback by default when using Gemini, so rate limits don't completely block exams.
        # Projects that truly want Gemini-only generation can later override this flag explicitly.
        self.allow_local_fallback = (llm_api == "gemini")
        
        # Load optimized generation model if requested and no model provided
        if llm_api == "local" and local_model is None and use_optimized_generation:
            try:
                from utils.generation_model_loader import load_generation_model
                model_name = generation_model_name or None  # None = use default (Qwen2.5-7B)
                self.local_model, self.local_tokenizer = load_generation_model(
                    model_name=model_name,  # Pass specific model name or None for default
                    use_quantization=False,  # Qwen2.5-7B performs better unquantized
                    quantization_bits=4,
                    use_cpu=not torch.cuda.is_available()
                )
                print("✓ Optimized generation model loaded")
            except Exception as e:
                print(f"Warning: Could not load optimized generation model: {e}")
                print("Falling back to provided model or will use analysis model if available")
                self.local_model = local_model
                self.local_tokenizer = local_tokenizer
        else:
            self.local_model = local_model
            self.local_tokenizer = local_tokenizer
        
        # Initialize vector store (simplified - can be upgraded to ChromaDB)
        self.chunks = []
        self.embeddings = None
        self.embedding_model_obj = None
        
        # Cache similarity search results to avoid redundant computations
        self._similarity_cache = {}  # Cache for similarity search results
        
        # Try to initialize embeddings
        try:
            self._init_embeddings()
        except Exception as e:
            print(f"Warning: Could not initialize embeddings: {e}. Will use simple text matching.")
    
    def _init_embeddings(self):
        """Initialize embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model_obj = SentenceTransformer(self.embedding_model)
            print(f"Loaded embedding model: {self.embedding_model}")
        except ImportError:
            print("sentence-transformers not installed. Using simple text matching.")
        except Exception as e:
            print(f"Could not load embedding model: {e}")
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts."""
        if self.embedding_model_obj:
            return self.embedding_model_obj.encode(texts, show_progress_bar=False).tolist()
        else:
            # Fallback: simple TF-IDF-like representation
            return [[hash(text) % 10000 / 10000] * 384 for text in texts]
    
    def _truncate_tokens(self, text: str, tokenizer, max_tokens: int = 1500) -> str:
        """Truncate text to max_tokens using tokenizer (not character-based)."""
        if not tokenizer:
            # Fallback to character-based if no tokenizer
            return text[:max_tokens * 4]  # Rough estimate: 4 chars per token
        
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) <= max_tokens:
                return text
            tokens = tokens[:max_tokens]
            return tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception as e:
            return text[:max_tokens * 4]
    
    def add_content(self, content_source: str, source_type: str = "text", metadata: Optional[Dict] = None):
        """
        Add content to the vector store.
        
        Args:
            content_source: Path to file or text content
            source_type: 'pdf', 'text_file', or 'text'
            metadata: Optional metadata about the content
        """
        # Extract and chunk content
        chunks = extract_and_chunk_content(content_source, source_type)
        
        if not chunks:
            raise ValueError("No content could be extracted from the source.")
        
        # Clear similarity cache when new content is added
        self._similarity_cache.clear()
        
        # Get embeddings for chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self._get_embeddings(chunk_texts)
        
        # Store chunks with metadata
        for i, chunk in enumerate(chunks):
            chunk_data = {
                'text': chunk['text'],
                'index': chunk['index'],
                'embedding': embeddings[i],
                'metadata': metadata or {}
            }
            self.chunks.append(chunk_data)
        
        print(f"Added {len(chunks)} chunks to vector store")
    
    def _similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks using cosine similarity (optimized with caching)."""
        if not self.chunks:
            return []
        
        # Check cache first (cache key includes query and top_k)
        cache_key = f"{query.lower().strip()}:{top_k}:{len(self.chunks)}"
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        # Limit search to reasonable number of chunks for speed (if too many chunks)
        # Reduced from 100 to 50 for faster search
        max_chunks_to_search = min(len(self.chunks), 50)  # Only search top 50 chunks for speed
        chunks_to_search = self.chunks[:max_chunks_to_search] if len(self.chunks) > 30 else self.chunks
        
        # Get query embedding
        query_embedding = self._get_embeddings([query])[0]
        query_emb = np.array(query_embedding, dtype=np.float32)  # Use float32 for speed
        norm_query = np.linalg.norm(query_emb)
        
        if norm_query == 0:
            result = chunks_to_search[:top_k]
            self._similarity_cache[cache_key] = result
            return result
        
        # Vectorized similarity calculation for speed
        similarities = []
        for chunk in chunks_to_search:
            chunk_emb = np.array(chunk['embedding'], dtype=np.float32)  # Use float32 for speed
            norm_chunk = np.linalg.norm(chunk_emb)
            
            if norm_chunk > 0:
                similarity = np.dot(chunk_emb, query_emb) / (norm_chunk * norm_query)
            else:
                similarity = 0.0
            
            similarities.append((similarity, chunk))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        result = [chunk for _, chunk in similarities[:top_k]]
        
        # Cache result (limit cache size to prevent memory issues)
        if len(self._similarity_cache) < 100:  # Limit cache to 100 entries
            self._similarity_cache[cache_key] = result
        else:
            # Clear cache if it gets too large (simple FIFO - remove oldest)
            oldest_key = next(iter(self._similarity_cache))
            del self._similarity_cache[oldest_key]
            self._similarity_cache[cache_key] = result
        
        return result
    
    def _generate_with_local_model(self, prompt: str, log_params: bool = False) -> str:
        """Generate text using local optimized model (quantized, fast)."""
        if not self.local_model or not self.local_tokenizer:
            raise ValueError("Local model not provided. Please pass model and tokenizer to RAGExamGenerator.")
        
        try:
            # Format prompt - try to detect if model uses chat template
            tokenizer = self.local_tokenizer
            model_name = getattr(tokenizer, 'name_or_path', '').lower()
            
            # Check if tokenizer has chat template (for instruction-tuned models)
            # For question generation, use simpler prompt format to avoid template leakage
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                # Try to use a simpler format without system message to reduce leakage
                # Some models leak the template, so use user message only
                try:
                    messages = [{"role": "user", "content": prompt}]
                    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    # If that fails, use full template but we'll extract carefully
                    messages = [
                        {"role": "system", "content": "Generate exam questions based on provided content."},
                        {"role": "user", "content": prompt}
                    ]
                    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # Fallback format for models without chat template (TinyLlama, etc.)
                # Use simpler format to avoid leakage
                full_prompt = prompt + "\n\nQuestion:"
            
            # Tokenize (reduced max_length for faster processing)
            inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=384)
            
            # Move to model device (handle quantized models)
            try:
                device = next(self.local_model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except (StopIteration, AttributeError):
                # Quantized models might not have .device attribute
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                else:
                    inputs = {k: v for k, v in inputs.items()}
            
            # Generate (increased tokens for complete questions)
            with torch.no_grad():
                generate_kwargs = {
                    "max_new_tokens": 120,  # Increased to ensure complete questions
                    "min_new_tokens": 20,  # Ensure minimum length for complete questions
                    "temperature": 0.7,  # Lower for more focused generation
                    "do_sample": True,
                    "top_p": 0.9,  # Slightly lower for better quality
                    "repetition_penalty": 1.2,  # Higher to avoid repetition
                    "num_beams": 3,  # Beam search for better quality
                    "early_stopping": False,  # Don't stop early - need full questions
                }
                
                # Try to add stop sequences using the model's supported method
                # Some models support 'stopping_criteria' or custom stop token handling
                # But we'll handle stopping in post-processing instead to avoid compatibility issues
                
                # Add pad_token_id if available
                if tokenizer.pad_token_id is not None:
                    generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
                elif tokenizer.eos_token_id is not None:
                    generate_kwargs["pad_token_id"] = tokenizer.eos_token_id
                
                # Try to use cache if available
                try:
                    generate_kwargs["use_cache"] = True
                except:
                    pass
                
                outputs = self.local_model.generate(**inputs, **generate_kwargs)
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # IMPORTANT: Extract ONLY the newly generated tokens (not the input prompt)
            # Get input length and extract only new tokens
            input_length = inputs['input_ids'].shape[1]
            generated_ids = outputs[0][input_length:]  # Only new tokens
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Extract question - try multiple strategies
            question = None
            
            # Strategy 1: Look for "Question:" marker
            if "Question:" in generated_text:
                parts = generated_text.split("Question:")
                if len(parts) > 1:
                    question = parts[-1].strip()
            
            # Strategy 2: Remove any prompt fragments that leaked through
            # Remove system/user/assistant markers aggressively
            import re
            question_text = generated_text
            # Remove chat template markers
            question_text = re.sub(r'^\s*(system|System|SYSTEM)\s*[:\-]?\s*', '', question_text, flags=re.IGNORECASE)
            question_text = re.sub(r'^\s*(user|User|USER)\s*[:\-]?\s*', '', question_text, flags=re.IGNORECASE)
            question_text = re.sub(r'^\s*(assistant|Assistant|ASSISTANT)\s*[:\-]?\s*', '', question_text, flags=re.IGNORECASE)
            # Remove prompt fragments like "You are an expert educator..."
            question_text = re.sub(r'You are an expert educator.*?Bloom.*?Taxonomy\.?\s*', '', question_text, flags=re.IGNORECASE | re.DOTALL)
            question_text = re.sub(r'You are an expert.*?creating.*?exam.*?\s*', '', question_text, flags=re.IGNORECASE | re.DOTALL)
            # Remove common prompt starters
            question_text = re.sub(r'^(Generate|Create|Write|Produce|Develop)\s+(a|an|the|one|the question)\s+', '', question_text, flags=re.IGNORECASE)
            # Remove phrases like "according to the text", "based on the content", etc.
            question_text = re.sub(r'\s*(according to|based on|as described|as mentioned|in the content|from the text|provided text|given text).*?[:\.,]?\s*', ' ', question_text, flags=re.IGNORECASE)
            question_text = re.sub(r'^(Based on|According to|Using|From)\s+(the|this|that)\s+(content|text|passage|material).*?:\s*', '', question_text, flags=re.IGNORECASE)
            question_text = re.sub(r'\s+', ' ', question_text).strip()
            
            # Strategy 3: Look for assistant response markers
            for marker in ["assistant", "Assistant", "Answer:", "Response:", "Question:"]:
                if marker in question_text:
                    parts = question_text.split(marker, 1)
                    if len(parts) > 1:
                        question_text = parts[-1].strip()
                        break
            
            # Strategy 4: Remove placeholder text and broken fragments
            question_text = re.sub(r'\{.*?\}', '', question_text)  # Remove {placeholders}
            question_text = re.sub(r'STRICT GENERATION RULES.*$', '', question_text, flags=re.IGNORECASE | re.DOTALL)
            question_text = re.sub(r'Generate ONLY.*$', '', question_text, flags=re.IGNORECASE | re.DOTALL)
            question_text = re.sub(r'Requirements.*$', '', question_text, flags=re.IGNORECASE | re.DOTALL)
            
            # Use the cleaned text
            question = question_text.strip() if question_text.strip() else generated_text.strip()
            
            # If we still have prompt-like text, try to extract just the question part
            if question and len(question) > 50 and ('system' in question.lower() or 'user' in question.lower() or 'You are' in question):
                # Try to find the actual question after prompt markers
                # Look for patterns like "What", "How", "Explain", "Describe" etc.
                question_starters = ['What', 'How', 'Why', 'When', 'Where', 'Who', 'Explain', 'Describe', 'Define', 'Compare', 'Analyze', 'Evaluate', 'Design', 'Develop', 'Create', 'List', 'Identify', 'State', 'Summarize']
                for starter in question_starters:
                    idx = question.find(starter)
                    if idx != -1 and idx < len(question) * 0.5:  # Question should start early in text
                        question = question[idx:].strip()
                        break
            
            # Clean up question
            if question:
                # Remove any leading/trailing whitespace
                question = question.strip()
                
                # Remove complexity tags like [INTERMEDIATE], [EASY], [COMPLEX] (aggressive removal)
                import re
                question = re.sub(r'\[(INTERMEDIATE|EASY|COMPLEX|intermediate|easy|complex)\]\s*', '', question, flags=re.IGNORECASE)
                question = re.sub(r'\[.*?INTERMEDIATE.*?\]', '', question, flags=re.IGNORECASE)
                question = re.sub(r'\[.*?EASY.*?\]', '', question, flags=re.IGNORECASE)
                question = re.sub(r'\[.*?COMPLEX.*?\]', '', question, flags=re.IGNORECASE)
                question = re.sub(r'\s*\[(INTERMEDIATE|EASY|COMPLEX)\]\s*', ' ', question, flags=re.IGNORECASE)
                question = re.sub(r'\s+', ' ', question).strip()
                
                # Remove common prefixes
                for prefix in ["Question:", "Answer:", "Response:", "Here's", "Here is", "The question is:", "Q:", "Q."]:
                    if question.lower().startswith(prefix.lower()):
                        question = question[len(prefix):].strip()
                
                # Remove quotes around the question
                question = question.strip('"').strip("'").strip()
                
                # Take only the first sentence/question (stop at newline)
                lines = question.split('\n')
                question = lines[0].strip()
                
                # Check for explanation patterns that indicate where the question ends
                explanation_indicators = [
                    ' this question', ' it requires', ' the question meets', ' it also', 
                    ' this ', ' it encourages', ' ensuring', ' specifically', ' along with',
                    ' followed by', ' endowing with', ' adheres to', ' and ends the question',
                    ' propose ways', ' enhance efficiency', ' reduce costs', ' within their organization'
                ]
                
                # Find the earliest explanation indicator
                earliest_explanation = None
                for indicator in explanation_indicators:
                    idx = question.lower().find(indicator)
                    if idx != -1 and (earliest_explanation is None or idx < earliest_explanation):
                        earliest_explanation = idx
                
                # If we found an explanation, cut the question there
                if earliest_explanation and earliest_explanation > 20:  # Make sure we have a substantial question
                    question = question[:earliest_explanation].strip()
                
                # If question contains a question mark, take everything up to and including the first one
                if '?' in question:
                    q_parts = question.split('?', 1)
                    question = q_parts[0].strip() + '?'
                    # Check if there's explanation text after the question mark
                    if len(q_parts) > 1:
                        after_q = q_parts[1].strip()
                        # If there's text after, check if it looks like an explanation
                        explanation_starters = ['this question', 'it requires', 'the question', 'it also', 'this', 'it encourages', 'ensuring', 'specifically', 'along with', 'followed by', 'endowing', 'adheres']
                        if any(after_q.lower().startswith(starter) for starter in explanation_starters):
                            # It's an explanation, we already have just the question
                            pass
                        elif len(after_q) > 20:
                            # Long text after question mark is likely explanation
                            pass
                elif len(question) > 10:  # If it's substantial but no question mark
                    # Only add question mark if it doesn't already have one and looks like a question
                    if '?' not in question:
                        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'explain', 'describe', 'define', 'compare', 'analyze', 'evaluate', 'create', 'design', 'identify', 'list', 'state', 'can', 'could', 'would', 'should']
                        if any(word in question.lower() for word in question_words):
                            # Check if it ends with a period and has explanation after
                            if question.endswith('.'):
                                # Check if there's explanation-like text before the period
                                sentences = question.split('.')
                                if len(sentences) > 1:
                                    # Take only the first sentence if it looks like a question
                                    first_sent = sentences[0].strip()
                                    if any(word in first_sent.lower() for word in question_words):
                                        question = first_sent + '?'
                                    else:
                                        question = question.rstrip('.!') + '?'
                                else:
                                    question = question.rstrip('.!') + '?'
                            else:
                                # Add question mark if it looks like a question
                                question = question.rstrip('.!') + '?'
                
                # Remove any trailing explanations or meta-commentary
                # Common patterns: "This question...", "It requires...", "The question meets..."
                explanation_patterns = [
                    r'\s+This question.*$',
                    r'\s+It requires.*$',
                    r'\s+The question meets.*$',
                    r'\s+It also requires.*$',
                    r'\s+This.*because.*$',
                    r'\s+It encourages.*$',
                    r'\s+ensuring that.*$',
                    r'\s+specifically.*$',
                    r'\s+along with.*$',
                    r'\s+followed by.*$',
                    r'\s+along with requirements.*$',
                    r'\s+endowing with.*$',
                    r'\s+adheres to.*$',
                    r'\s+and ends the question.*$',
                    r'\s+It also.*$',
                    r'\s+This.*meets.*$',
                    r'\s+The question.*criteria.*$',
                    r'\s+It.*consider.*$',
                    r'\s+propose ways.*$',
                    r'\s+enhance efficiency.*$',
                    r'\s+reduce costs.*$',
                    r'\s+increase customer.*$',
                    r'\s+improve overall.*$',
                    r'\s+within their organization.*$',
                    r'\s+maximizing profit.*$',
                    r'\s+delivering exceptional.*$'
                ]
                for pattern in explanation_patterns:
                    question = re.sub(pattern, '', question, flags=re.IGNORECASE).strip()
                
                # Additional cleanup: if question ends with explanation-like text, remove it
                # Look for sentences that start with capital letters after the question mark
                if '?' in question:
                    # Split and take only the part before the first question mark
                    parts = question.split('?')
                    if len(parts) > 1:
                        # Check if what comes after looks like an explanation
                        after = parts[1].strip()
                        if after and (len(after) > 30 or any(after.lower().startswith(word) for word in ['this', 'it', 'the', 'specifically', 'along', 'followed', 'ensuring'])):
                            question = parts[0].strip() + '?'
                
                # Final cleanup - remove any remaining quotes
                question = question.strip('"').strip("'").strip()
                
                # Final validation - ensure minimum length
                if len(question) < 15:
                    # Too short, likely incomplete - return None to trigger fallback in _generate_question
                    question = None
            
            # Return question or None if extraction failed (fallback handled in _generate_question)
            return question if question else None
            
        except Exception as e:
            # Return None on error - fallback will be handled in _generate_question
            print(f"Warning: Error generating question: {str(e)}")
            return None
    
    def _generate_with_gemini(self, prompt: str, log_params: bool = False) -> str:
        """Generate text using Google Gemini API with automatic model fallback on rate limits."""
        api_key = self.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            error_msg = "Gemini API key not provided. Set GEMINI_API_KEY environment variable."
            raise ValueError(error_msg)
        
        import time
        
        # Try each model in the fallback chain
        last_exception = None
        for model_index in range(self._current_gemini_model_index, len(self._gemini_models)):
            model_name = self._gemini_models[model_index]
            self._current_gemini_model_index = model_index
            
            try:
                return self._try_gemini_model(api_key, model_name, prompt, log_params)
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = (
                    "rate" in error_str or 
                    "quota" in error_str or 
                    "429" in error_str or
                    "resourceexhausted" in error_str or
                    "too many requests" in error_str
                )
                
                if is_rate_limit and model_index < len(self._gemini_models) - 1:
                    # Rate limit hit - try next model in fallback chain
                    next_model = self._gemini_models[model_index + 1]
                    last_exception = e
                    continue
                else:
                    # Not a rate limit, or last model - raise the exception
                    raise
        
        # If we exhausted all models, raise last exception
        if last_exception:
            raise Exception(f"All Gemini models exhausted due to rate limits. Last error: {str(last_exception)}")
        raise Exception("Failed to generate with Gemini API")
    
    def _try_gemini_model(self, api_key: str, model_name: str, prompt: str, log_params: bool = False) -> str:
        """Try generating with a specific Gemini model."""
        import time
        
        max_retries = 2  # Just 2 retries per model (fallback handles the rest)
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Gemini API endpoint
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                
                data = {
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 500  # Restored to 500 for complete, high-quality questions
                    }
                }
                
                # Use session with connection pooling for faster requests
                session = get_session()
                response = session.post(
                    url,
                    json=data,
                    timeout=30  # Restored to 30s for reliable API calls
                )
                
                # Handle rate limiting - raise to trigger model fallback
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        # Small delay before retry
                        time.sleep(1)
                        continue
                    # Raise to trigger fallback to next model
                    raise Exception(f"Gemini API rate limit exceeded (429) for model {model_name}.")
                
                response.raise_for_status()
                result = response.json()
                
                # Extract generated text from Gemini response
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    # Check for finishReason - if it's SAFETY or other, there might be an issue
                    finish_reason = candidate.get("finishReason", "UNKNOWN")
                    
                    if finish_reason not in ["STOP", "MAX_TOKENS"]:
                        if finish_reason == "SAFETY":
                            raise Exception("Gemini API blocked content due to safety filters. Try a different prompt.")
                        elif finish_reason == "RECITATION":
                            raise Exception("Gemini API detected recitation. Try a different prompt.")
                    
                    if "content" in candidate:
                        if "parts" in candidate["content"] and len(candidate["content"]["parts"]) > 0:
                            generated_text = candidate["content"]["parts"][0]["text"].strip()
                            
                            return generated_text
                        else:
                            raise Exception("Candidate has content but no parts or empty parts")
                    else:
                        raise Exception("Candidate has no 'content' field")
                else:
                    pass
                
                # If we get here, check for error in response
                if "error" in result:
                    error_msg = result["error"].get("message", "Unknown error")
                    raise Exception(f"Gemini API error: {error_msg}")
                raise Exception(f"No generated text in Gemini API response: {result}")
                
            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < max_retries - 1:
                    continue
                raise Exception(f"Gemini API timeout after {max_retries} attempts.")
            except requests.exceptions.HTTPError as e:
                last_exception = e
                # response should be available from the try block
                response = getattr(e, 'response', None)
                error_detail = ""
                status_code = 0
                
                if response is not None:
                    try:
                        error_json = response.json()
                        error_detail = json.dumps(error_json, indent=2)
                        status_code = response.status_code
                    except:
                        error_detail = response.text if hasattr(response, 'text') else str(e)
                        status_code = response.status_code if hasattr(response, 'status_code') else 0
                else:
                    error_detail = str(e)
                
                if status_code == 429:
                    # Rate limit - raise to trigger model fallback
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    raise Exception(f"Gemini API rate limit exceeded (429) for model {model_name}.")
                else:
                    # Non-rate-limit errors - fail immediately
                    raise Exception(f"Error generating with Gemini API (HTTP {status_code}): {error_detail}")
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check if it's a rate limit error
                is_rate_limit = (
                    "rate" in error_str or 
                    "quota" in error_str or 
                    "429" in error_str or
                    "resourceexhausted" in error_str or
                    "too many requests" in error_str
                )
                
                # If it's not a rate-limit error, fail immediately
                if not is_rate_limit:
                    raise
                
                # Rate limit detected - retry once, then raise to trigger model fallback
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                
                # Re-raise to trigger fallback to next model
                raise Exception(f"Gemini API rate limit exceeded for model {model_name}. Last error: {str(e)}")
        
        # Should never reach here, but just in case
        if last_exception:
            raise Exception(f"Gemini API failed after {max_retries} attempts with model {model_name}: {str(last_exception)}")
        raise Exception(f"Gemini API failed with model {model_name}")
    
    def _ensure_local_model_available(self) -> bool:
        """Ensure a local model (or analysis model) is ready for fallback generation."""
        if self.local_model and self.local_tokenizer:
            return True
        
        fallback_model = getattr(self, "_analysis_model", None)
        fallback_tokenizer = getattr(self, "_analysis_tokenizer", None)
        
        if fallback_model and fallback_tokenizer:
            if not getattr(self, "_analysis_model_attached", False):
                self._analysis_model_attached = True
            self.local_model = fallback_model
            self.local_tokenizer = fallback_tokenizer
            return True
        
        return False
    
    def _parse_topics(self, topic_string: str) -> List[str]:
        """
        Parse multiple topics from a topic string.
        Handles formats like: "C++ and Python, Autoencoders, GANs" or "Topic1, Topic2, Topic3"
        Returns a list of individual topics.
        """
        if not topic_string or not topic_string.strip():
            return []
        
        import re
        # Split by comma, "and", or "&"
        # Remove common prefixes like "Topics:", "Subject:", etc.
        topic_string = re.sub(r'^(topics?|subjects?|areas?):\s*', '', topic_string.strip(), flags=re.IGNORECASE)
        
        # Split by comma, " and ", " & ", or just "and" (with spaces)
        topics = re.split(r',\s*|\s+and\s+|\s+&\s+', topic_string)
        
        # Clean each topic
        cleaned_topics = []
        for topic in topics:
            topic = topic.strip()
            # Remove trailing punctuation
            topic = re.sub(r'[.,;]+$', '', topic)
            if topic and len(topic) > 1:
                cleaned_topics.append(topic)
        
        # If only one topic after cleaning, return it
        # If multiple topics, return all
        return cleaned_topics if len(cleaned_topics) > 1 else [topic_string.strip()] if topic_string.strip() else []
    
    def _clean_question_final(self, question: str, topic: str = None) -> str:
        """Final aggressive cleanup of question text - removes tags, explanations, placeholders."""
        if not question:
            return question
        
        import re
        
        # STEP 1: Remove ALL complexity tags (ULTRA AGGRESSIVE - multiple passes)
        # Try every possible pattern variation
        for _ in range(3):  # Multiple passes to catch nested/overlapping patterns
            question = re.sub(r'\[(INTERMEDIATE|EASY|COMPLEX|intermediate|easy|complex)\]\s*', '', question, flags=re.IGNORECASE)
            question = re.sub(r'\[.*?INTERMEDIATE.*?\]', '', question, flags=re.IGNORECASE)
            question = re.sub(r'\[.*?EASY.*?\]', '', question, flags=re.IGNORECASE)
            question = re.sub(r'\[.*?COMPLEX.*?\]', '', question, flags=re.IGNORECASE)
            question = re.sub(r'\s*\[(INTERMEDIATE|EASY|COMPLEX)\]\s*', ' ', question, flags=re.IGNORECASE)
            question = re.sub(r'^\s*\[(INTERMEDIATE|EASY|COMPLEX)\]\s*', '', question, flags=re.IGNORECASE)
            question = re.sub(r'^(INTERMEDIATE|EASY|COMPLEX)\]\s*', '', question, flags=re.IGNORECASE)
            # Also remove if it appears without brackets
            question = re.sub(r'^\s*(INTERMEDIATE|EASY|COMPLEX)\s+', '', question, flags=re.IGNORECASE)
        
        # STEP 2: Remove "Explanation:", "Example:", "Note:" patterns and everything after them
        question = re.sub(r'\s*Explanation:\s*.*$', '', question, flags=re.IGNORECASE)
        question = re.sub(r'\s*Example:\s*.*$', '', question, flags=re.IGNORECASE)
        question = re.sub(r'\s*Note:\s*.*$', '', question, flags=re.IGNORECASE)
        question = re.sub(r'\s*\(Note:.*?\)', '', question, flags=re.IGNORECASE)
        question = re.sub(r'\s*Explanation\s*.*$', '', question, flags=re.IGNORECASE)
        question = re.sub(r'\s*Example\s*.*$', '', question, flags=re.IGNORECASE)
        question = re.sub(r'\s*Please provide.*$', '', question, flags=re.IGNORECASE)
        question = re.sub(r'\s*Provide details.*$', '', question, flags=re.IGNORECASE)
        question = re.sub(r'\s*Create a question.*$', '', question, flags=re.IGNORECASE)
        question = re.sub(r'\s*- Create.*$', '', question, flags=re.IGNORECASE)
        question = re.sub(r'\s*including steps.*$', '', question, flags=re.IGNORECASE)
        
        # STEP 3: Replace placeholder text like [topic], [concept], [technology name], etc.
        if topic:
            question = re.sub(r'\[topic\]', topic, question, flags=re.IGNORECASE)
            question = re.sub(r'\[concept\]', topic, question, flags=re.IGNORECASE)
            question = re.sub(r'\[subject\]', topic, question, flags=re.IGNORECASE)
        # Replace generic placeholders with generic terms
        question = re.sub(r'\[technology name\]', 'technology', question, flags=re.IGNORECASE)
        question = re.sub(r'\[.*?name\]', 'technology', question, flags=re.IGNORECASE)
        question = re.sub(r'\[.*?\]', '', question)  # Remove any remaining placeholders
        
        # Fix incomplete placeholders like "'s favorite" without context
        if topic:
            # Replace patterns like "'s favorite" with proper context
            question = re.sub(r"'s favorite", f"{topic}'s", question, flags=re.IGNORECASE)
            question = re.sub(r"'s", "s", question)  # Remove standalone possessive that might be incomplete
            # But preserve common contractions
            question = re.sub(r'\bit s\b', "it's", question, flags=re.IGNORECASE)
            question = re.sub(r'\bthat s\b', "that's", question, flags=re.IGNORECASE)
            question = re.sub(r'\bwhat s\b', "what's", question, flags=re.IGNORECASE)
            question = re.sub(r'\bthere s\b', "there's", question, flags=re.IGNORECASE)
            question = re.sub(r'\bhere s\b', "here's", question, flags=re.IGNORECASE)
        
        # STEP 4: Remove quotes
        question = question.strip('"').strip("'").strip()
        
        # STEP 5: Remove common prefixes
        prefixes_to_remove = ["Question:", "Answer:", "Response:", "Here's", "Here is", "The question is:", "Q:", "Q.", "[INTERMEDIATE]", "[EASY]", "[COMPLEX]", "INTERMEDIATE]", "EASY]", "COMPLEX]"]
        for prefix in prefixes_to_remove:
            if question.lower().startswith(prefix.lower()):
                question = question[len(prefix):].strip()
        
        # STEP 6: If question mark exists, take ONLY up to first question mark (remove everything after)
        # Also limit question length to 150 characters max (truncate if too long)
        if '?' in question:
            parts = question.split('?', 1)
            question = parts[0].strip()
            # Limit length before adding question mark
            if len(question) > 150:
                # Try to truncate at a sentence boundary
                truncated = question[:150]
                last_period = truncated.rfind('.')
                last_comma = truncated.rfind(',')
                last_space = truncated.rfind(' ')
                # Use the last punctuation or space
                cut_point = max(last_period, last_comma, last_space)
                if cut_point > 100:  # Only if we have substantial text
                    question = question[:cut_point].strip()
                else:
                    question = question[:150].strip()
            question = question + '?'
            # Everything after first ? is removed (explanations, examples, etc.)
        elif len(question) > 150:
            # No question mark but too long - truncate
            truncated = question[:150]
            last_period = truncated.rfind('.')
            last_comma = truncated.rfind(',')
            last_space = truncated.rfind(' ')
            cut_point = max(last_period, last_comma, last_space)
            if cut_point > 100:
                question = question[:cut_point].strip() + '?'
            else:
                question = question[:150].strip() + '?'
        
        # STEP 7: Remove explanation patterns that might appear before the question mark
        explanation_patterns = [
            r'\s+This question.*$',
            r'\s+It requires.*$',
            r'\s+The question meets.*$',
            r'\s+This.*because.*$',
            r'\s+It encourages.*$',
            r'\s+specifically.*$',
            r'\s+along with.*$',
            r'\s+followed by.*$',
            r'\s+endowing with.*$',
            r'\s+adheres to.*$',
            r'\s+and ends the question.*$',
            r'\s+It also.*$',
            r'\s+propose ways.*$',
            r'\s+enhance efficiency.*$',
            r'\s+reduce costs.*$',
            r'\s+increase customer.*$',
            r'\s+improve overall.*$',
            r'\s+within their organization.*$',
            r'\s+maximizing profit.*$',
            r'\s+delivering exceptional.*$',
            r'\s+consider their current.*$',
            r'\s+utilize IT solutions.*$',
            r'\s+think creatively.*$',
            r'\s+ensuring that each step.*$',
            r'\s+The task requires.*$',
            r'\s+Since.*$',
            r'\s+What advancements.*$',
            r'\s+Please provide.*$',
            r'\s+Provide details.*$',
            r'\s+Create a question.*$',
            r'\s+- Create.*$',
            r'\s+including steps.*$',
            r'\s+such as.*$',  # Remove long lists like "such as X, Y, Z, etc"
        ]
        for pattern in explanation_patterns:
            question = re.sub(pattern, '', question, flags=re.IGNORECASE).strip()
        
        # STEP 8: Clean up whitespace
        question = re.sub(r'\s+', ' ', question).strip()
        
        # STEP 9: Remove any remaining quotes
        question = question.strip('"').strip("'").strip()
        
        # STEP 10: Only add question mark if question doesn't already end with proper punctuation
        # AND it looks like a question (has question words)
        # Only check if question is complete, don't force question mark
        if question and len(question.strip()) < 10:
            question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'explain', 'describe', 'define', 'compare', 'analyze', 'evaluate', 'create', 'design', 'identify', 'list', 'state', 'develop', 'can', 'could', 'would', 'should', 'examine', 'given']
            if any(word in question.lower() for word in question_words):
                question = question.rstrip('.!') + '?'
        
        return question.strip()
    
    def _extract_key_terms(self, text: str, max_terms: int = 5) -> List[str]:
        """
        Extract key terms using KeyBERT (better than regex).
        Rejects function words, fragments, and stopwords.
        """
        # Use KeyBERT for proper keyword extraction (rejects "type of", "defined as", etc.)
        try:
            from keybert import KeyBERT
            # Use a lightweight embedding model for speed
            try:
                kw_model = KeyBERT('all-MiniLM-L6-v2')  # Fast, good quality
            except:
                kw_model = KeyBERT()  # Fallback to default
            
            keywords = kw_model.extract_keywords(
                text[:2000],  # Process first 2000 chars for speed
                keyphrase_ngram_range=(1, 2),  # 1-2 word phrases
                stop_words='english',  # Remove stopwords
                top_n=max_terms * 2,  # Get more to filter
                use_mmr=True,  # Maximum Marginal Relevance for diversity
                diversity=0.7
            )
            
            # Extract terms and filter out function words
            function_words = {
                'type', 'of', 'integral', 'to', 'defined', 'as', 'greater', 'than', 'less', 'the',
                'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
                'this', 'that', 'these', 'those', 'it', 'they', 'we', 'you', 'i', 'in', 'on', 'at',
                'for', 'with', 'by', 'from', 'to', 'about', 'into', 'through', 'during', 'before'
            }
            
            valid_terms = []
            for keyword, score in keywords:
                # Skip function words and fragments
                words = keyword.lower().split()
                
                # Reject if all words are function words
                if all(w in function_words for w in words):
                    continue
                
                # Reject if keyword contains only function words
                if keyword.lower() in function_words:
                    continue
                
                # Reject single-letter or very short meaningless terms
                if len(keyword.strip()) < 3:
                    continue
                
                # Reject fragments like "type of", "defined as", "integral to"
                if any(fragment in keyword.lower() for fragment in ['type of', 'defined as', 'integral to', 'greater than', 'less than']):
                    continue
                
                # Accept if it has at least one meaningful word
                meaningful_words = [w for w in words if w not in function_words]
                if meaningful_words:
                    valid_terms.append(keyword.strip())
                    if len(valid_terms) >= max_terms:
                        break
            
            if valid_terms:
                return valid_terms
        
        except (ImportError, Exception):
            # Fallback to regex-based extraction if KeyBERT is not available
            pass
        
        # Improved regex-based extraction (better than before)
        import re
        # Only extract capitalized proper nouns and phrases (not mid-sentence fragments)
        # Pattern: Start of sentence OR after punctuation, then capitalized word(s)
        capitalized_phrases = re.findall(r'(?:^|\.\s+|:\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})', text[:1500])
        
        # Extract quoted terms
        quoted_terms = re.findall(r'"([^"]{4,40})"', text[:1500])
        
        # Filter out function words and fragments
        function_words = {'type', 'of', 'integral', 'to', 'defined', 'as', 'greater', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'this', 'that'}
        
        valid_terms = []
        seen = set()
        
        for term in capitalized_phrases + quoted_terms:
            term = term.strip()
            words = term.lower().split()
            
            # Reject if all words are function words
            if all(w in function_words for w in words):
                continue
            
            # Reject fragments
            if term.lower() in function_words:
                continue
            
            # Reject if contains function word fragments
            if any(frag in term.lower() for frag in ['type of', 'defined as', 'integral to']):
                continue
            
            if term.lower() not in seen and len(term) > 3:
                seen.add(term.lower())
                valid_terms.append(term)
                if len(valid_terms) >= max_terms:
                    break
        
        return valid_terms if valid_terms else ["concept"]
    
    def _check_question_similarity(self, question1: str, question2: str, threshold: float = 0.7) -> bool:
        """
        Check if two questions are similar (duplicates or near-duplicates).
        Returns True if questions are similar (above threshold), False otherwise.
        
        Uses word overlap and key phrase matching for lightweight similarity checking.
        Enhanced to detect semantic redundancy across different Bloom levels.
        """
        if not question1 or not question2:
            return False
        
        q1_lower = question1.lower().strip()
        q2_lower = question2.lower().strip()
        
        # Exact match (after normalization)
        if q1_lower == q2_lower:
            return True
        
        # Remove punctuation and normalize whitespace
        import re
        q1_words = set(re.findall(r'\b\w+\b', q1_lower))
        q2_words = set(re.findall(r'\b\w+\b', q2_lower))
        
        # Remove common stopwords for better comparison
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
                     'should', 'may', 'might', 'must', 'this', 'that', 'these', 'those', 
                     'it', 'they', 'we', 'you', 'i', 'in', 'on', 'at', 'for', 'with', 'by', 
                     'from', 'to', 'of', 'about', 'into', 'through', 'during', 'before', 
                     'after', 'how', 'what', 'when', 'where', 'why', 'which', 'who'}
        
        q1_words = q1_words - stopwords
        q2_words = q2_words - stopwords
        
        if not q1_words or not q2_words:
            return False
        
        # Calculate Jaccard similarity (intersection over union)
        intersection = len(q1_words & q2_words)
        union = len(q1_words | q2_words)
        jaccard = intersection / union if union > 0 else 0
        
        # Also check for key phrase overlap (longer phrases are more significant)
        # Extract 2-3 word phrases
        q1_phrases = set()
        q1_words_list = list(q1_words)
        for i in range(len(q1_words_list) - 1):
            q1_phrases.add(f"{q1_words_list[i]} {q1_words_list[i+1]}")
        for i in range(len(q1_words_list) - 2):
            q1_phrases.add(f"{q1_words_list[i]} {q1_words_list[i+1]} {q1_words_list[i+2]}")
        
        q2_phrases = set()
        q2_words_list = list(q2_words)
        for i in range(len(q2_words_list) - 1):
            q2_phrases.add(f"{q2_words_list[i]} {q2_words_list[i+1]}")
        for i in range(len(q2_words_list) - 2):
            q2_phrases.add(f"{q2_words_list[i]} {q2_words_list[i+1]} {q2_words_list[i+2]}")
        
        phrase_overlap = len(q1_phrases & q2_phrases)
        max_phrases = max(len(q1_phrases), len(q2_phrases))
        phrase_similarity = phrase_overlap / max_phrases if max_phrases > 0 else 0
        
        # Combined similarity score (weighted: 60% word overlap, 40% phrase overlap)
        combined_similarity = 0.6 * jaccard + 0.4 * phrase_similarity
        
        # Enhanced: Check for semantic redundancy across Bloom levels
        # Map semantically similar verbs (e.g., "discuss" vs "dissect", "explain" vs "paraphrase")
        semantic_verb_groups = [
            ['discuss', 'dissect', 'examine', 'analyze', 'explore'],
            ['explain', 'paraphrase', 'describe', 'clarify', 'illustrate'],
            ['compare', 'contrast', 'differentiate', 'distinguish'],
            ['evaluate', 'assess', 'appraise', 'judge', 'critique'],
            ['create', 'design', 'develop', 'formulate', 'produce', 'construct']
        ]
        
        # Extract key verbs from both questions
        key_verbs = ['explain', 'describe', 'define', 'compare', 'analyze', 'evaluate', 
                     'create', 'design', 'implement', 'demonstrate', 'calculate', 'identify', 
                     'list', 'state', 'examine', 'differentiate', 'justify', 'assess',
                     'discuss', 'dissect', 'paraphrase', 'contrast', 'distinguish', 'appraise',
                     'judge', 'critique', 'develop', 'formulate', 'produce', 'construct']
        
        q1_verbs = [v for v in key_verbs if v in q1_lower]
        q2_verbs = [v for v in key_verbs if v in q2_lower]
        
        # Check if verbs are semantically similar (same group)
        verbs_semantically_similar = False
        if q1_verbs and q2_verbs:
            for verb_group in semantic_verb_groups:
                if any(v in verb_group for v in q1_verbs) and any(v in verb_group for v in q2_verbs):
                    verbs_semantically_similar = True
                    break
            # Also check exact verb match
            if any(v in q1_verbs for v in q2_verbs):
                verbs_semantically_similar = True
        
        # If semantically similar verbs AND high word/phrase overlap, likely redundant
        # This catches cases like "Discuss factors..." vs "Dissect factors..." (same topic, different Bloom level)
        if verbs_semantically_similar and combined_similarity > 0.5:
            # Check if core topic/subject is the same (high word overlap suggests same topic)
            if jaccard > 0.4 or phrase_similarity > 0.3:
                combined_similarity += 0.25  # Strong boost for semantic redundancy
        elif q1_verbs and q2_verbs and q1_verbs[0] == q2_verbs[0] and combined_similarity > 0.5:
            combined_similarity += 0.2  # Boost similarity if same verb
        
        return combined_similarity >= threshold
    
    def _is_duplicate_question(self, new_question: str, previous_questions: List[str], threshold: float = 0.7) -> bool:
        """
        Check if a new question is a duplicate of any previous question.
        Returns True if duplicate found, False otherwise.
        """
        if not new_question or not previous_questions:
            return False
        
        for prev_q in previous_questions:
            if self._check_question_similarity(new_question, prev_q, threshold):
                return True
        
        return False
    
    def _validate_question_uses_content(self, question: str, context: str, quick_check: bool = False) -> bool:
        """Validate that a generated question actually references terms from the content."""
        if not question or not context:
            return False
        
        # Skip validation if question is obviously incomplete/malformed
        if len(question) < 15 or question.endswith((' an', ' a', ' the', ' if', ' how', ' what', ' when', ' where')):
            return False
        
        # For very short contexts (web search, sparse content), be more lenient
        context_is_sparse = len(context) < 500
        
        question_lower = question.lower()
        
        # Quick check - just look for capitalized terms in first 1000 chars (faster)
        if quick_check:
            import re
            # Quick check: look for capitalized words in first part of context
            context_caps = re.findall(r'\b([A-Z][a-z]{4,})\b', context[:1000])
            for cap_term in context_caps[:5]:  # Only check first 5 (faster)
                if len(cap_term) > 4 and cap_term.lower() in question_lower:
                    return True
            # If quick check finds nothing, be lenient for API-generated questions
            return len(question) > 20  # Accept if question is reasonably long
        
        # Full validation (slower but more thorough)
        # Extract key terms from context
        context_terms = self._extract_key_terms(context, max_terms=10)  # Reduced from 15 to 10
        
        # Check for exact term matches (prefer longer terms)
        for term in sorted(context_terms, key=len, reverse=True):  # Check longer terms first
            if len(term) > 3 and term.lower() in question_lower:
                return True  # Return immediately on first match
        
        # Also check for partial matches of important capitalized terms
        import re
        context_caps = re.findall(r'\b([A-Z][a-z]{4,}(?:\s+[A-Z][a-z]+)?)\b', context[:1000])  # Reduced from 1500 to 1000
        unique_caps = list(dict.fromkeys(context_caps))  # Remove duplicates while preserving order
        for cap_term in unique_caps[:5]:  # Reduced from 10 to 5
            if len(cap_term) > 4 and cap_term.lower() in question_lower:
                return True
        
        # If no matches found but question looks complete and natural, be lenient
        # (some questions might use paraphrased terms)
        if len(question) > 20:
            # Check if question has specific words that suggest it's about content
            content_indicators = ['according to', 'in the content', 'as described', 'as mentioned', 'based on']
            if any(indicator in question_lower for indicator in content_indicators):
                return True
        
        # For sparse contexts (web search), be more lenient - accept if question is well-formed
        # and has reasonable length, even without exact term matches
        if context_is_sparse and len(question) > 25 and '?' in question:
            # Check if question has any substantive content (not just generic words)
            substantive_words = [w for w in question.split() if len(w) > 4 and w.lower() not in ['what', 'when', 'where', 'which', 'would', 'could', 'should']]
            if len(substantive_words) >= 3:
                return True
        
        return False
    
    def _extract_verb_from_question(self, question: str, bloom_level: str) -> Optional[str]:
        """
        Extract the main verb from a question for verb variety tracking.
        Returns the verb if found in the verb pool for this Bloom level, None otherwise.
        """
        import re
        question_lower = question.lower().strip()
        
        # Verb pools for each Bloom level
        verb_pools = {
            'Remembering': ['define', 'list', 'name', 'identify', 'recall', 'state', 'label', 'recognize', 'select', 'match', 'memorize', 'repeat', 'reproduce', 'retrieve', 'locate'],
            'Understanding': ['explain', 'describe', 'summarize', 'interpret', 'discuss', 'clarify', 'paraphrase', 'classify', 'compare', 'contrast', 'exemplify', 'illustrate', 'outline', 'relate', 'translate'],
            'Applying': ['apply', 'use', 'demonstrate', 'implement', 'solve', 'calculate', 'execute', 'illustrate', 'utilize', 'employ', 'practice', 'operate', 'show', 'sketch', 'construct'],
            'Analyzing': ['compare', 'analyze', 'examine', 'differentiate', 'distinguish', 'contrast', 'dissect', 'investigate', 'categorize', 'organize', 'deconstruct', 'break down', 'separate', 'test', 'question'],
            'Evaluating': ['evaluate', 'assess', 'judge', 'critique', 'justify', 'defend', 'argue', 'appraise', 'prioritize', 'rate', 'rank', 'recommend', 'select', 'support', 'validate'],
            'Creating': ['design', 'develop', 'construct', 'formulate', 'create', 'plan', 'compose', 'synthesize', 'invent', 'produce', 'generate', 'build', 'assemble', 'devise', 'originate']
        }
        
        verb_pool = verb_pools.get(bloom_level, [])
        
        # Try to find verb from question patterns
        # Pattern 1: Question word + verb (e.g., "What is...", "How would you apply...")
        verb_match = re.search(r'^(what|how|why|when|where|who|which|can|could|would|should|do|does|did|is|are|was|were)\s+(\w+)', question_lower)
        if verb_match:
            verb = verb_match.group(2)
            if verb in verb_pool:
                return verb
        
        # Pattern 2: Imperative form (verb at start)
        first_word = question_lower.split()[0] if question_lower.split() else ''
        if first_word in verb_pool:
            return first_word
        
        # Pattern 3: Find any verb from pool in the question
        for verb in verb_pool:
            # Check for verb as whole word (not substring)
            if re.search(r'\b' + re.escape(verb) + r'\b', question_lower):
                return verb
        
        return None
    
    def _generate_fallback_question(self, terms: List[str], bloom_level: str) -> str:
        """
        Generate fallback question with validation.
        Rejects function words and ensures terms are valid concepts.
        """
        # Filter out function words from terms
        function_words = {'type', 'of', 'integral', 'to', 'defined', 'as', 'greater', 'less', 'than', 'the', 'a', 'an', 'is', 'are', 'was', 'were'}
        valid_terms = []
        for term in terms:
            words = term.lower().split()
            # Reject if all words are function words
            if not all(w in function_words for w in words):
                # Reject if term itself is a function word
                if term.lower() not in function_words:
                    # Reject fragments
                    if not any(frag in term.lower() for frag in ['type of', 'defined as', 'integral to']):
                        valid_terms.append(term)
        
        if not valid_terms:
            return "Explain a key concept from the content."
        
        t1 = valid_terms[0]
        t2 = valid_terms[1] if len(valid_terms) > 1 else None
        
        if bloom_level == "Analyzing":
            if t2 and len(t2) > 3:  # Ensure t2 is valid
                return f"Compare {t1} and {t2}."
            else:
                # Don't use incomplete comparison
                return f"Analyze the key characteristics of {t1}."
        
        if bloom_level == "Remembering":
            return f"What is {t1}?"
        
        if bloom_level == "Understanding":
            return f"Explain the concept of {t1} using the details from the content."
        
        if bloom_level == "Applying":
            return f"How can {t1} be applied in a practical scenario?"
        
        if bloom_level == "Evaluating":
            return f"Evaluate the importance of {t1}."
        
        if bloom_level == "Creating":
            return f"Design something using {t1}."
        
        return f"Explain {t1}."
    
    def _is_valid_question(self, question: str) -> bool:
        """
        Enhanced validation - reject incomplete, generic, placeholder, and function-word questions.
        Rejects questions containing function words as "concepts".
        """
        if not question or len(question.strip()) < 10:
            return False
        
        # Must end with question mark
        # Don't force question mark - natural questions may end with period
        if not question or len(question.strip()) < 10:
            return False
        
        # Must have minimum word count (at least 7 meaningful words)
        words = question.split()
        if len(words) < 7:
            return False
        
        question_lower = question.lower()
        
        # Function words that should NEVER appear as "concepts" in questions
        function_words_as_concepts = [
            "type of", "defined as", "integral to", "greater than", "less than",
            "greater", "less", "type", "of", "defined", "as", "integral", "to",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "the", "a", "an", "this", "that", "these", "those", "it", "they", "we", "you"
        ]
        
        # Reject if question contains function words as standalone concepts
        for func_word in function_words_as_concepts:
            # Check if function word appears as a standalone concept (not part of larger phrase)
            pattern = r'\b' + re.escape(func_word) + r'\b'
            if re.search(pattern, question_lower):
                # But allow if it's part of a legitimate phrase
                legitimate_phrases = ["what is", "how is", "when is", "where is", "why is"]
                is_legitimate = any(legit in question_lower for legit in legitimate_phrases)
                if not is_legitimate and func_word in function_words_as_concepts[:8]:  # Check first 8 (the bad ones)
                    return False
        
        # Reject bad phrases (generic, incomplete, placeholder)
        bad_phrases = [
            "based on the content",
            "what is what is",  # Repeated
            "with what is",  # Incomplete comparison
            "compare with what",  # Incomplete
            "how can type of",  # Function word as concept
            "how can integral to",  # Function word as concept
            "how can defined as",  # Function word as concept
            "analyze the key characteristics of greater",  # Fragment
            "design something using defined as",  # Fragment
            "{",  # Placeholder
            "}",  # Placeholder
        ]
        
        for phrase in bad_phrases:
            if phrase in question_lower:
                return False
        
        # Reject if too generic (need at least 4 meaningful words)
        generic_words = ["the", "is", "what", "how", "why", "when", "where", "a", "an", "of", "to", "in", "on", "at", "for", "with", "by", "from"]
        meaningful_words = [w for w in words if w.lower() not in generic_words and len(w) > 2]
        if len(meaningful_words) < 4:  # Need at least 4 meaningful words
            return False
        
        # Reject if primary term appears to be a function word (check start of question)
        question_start = question_lower[:50]  # First 50 chars
        for func_word in ["type of", "defined as", "integral to", "greater", "less"]:
            # Check if function word appears right after question starters
            patterns = [
                rf"how can {re.escape(func_word)}",
                rf"what is {re.escape(func_word)}",
                rf"analyze.*{re.escape(func_word)}",
                rf"design.*{re.escape(func_word)}",
                rf"explain.*{re.escape(func_word)}",
            ]
            for pattern in patterns:
                if re.search(pattern, question_start):
                    return False
        
        return True
    
    def _generate_question_single(
        self,
        context: str,
        bloom_level: str,
        topic: str,
        complexity: str,
        mode: str,
        previous_questions: List[str],
        question_index: int,
        all_previous_questions: List[str],
        max_retries: int = 3,  # Restored to 3 for better quality
        used_verbs: Optional[List[str]] = None  # Track used verbs for verb variety
    ) -> Optional[str]:
        """Helper method for parallel generation - generates a single question with retries."""
        for attempt in range(max_retries):
            try:
                question = self._generate_question(
                    context=context,
                    bloom_level=bloom_level,
                    topic=topic,
                    complexity=complexity,
                    log_generation=False,
                    mode=mode,
                    previous_questions=previous_questions,
                    question_index=question_index,
                    all_previous_questions=all_previous_questions,
                    used_verbs=used_verbs  # Pass used verbs
                )
                if question and len(question) >= 15:
                    return question
            except Exception as e:
                if attempt == max_retries - 1:
                    return None
        return None
    
    def _generate_question(
        self, 
        context: str, 
        bloom_level: str, 
        topic: str, 
        complexity: str = "intermediate", 
        log_generation: bool = True,
        mode: str = "fresh",  # "improvement" or "fresh"
        previous_questions: Optional[List[str]] = None,
        question_index: int = 0,
        all_previous_questions: Optional[List[str]] = None,  # All questions (original + generated)
        used_verbs: Optional[List[str]] = None  # Track used verbs for verb variety
    ) -> str:
        """
        Generate a single question using LLM with deduplication and diversity.
        
        Args:
            context: Content to generate question from
            bloom_level: Target Bloom's Taxonomy level
            topic: Subject topic
            complexity: Question complexity level
            log_generation: Whether to log generation details
            mode: "improvement" (uses existing questions as context) or "fresh" (uses book/content)
            previous_questions: Previously generated questions in this batch (for deduplication)
            question_index: Index of this question (for diversity)
            all_previous_questions: All questions including originals (for improvement mode)
        """
        start_time = time.time()
        previous_questions = previous_questions or []
        all_previous_questions = all_previous_questions or []
        
        # For diversity: use different context chunks for each question
        # Rotate through context to avoid always using the same part
        context_chunks = []
        chunk_size = 2000
        for i in range(0, len(context), chunk_size):
            chunk = context[i:i+chunk_size]
            if len(chunk.strip()) > 100:  # Only add substantial chunks
                context_chunks.append(chunk)
        
        # Select context chunk based on question index (round-robin for diversity)
        if context_chunks:
            selected_chunk_index = question_index % len(context_chunks)
            selected_context = context_chunks[selected_chunk_index]
        else:
            selected_context = context[:chunk_size]
        
        # For APIs (Gemini): no truncation (APIs handle context automatically)
        # For local models: truncate if needed based on model limits
        original_context_len = len(selected_context)
        if self.llm_api == "gemini":
            # APIs handle context automatically - no truncation needed
            full_context = selected_context
            truncated_tokens = 0
        elif self.local_tokenizer:
            # Local models may need truncation based on model limits
            full_context = self._truncate_tokens(selected_context, self.local_tokenizer, max_tokens=1500)
            truncated_tokens = len(self.local_tokenizer.encode(selected_context, add_special_tokens=False)) - len(self.local_tokenizer.encode(full_context, add_special_tokens=False))
        else:
            # Fallback if no tokenizer available
            full_context = selected_context[:6000]  # Rough estimate: 1500 tokens * 4 chars
            truncated_tokens = 0
        
        # Comprehensive verb pools for each Bloom level
        bloom_verb_pools = {
            'Remembering': ['define', 'list', 'name', 'identify', 'recall', 'state', 'label', 'recognize', 'select', 'match', 'memorize', 'repeat', 'reproduce', 'retrieve', 'locate'],
            'Understanding': ['explain', 'describe', 'summarize', 'interpret', 'discuss', 'clarify', 'paraphrase', 'classify', 'compare', 'contrast', 'exemplify', 'illustrate', 'outline', 'relate', 'translate'],
            'Applying': ['apply', 'use', 'demonstrate', 'implement', 'solve', 'calculate', 'execute', 'illustrate', 'utilize', 'employ', 'practice', 'operate', 'show', 'sketch', 'construct'],
            'Analyzing': ['compare', 'analyze', 'examine', 'differentiate', 'distinguish', 'contrast', 'dissect', 'investigate', 'categorize', 'organize', 'deconstruct', 'break down', 'separate', 'test', 'question'],
            'Evaluating': ['evaluate', 'assess', 'judge', 'critique', 'justify', 'defend', 'argue', 'appraise', 'prioritize', 'rate', 'rank', 'recommend', 'select', 'support', 'validate'],
            'Creating': ['design', 'develop', 'construct', 'formulate', 'create', 'plan', 'compose', 'synthesize', 'invent', 'produce', 'generate', 'build', 'assemble', 'devise', 'originate']
        }
        
        # Get verb pool for this Bloom level
        verb_pool = bloom_verb_pools.get(bloom_level, ['explain', 'describe'])
        verbs_to_use = ', '.join(verb_pool[:8])  # Show first 8 as examples
        
        # Use passed used_verbs or extract from previous_questions
        if used_verbs is None:
            used_verbs = []
            if previous_questions:
                for prev_q in previous_questions:
                    extracted_verb = self._extract_verb_from_question(prev_q, bloom_level)
                    if extracted_verb and extracted_verb not in used_verbs:
                        used_verbs.append(extracted_verb)
        
        # Get unused verbs for this question
        unused_verbs = [v for v in verb_pool if v not in used_verbs]
        if not unused_verbs:
            unused_verbs = verb_pool  # If all used, reset (allow reuse but prefer variety)
        
        # Select a verb to suggest (rotate through unused verbs)
        suggested_verb = unused_verbs[question_index % len(unused_verbs)] if unused_verbs else verb_pool[question_index % len(verb_pool)]
        
        # Build diversity instructions based on mode and previous questions
        diversity_instructions = ""
        verb_variety_instruction = f"""
- CRITICAL - VERB VARIETY REQUIREMENT: You MUST use a DIFFERENT verb than those already used in previous questions at this Bloom level.
- Previously used verbs to AVOID: {', '.join(used_verbs[:5]) if used_verbs else 'None yet'}
- Suggested verb for this question: "{suggested_verb}" (or choose another from: {', '.join(unused_verbs[:5])})
- Available verbs for {bloom_level}: {verbs_to_use}
- DO NOT repeat verbs like "{', '.join(used_verbs[:3]) if used_verbs else 'N/A'}" that were already used.
- The question will be verified to ensure it matches Bloom level {bloom_level}, so focus on the cognitive complexity rather than specific verb choice.
"""
        
        if mode == "improvement":
            if all_previous_questions:
                diversity_instructions = f"""
IMPORTANT - DIVERSITY REQUIREMENTS:
- Do NOT repeat any of these existing questions verbatim or with minor word changes:
{chr(10).join([f"  - {q}" for q in all_previous_questions[:5]])}
- Cover the SAME concepts but use DIFFERENT wording, scenarios, or perspectives.
- Ensure your question is UNIQUE and DISTINCT from all previous questions.
- If generating multiple questions, each must approach the topic from a different angle.
{verb_variety_instruction}
"""
            else:
                diversity_instructions = verb_variety_instruction
            if previous_questions:
                diversity_instructions += f"""
- You have already generated these questions in this batch:
{chr(10).join([f"  - {q}" for q in previous_questions[:3]])}
- Make sure your new question is DIFFERENT from all of these, including using different verbs.
"""
        else:  # fresh mode
            diversity_instructions = verb_variety_instruction
            if previous_questions:
                diversity_instructions += f"""
IMPORTANT - DIVERSITY REQUIREMENTS:
- You have already generated these questions:
{chr(10).join([f"  - {q}" for q in previous_questions[:3]])}
- Ensure your new question is UNIQUE and covers DIFFERENT aspects or concepts.
- Use different wording, perspectives, or scenarios even if the content is limited.
- Use different verbs to avoid repetition.
"""
        
        # Parse topics from topic string
        parsed_topics = self._parse_topics(topic)
        has_multiple_topics = len(parsed_topics) > 1
        
        # Build topic coverage instruction if multiple topics
        topic_coverage_instruction = ""
        if has_multiple_topics:
            topic_coverage_instruction = f"""
- TOPIC COVERAGE REQUIREMENT: The exam must cover ALL of these topics: {', '.join(parsed_topics)}
- Ensure balanced coverage across all topics. Do NOT focus only on one topic.
- If previous questions have heavily covered one topic, prioritize other topics for this question.
"""
        
        # Vary prompt based on question index for more diversity
        variation_hints = [
            "Focus on a different aspect or application.",
            "Use a different scenario or example.",
            "Approach from a different angle or perspective.",
            "Consider a different use case or context.",
            "Explore a different relationship or connection."
        ]
        variation_hint = variation_hints[question_index % len(variation_hints)] if question_index > 0 else ""
        
        # Generate natural exam questions with mode-specific instructions
        if mode == "improvement":
            prompt = f"""You are an exam generator. Write ONE natural, improved exam question for Bloom level: {bloom_level}.

MODE: Improvement Exam (existing questions provided as context)
GOAL: Generate a UNIQUE, better-structured question that covers the same concepts but does NOT repeat existing questions.

RULES:
- Use information from the provided content (existing questions).
- Write naturally, like a real exam question.
- Keep it to ONE sentence only - be concise.
- Do NOT add phrases like "according to the text", "based on the content", "as described", etc.
- Reference concepts from the content naturally.
- Use question mark only if it's a question. Use period for statements/imperatives.
- Cover the SAME concepts as existing questions but use DIFFERENT wording, scenarios, or perspectives.
- Ensure the question matches Bloom level: {bloom_level} (this will be verified automatically).
{topic_coverage_instruction}
{diversity_instructions}
{variation_hint}

CONTENT (existing questions):

{full_context}

Write ONE unique, improved exam question:"""
        else:  # fresh mode
            prompt = f"""You are an exam generator. Write ONE natural exam question for Bloom level: {bloom_level}.

MODE: Fresh Exam (content-based generation)
GOAL: Generate a UNIQUE question from the provided content.

RULES:
- Use information from the provided content.
- Write naturally, like a real exam question.
- Keep it to ONE sentence only - be concise.
- Do NOT add phrases like "according to the text", "based on the content", "as described", etc.
- Reference concepts from the content naturally.
- Use question mark only if it's a question. Use period for statements/imperatives.
- Ensure the question matches Bloom level: {bloom_level} (this will be verified automatically).
{topic_coverage_instruction}
{diversity_instructions}
{variation_hint}

CONTENT:

{full_context}

Write ONE unique exam question:"""
        
        # Log prompt details
        if log_generation:
            prompt_tokens = len(prompt.split()) if self.local_tokenizer is None else len(self.local_tokenizer.encode(prompt))
            context_tokens = len(full_context.split()) if self.local_tokenizer is None else len(self.local_tokenizer.encode(full_context))
            
            truncated_chars = original_context_len - len(full_context)
        
        def _validate_and_filter(candidate: Optional[str], source: str) -> Optional[str]:
            """Common validation/dedup pipeline for generated questions."""
            if not candidate:
                return None
            candidate = candidate.strip()
            if not self._is_valid_question(candidate):
                return None
            all_to_check = list(previous_questions) + list(all_previous_questions)
            if self._is_duplicate_question(candidate, all_to_check, threshold=0.7):
                return None
            return candidate
        
        question = None
        generation_error = None
        used_fallback = False
        fallback_error = None
        gemini_cooldown_remaining = 0
        if self.llm_api == "gemini":
            gemini_cooldown_remaining = max(0, getattr(self, "_gemini_disabled_until", 0) - time.time())
        
        try:
            # Primary path: Gemini API
            if self.llm_api == "gemini":
                if gemini_cooldown_remaining > 0:
                    raise Exception("Gemini temporarily rate-limited. Please wait for the cooldown and try again.")
                
                question = self._generate_with_gemini(prompt, log_params=log_generation)
            
            # Local generation path (e.g., Qwen) – used when RAGExamGenerator is instantiated with llm_api='local'
            elif self.llm_api == "local":
                if not self.local_model or not self.local_tokenizer:
                    error_msg = (
                        f"Local generation requested (llm_api='local') but local_model/local_tokenizer are not available. "
                        f"local_model={self.local_model is not None}, local_tokenizer={self.local_tokenizer is not None}"
                    )
                    raise ValueError(error_msg)
                question = self._generate_with_local_model(prompt, log_params=log_generation)
            
            else:
                raise ValueError(f"Invalid llm_api='{self.llm_api}' for generation. Expected 'gemini' or 'local'.")
        except Exception as e:
            generation_error = str(e)
            
            if self.llm_api == "gemini":
                # Handle Gemini-specific failures, and optionally fall back to local model
                error_lower = generation_error.lower()
                if "rate limit" in error_lower:
                    cooldown_seconds = 180  # 3 minutes cooldown to avoid hammering free tier
                    self._gemini_disabled_until = time.time() + cooldown_seconds

                # Optional local-model fallback when Gemini is unavailable
                if self.allow_local_fallback:
                    try:
                        if not self.local_model or not self.local_tokenizer:
                            from utils.generation_model_loader import load_generation_model
                            self.local_model, self.local_tokenizer = load_generation_model(
                                model_name=None,
                                use_quantization=False,
                                use_cpu=not torch.cuda.is_available()
                            )
                        question = self._generate_with_local_model(prompt, log_params=log_generation)
                        used_fallback = True
                        generation_error = None  # Clear primary error if fallback succeeds
                    except Exception as fe:
                        fallback_error = str(fe)
        
        
        # Validate question before accepting (includes deduplication)
        question = _validate_and_filter(question, "primary")
        
        
        # No fallback - raise exception if generation failed
        if not question:
            cooldown_note = ""
            if self.llm_api == "gemini":
                cooldown_remaining = max(0, int(getattr(self, "_gemini_disabled_until", 0) - time.time()))
                if cooldown_remaining > 0:
                    cooldown_note = f" Gemini free-tier cooldown ~{cooldown_remaining}s."
            if generation_error:
                error_msg = f"Generation failed ({self.llm_api}): {generation_error}{cooldown_note}"
            elif fallback_error:
                error_msg = f"Generation failed ({self.llm_api}): Local fallback failed: {fallback_error}{cooldown_note}"
            else:
                error_msg = f"Generation failed ({self.llm_api}): No question generated.{cooldown_note}"
            raise Exception(error_msg)
        
        # Log generation result
        if log_generation:
            latency = time.time() - start_time
            question_tokens = len(question.split()) if self.local_tokenizer is None else len(self.local_tokenizer.encode(question))
            
            # Store in log
            log_entry = {
                "timestamp": time.time(),
                "bloom_level": bloom_level,
                "context_length": len(full_context),
                "context_truncated": truncated_chars,
                "prompt_tokens": prompt_tokens if 'prompt_tokens' in locals() else 0,
                "question": question,
                "question_tokens": question_tokens,
                "latency": latency,
                "error": generation_error,
                "used_fallback": used_fallback
            }
            self.generation_logs.append(log_entry)
            _generation_logs.append(log_entry)
        
        return question
    
    def generate_manual_question(self, context: str, bloom_level: str = "Understanding") -> str:
        """Generate a single question manually (for testing) - bypasses Bloom pipeline."""
        return self._generate_question(context, bloom_level, "test", log_generation=True)
    
    def get_last_generation_logs(self, n: int = 10) -> List[Dict]:
        """Get last N generation logs."""
        return list(_generation_logs)[-n:]
    
    def print_generation_logs(self, n: int = 10):
        """Print last N generation logs to terminal."""
        pass
    
    def generate_exam_from_content(
        self,
        total_questions: int = 20,
        topic: str = "",
        custom_distribution: Optional[Dict[str, int]] = None,
        complexity: str = "intermediate",
        specific_bloom_level: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a complete exam from stored content following Bloom's taxonomy distribution.
        
        Args:
            total_questions: Total number of questions to generate
            topic: Topic/subject area (for context)
            custom_distribution: Optional custom distribution per Bloom level
            complexity: Difficulty level - 'easy', 'intermediate', or 'complex'
            specific_bloom_level: If specified, generate all questions at this Bloom level only
            progress_callback: Optional callback function(progress: float, status: str) for progress updates
        
        Returns:
            List of question dictionaries with 'question', 'bloom_level', and 'complexity' keys
        """
        from utils.bloom_analyzer_complete import IDEAL_DISTRIBUTION, BLOOM_LEVELS
        
        if not self.chunks:
            raise ValueError("No content in vector store. Please add content first using add_content().")
        
        if progress_callback:
            progress_callback(0.0, "Preparing question distribution...")
        
        # If specific bloom level requested, generate all questions at that level
        if specific_bloom_level:
            if specific_bloom_level not in BLOOM_LEVELS:
                raise ValueError(f"Invalid Bloom level: {specific_bloom_level}. Must be one of: {BLOOM_LEVELS}")
            distribution = {specific_bloom_level: total_questions}
        # Calculate question distribution based on ideal distribution
        elif custom_distribution:
            distribution = custom_distribution
        else:
            distribution = {}
            for level in BLOOM_LEVELS:
                # Calculate base distribution using proper rounding (not truncation)
                # This ensures we get closer to ideal percentages
                ideal_count = total_questions * IDEAL_DISTRIBUTION[level]
                base_count = round(ideal_count)
                if total_questions >= 6:
                    distribution[level] = max(1, base_count)
                else:
                    # For small question counts, allow 0 for some levels
                    distribution[level] = max(0, base_count)
            
            # Adjust to match total exactly
            current_total = sum(distribution.values())
            if current_total != total_questions:
                diff = total_questions - current_total
                if diff > 0:
                    # Add extra questions to levels that need them most (prioritize Understanding, Applying, Analyzing)
                    # But also consider which levels are furthest below their ideal
                    level_deficits = []
                    for level in BLOOM_LEVELS:
                        ideal_count = total_questions * IDEAL_DISTRIBUTION[level]
                        ideal_rounded = round(ideal_count)
                        deficit = ideal_rounded - distribution[level]
                        if deficit > 0:
                            level_deficits.append((deficit, ideal_count, level))
                    # Sort by deficit (largest first), then by ideal percentage
                    level_deficits.sort(key=lambda x: (x[0], x[1]), reverse=True)
                    priority_levels = [level for _, _, level in level_deficits]
                    # Fallback to default priority if no deficits
                    if not priority_levels:
                        priority_levels = ['Understanding', 'Applying', 'Analyzing', 'Remembering', 'Evaluating', 'Creating']
                    
                    for level in priority_levels:
                        if diff <= 0:
                            break
                        distribution[level] += 1
                        diff -= 1
                elif diff < 0:
                    # Remove questions from levels with excess (start with higher counts)
                    # Prioritize removing from levels that are above their ideal
                    level_excess = []
                    for level in BLOOM_LEVELS:
                        ideal_count = total_questions * IDEAL_DISTRIBUTION[level]
                        ideal_rounded = round(ideal_count)
                        excess = distribution[level] - ideal_rounded
                        if excess > 0:
                            level_excess.append((excess, level))
                    # Sort by excess (largest first)
                    level_excess.sort(key=lambda x: x[0], reverse=True)
                    sorted_levels = [level for _, level in level_excess]
                    # Fallback to default if no excess
                    if not sorted_levels:
                        sorted_levels = [level for level, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True)]
                    
                    for level in sorted_levels:
                        if diff >= 0:
                            break
                        if distribution[level] > 1:  # Don't reduce below 1
                            reduction = min(abs(diff), distribution[level] - 1)
                            distribution[level] -= reduction
                            diff += reduction
        
        generated_questions = []
        # Use set for O(1) duplicate checks instead of nested loops
        seen_questions = set()  # Track lowercase questions for fast duplicate detection
        all_generated_questions = []  # Track all generated questions for deduplication (fresh mode)
        
        # Track used verbs per Bloom level to enforce verb variety
        used_verbs_by_level = {level: [] for level in BLOOM_LEVELS}
        
        if progress_callback:
            progress_callback(0.05, "Caching relevant content chunks...")
        
        # Get diverse chunks for each Bloom level across the entire document
        # Problem: Similarity search was finding same chunks (often early chunks).
        # Solution: Each Bloom level gets chunks from different document sections.
        bloom_chunks_cache = {}
        total_chunks = len(self.chunks)
        
        # Get Bloom levels that need questions
        bloom_levels_list = [level for level, count in distribution.items() if count > 0]
        num_active_levels = len(bloom_levels_list)
        
        if num_active_levels == 0:
            return []
        
        # Calculate section size for each Bloom level (divide document into sections)
        section_size = max(1, total_chunks // num_active_levels)
        
        # Assign chunks to each Bloom level from different document sections
        for level_idx, bloom_level in enumerate(bloom_levels_list):
            num_questions = distribution[bloom_level]
            if num_questions <= 0:
                continue
            
            bloom_chunks = []
            
            # Each Bloom level gets chunks from a different section of the document
            # Level 0: chunks 0-33%, Level 1: chunks 33-66%, Level 2: chunks 66-100%, etc.
            section_start = (level_idx * section_size) % total_chunks
            section_end = min(section_start + section_size, total_chunks)
            
            # If section wraps around (near end of document), also include beginning
            if section_end < section_start + section_size:
                # Wraps around - get from end + beginning
                chunks_from_end = self.chunks[section_start:]
                chunks_from_start = self.chunks[:section_size - len(chunks_from_end)]
                section_chunks = chunks_from_end + chunks_from_start
            else:
                section_chunks = self.chunks[section_start:section_end]
            
            # Sample diverse chunks from this section (not all adjacent)
            if section_chunks:
                # Take chunks with stride to avoid adjacent chunks
                stride = max(1, len(section_chunks) // max(num_questions, 1))
                for i in range(num_questions * 2):  # Get 2x for variety
                    idx = (i * stride) % len(section_chunks)
                    chunk = section_chunks[idx]
                    if chunk not in bloom_chunks:
                        bloom_chunks.append(chunk)
                    if len(bloom_chunks) >= num_questions * 2:
                        break
            
            # If still need more, sample from other sections too (but less frequently)
            if len(bloom_chunks) < num_questions:
                # Get chunks from middle and end sections for extra variety
                for alt_section in [total_chunks // 3, (total_chunks * 2) // 3]:
                    for i in range(num_questions):
                        idx = (alt_section + i * 5) % total_chunks
                        chunk = self.chunks[idx]
                        if chunk not in bloom_chunks:
                            bloom_chunks.append(chunk)
                        if len(bloom_chunks) >= num_questions * 3:
                            break
                    if len(bloom_chunks) >= num_questions * 3:
                        break
            
            # Final fallback: use all chunks but shuffled
            if not bloom_chunks:
                import random
                bloom_chunks = random.sample(self.chunks, min(num_questions * 3, total_chunks))
            elif len(bloom_chunks) < num_questions:
                # Pad with remaining chunks
                remaining = [c for c in self.chunks if c not in bloom_chunks]
                bloom_chunks.extend(remaining[:num_questions])
            
            bloom_chunks_cache[bloom_level] = bloom_chunks[:num_questions * 3]  # Keep extra for retries
        
        total_to_generate = sum(distribution.values())
        generated_count = 0
        
        # Pre-process topic string to avoid repeated string operations
        topic_str = topic or "the subject"
        
        # Parse multiple topics from the topic string
        parsed_topics = self._parse_topics(topic_str)
        has_multiple_topics = len(parsed_topics) > 1
        
        # Track topic distribution to ensure balanced coverage
        topic_distribution = {}
        if has_multiple_topics:
            for t in parsed_topics:
                topic_distribution[t] = 0
        
        if progress_callback:
            progress_callback(0.1, f"Generating {total_to_generate} questions...")
        
        # Track which chunks have been used to avoid repetition
        chunk_usage_tracker = {}  # Track how many times each chunk index has been used
        
        # Use parallel generation ONLY for local models.
        # For Gemini, force sequential generation to avoid hammering the free-tier rate limits.
        use_parallel = (self.llm_api == "local" and total_to_generate > 3)
        
        # Container for batched generation tasks (used only when use_parallel is True)
        all_generation_tasks = []
        task_index = 0
        
        for bloom_level, num_questions in distribution.items():
            if num_questions <= 0:
                continue
            
            # Use cached chunks for this Bloom level
            relevant_chunks = bloom_chunks_cache.get(bloom_level, [])
            
            # If no chunks cached, get diverse chunks now
            if not relevant_chunks:
                total_chunks = len(self.chunks)
                chunk_indices = []
                for i in range(min(num_questions * 2, total_chunks)):
                    idx = (task_index * 5 + i * 7) % total_chunks
                    if idx not in chunk_usage_tracker or chunk_usage_tracker[idx] < 2:
                        chunk_indices.append(idx)
                relevant_chunks = [self.chunks[idx] for idx in chunk_indices[:num_questions * 2]]
            
            # Prepare tasks for this Bloom level
            for i in range(num_questions):
                # Select chunk
                if relevant_chunks:
                    min_usage = float('inf')
                    selected_chunk = None
                    selected_chunk_idx = None
                    for j, chunk in enumerate(relevant_chunks):
                        chunk_idx = chunk.get('index', j)
                        usage_count = chunk_usage_tracker.get(chunk_idx, 0)
                        if usage_count < min_usage:
                            min_usage = usage_count
                            selected_chunk = chunk
                            selected_chunk_idx = chunk_idx
                    if selected_chunk:
                        chunk = selected_chunk
                        if selected_chunk_idx is not None:
                            chunk_usage_tracker[selected_chunk_idx] = chunk_usage_tracker.get(selected_chunk_idx, 0) + 1
                        if self.local_tokenizer:
                            context = self._truncate_tokens(chunk['text'], self.local_tokenizer, max_tokens=1500)
                        else:
                            context = chunk['text'][:6000]
                    else:
                        chunk_idx = i % len(relevant_chunks)
                        chunk = relevant_chunks[chunk_idx]
                        if self.local_tokenizer:
                            context = self._truncate_tokens(chunk['text'], self.local_tokenizer, max_tokens=1500)
                        else:
                            context = chunk['text'][:6000]
                else:
                    total_chunks = len(self.chunks)
                    chunk_idx = (task_index * 13 + i * 17) % total_chunks
                    chunk = self.chunks[chunk_idx]
                    chunk_usage_tracker[chunk_idx] = chunk_usage_tracker.get(chunk_idx, 0) + 1
                    if self.local_tokenizer:
                        context = self._truncate_tokens(chunk['text'], self.local_tokenizer, max_tokens=1500)
                    else:
                        context = chunk['text'][:6000]
                
                # Only build batched tasks when parallel generation is enabled
                if use_parallel and total_to_generate > 1:
                    all_generation_tasks.append({
                        'context': context,
                        'bloom_level': bloom_level,
                        'question_index': task_index,
                        'chunk': chunk
                    })
                task_index += 1
                
            # Execute ALL questions in parallel (much faster!) for local models when enabled
        if use_parallel and total_to_generate > 1 and all_generation_tasks:
            max_workers = min(25, total_to_generate)
            completed = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {}
                for task in all_generation_tasks:
                    future = executor.submit(
                        self._generate_question_single,
                        context=task['context'],
                        bloom_level=task['bloom_level'],
                        topic=topic_str,
                        complexity=complexity,
                        mode="fresh",
                        previous_questions=all_generated_questions,
                        question_index=task['question_index'],
                        all_previous_questions=[],
                        max_retries=3,  # Restored to 3 for better quality
                        used_verbs=used_verbs_by_level.get(task['bloom_level'], [])
                    )
                    future_to_task[future] = task
            
                # Process results as they complete
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    completed += 1
                    if progress_callback:
                        progress = 0.1 + (completed / total_to_generate) * 0.9
                        progress_callback(progress, f"Generating question {completed}/{total_to_generate} ({task['bloom_level']})...")
                    
                    try:
                        question = future.result()
                        if question and len(question) >= 15:
                            question_lower = question.lower().strip()
                            # Restored: Full semantic duplicate check against all questions
                            if question_lower not in seen_questions:
                                # Check against all previously generated questions for duplicates
                                is_semantic_duplicate = False
                                if len(all_generated_questions) > 0:
                                    is_semantic_duplicate = self._is_duplicate_question(question, all_generated_questions, threshold=0.8)
                                
                                if not is_semantic_duplicate:
                                    # Restored: Full content validation for all questions
                                    content_valid = self._validate_question_uses_content(question, task['context'], quick_check=True)
                                    
                                    if content_valid:
                                        question = self._clean_question_final(question, topic=topic_str)
                                        seen_questions.add(question_lower)
                                        all_generated_questions.append(question)
                                        
                                        # Track verb used
                                        extracted_verb = self._extract_verb_from_question(question, task['bloom_level'])
                                        if extracted_verb and extracted_verb not in used_verbs_by_level[task['bloom_level']]:
                                            used_verbs_by_level[task['bloom_level']].append(extracted_verb)
                                        
                                        generated_questions.append({
                                            'question': question,
                                            'bloom_level': task['bloom_level'],
                                            'complexity': complexity,
                                        })
                                        generated_count += 1
                    except Exception:
                        # Skip failed questions but keep the loop running
                        pass
        
        # Sequential generation for local models or if parallel was skipped / not used (e.g., Gemini)
        if not use_parallel or generated_count < total_to_generate:
            for bloom_level, num_questions in distribution.items():
                if num_questions <= 0:
                    continue
                
                # Use cached chunks for this Bloom level (already diverse from preprocessing)
                relevant_chunks = bloom_chunks_cache.get(bloom_level, [])
                
                # If no chunks cached, get diverse chunks now
                if not relevant_chunks:
                    # Sample from different parts of document
                    total_chunks = len(self.chunks)
                    chunk_indices = []
                    for i in range(min(num_questions * 2, total_chunks)):  # Get 2x chunks for selection
                        idx = (generated_count * 5 + i * 7) % total_chunks  # Large stride to get diverse chunks
                        if idx not in chunk_usage_tracker or chunk_usage_tracker[idx] < 2:  # Limit reuse
                            chunk_indices.append(idx)
                    relevant_chunks = [self.chunks[idx] for idx in chunk_indices[:num_questions * 2]]
                
                # Sequential generation for this Bloom level - keep trying until we get required number
                questions_added_this_level = 0
                max_attempts = max(10, num_questions * 5)  # at least 10 attempts to be a bit more robust
                attempt = 0
                early_exit_all = False
                
                while questions_added_this_level < num_questions and attempt < max_attempts:
                    attempt += 1
                    
                    # If the total requested questions have already been produced elsewhere, set flag and break
                    if len(generated_questions) >= total_to_generate:
                        early_exit_all = True
                        break  # allow outer loop to handle stopping
                    
                    # Update progress
                    if progress_callback:
                        progress = 0.1 + (len(generated_questions) / total_to_generate) * 0.9
                        progress_callback(progress, f"Generating question {len(generated_questions) + 1}/{total_to_generate} ({bloom_level})...")
                    
                    # Select chunk
                    if relevant_chunks:
                        chunk_idx = questions_added_this_level % len(relevant_chunks)
                        chunk = relevant_chunks[chunk_idx]
                        if self.local_tokenizer:
                            context = self._truncate_tokens(chunk['text'], self.local_tokenizer, max_tokens=1500)
                        else:
                            context = chunk['text'][:6000]
                    else:
                        total_chunks = len(self.chunks)
                        chunk_idx = (len(generated_questions) * 13 + attempt * 17) % total_chunks
                        chunk = self.chunks[chunk_idx]
                        if self.local_tokenizer:
                            context = self._truncate_tokens(chunk['text'], self.local_tokenizer, max_tokens=1500)
                        else:
                            context = chunk['text'][:6000]
                    
                    # Generate question
                    question = None
                    try:
                        question = self._generate_question(
                            context=context,
                            bloom_level=bloom_level,
                            topic=topic_str,
                            complexity=complexity,
                            log_generation=False,
                            mode="fresh",
                            previous_questions=all_generated_questions,
                            question_index=len(generated_questions),
                            all_previous_questions=[],
                            used_verbs=used_verbs_by_level.get(bloom_level, [])
                        )
                    except Exception as e:
                        # The exponential backoff retry is handled inside _generate_with_gemini
                        # If it still fails after all retries, wait longer before continuing
                        error_str = str(e).lower()
                        if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
                            # After 6 failed retries, wait 60s before trying next question
                            wait_seconds = 60
                            if progress_callback:
                                progress_callback(0.5, f"Rate limit reached. Waiting {wait_seconds}s before continuing...")
                            time.sleep(wait_seconds)
                        continue
                    
                    # Basic validation
                    if not question or len(question.strip()) < 15:
                        continue
                    
                    question = self._clean_question_final(question, topic=topic_str)
                    question_lower = question.lower().strip()
                    
                    # Check duplicates (exact)
                    if question_lower in seen_questions:
                        continue
                    
                    # Check near-duplicates against recent questions
                    if all_generated_questions:
                        recent = all_generated_questions[-10:]
                        if self._is_duplicate_question(question, recent, threshold=0.8):
                            continue
                    
                    # Passed all checks -> add the question
                    generated_questions.append({
                        'question': question,
                        'bloom_level': bloom_level,
                        'complexity': complexity,
                        'source': 'generated'
                    })
                    seen_questions.add(question_lower)
                    all_generated_questions.append(question)
                    # Keep generated_count in sync if you still need it elsewhere
                    generated_count = len(generated_questions)  # make it authoritative and atomic here
                    questions_added_this_level += 1
                    
                    # Track verb
                    verb = self._extract_verb_from_question(question, bloom_level)
                    if verb and verb not in used_verbs_by_level[bloom_level]:
                        used_verbs_by_level[bloom_level].append(verb)
                
                # End per-level loop
                
                if early_exit_all:
                    break  # break the outer for-level loop
        
        # Post-generation verb repetition validation and regeneration
        if progress_callback:
            progress_callback(0.95, "Validating verb variety...")
        
        # Check for verb repetition and regenerate if needed
        verb_repetition_found = False
        questions_to_regenerate = []
        
        for level in BLOOM_LEVELS:
            level_questions = [q for q in generated_questions if q['bloom_level'] == level]
            if len(level_questions) > 1:
                # Extract verbs from all questions at this level
                level_verbs = []
                for q_dict in level_questions:
                    verb = self._extract_verb_from_question(q_dict['question'], level)
                    if verb:
                        level_verbs.append(verb)
                
                # Check for repetition (same verb used more than once)
                from collections import Counter
                verb_counts = Counter(level_verbs)
                repeated_verbs = {v: count for v, count in verb_counts.items() if count > 1}
                
                if repeated_verbs:
                    verb_repetition_found = True
                    # Mark questions with repeated verbs for regeneration
                    for q_dict in level_questions:
                        verb = self._extract_verb_from_question(q_dict['question'], level)
                        if verb in repeated_verbs:
                            questions_to_regenerate.append({
                                'question_dict': q_dict,
                                'bloom_level': level,
                                'repeated_verb': verb
                            })
        
        # Regenerate questions with verb repetition (limit to avoid infinite loops)
        if verb_repetition_found and questions_to_regenerate and len(questions_to_regenerate) <= 5:
            if progress_callback:
                progress_callback(0.97, f"Regenerating {len(questions_to_regenerate)} questions to improve verb variety...")
            
            for regen_item in questions_to_regenerate:
                q_dict = regen_item['question_dict']
                level = regen_item['bloom_level']
                repeated_verb = regen_item['repeated_verb']
                
                # Get used verbs for this level (excluding the repeated one for this regeneration)
                current_used_verbs = [v for v in used_verbs_by_level[level] if v != repeated_verb]
                
                # Find a chunk for regeneration
                relevant_chunks = bloom_chunks_cache.get(level, [])
                if not relevant_chunks:
                    relevant_chunks = self.chunks[:10]  # Fallback
                
                chunk = relevant_chunks[len(generated_questions) % len(relevant_chunks)]
                if self.local_tokenizer:
                    context = self._truncate_tokens(chunk['text'], self.local_tokenizer, max_tokens=1500)
                else:
                    context = chunk['text'][:6000]
                
                # Regenerate with explicit verb variety constraint
                try:
                    new_question = self._generate_question(
                        context=context,
                        bloom_level=level,
                        topic=topic_str,
                        complexity=complexity,
                        log_generation=False,
                        mode="fresh",
                        previous_questions=[q['question'] for q in generated_questions if q['bloom_level'] == level],
                        question_index=len(generated_questions),
                        all_previous_questions=[],
                        used_verbs=current_used_verbs  # Pass used verbs to avoid repetition
                    )
                    
                    if new_question and len(new_question) >= 15:
                        new_verb = self._extract_verb_from_question(new_question, level)
                        # Only replace if new verb is different
                        if new_verb and new_verb != repeated_verb:
                            # Remove old question
                            generated_questions.remove(q_dict)
                            seen_questions.discard(q_dict['question'].lower().strip())
                            
                            # Add new question
                            new_question = self._clean_question_final(new_question, topic=topic_str)
                            new_question_lower = new_question.lower().strip()
                            if new_question_lower not in seen_questions:
                                generated_questions.append({
                                    'question': new_question,
                                    'bloom_level': level,
                                    'complexity': complexity,
                                    'source': 'generated'  # Fixed: UI checks for 'generated', not 'rag_generated'
                                })
                                seen_questions.add(new_question_lower)
                                if new_verb not in used_verbs_by_level[level]:
                                    used_verbs_by_level[level].append(new_verb)
                except Exception as e:
                    # If regeneration fails, keep original question
                    pass
        
        if progress_callback:
            progress_callback(1.0, "Question generation complete!")
        
        return generated_questions
    
    def improve_exam_with_rag(
        self,
        original_questions: List[str],
        analysis_model,
        analysis_tokenizer,
        topic: str = "",
        exam_name: str = "Exam",
        use_analysis_model_for_gen: bool = True,
        analysis_result: Optional[Dict[str, Any]] = None  # Accept pre-computed analysis
    ) -> Dict[str, Any]:
        """
        Improve an exam by generating additional questions using RAG to fill Bloom taxonomy gaps.
        
        Args:
            original_questions: List of original exam questions
            analysis_model: Model for analyzing Bloom levels
            analysis_tokenizer: Tokenizer for analysis model
            topic: Topic/subject area
            exam_name: Name of the exam
            use_analysis_model_for_gen: If True, use analysis_model for generation too (faster)
        
        Returns:
            Dictionary with improvement results
        """
        from utils.bloom_analyzer_complete import analyze_exam, BLOOM_LEVELS, IDEAL_DISTRIBUTION
        
        if not self.chunks:
            raise ValueError("No content in vector store. Please add content first using add_content().")
        
        # Cache analysis model/tokenizer for Gemini fallbacks
        self._analysis_model = analysis_model
        self._analysis_tokenizer = analysis_tokenizer
        self._analysis_model_attached = False
        
        # Generation strategy:
        # - Gemini is the primary generator when llm_api == "gemini" and an API key is available.
        # - Local generation (Qwen, etc.) is used when llm_api == "local".
        # - TinyLlama analysis model is used ONLY for Bloom analysis / verification, never as a generator.
        if self.llm_api == "gemini" and not self.api_key:
            raise ValueError(
                "Gemini selected for generation (llm_api='gemini') but GEMINI_API_KEY is not available. "
                "Provide a Gemini key or construct RAGExamGenerator with llm_api='local' for offline generation."
            )
        
        # Use provided analysis if available (from improve_exam_smart)
        import time
        start_time = time.time()
        
        if analysis_result is not None:
            original_analysis = analysis_result
        else:
            original_analysis = analyze_exam(original_questions, analysis_model, analysis_tokenizer)
        
        total_questions = len(original_questions)
        
        # Parse recommendations directly from analysis instead of recalculating
        # This ensures we follow the exact recommendations shown to the user
        removals: Dict[str, int] = {}  # Track how many to remove per level
        removal_changes = []
        add_changes = []
        changes = []
        
        # ----------------------------------------------------------------------------------
        # STEP 1: Build current counts and compute a FEASIBLE final exam size
        # ----------------------------------------------------------------------------------
        current_counts = {
            level: original_analysis['comparison'][level]['count']
            for level in BLOOM_LEVELS
        }
        
        # First, compute ideal counts for the ORIGINAL total (for diagnostics only)
        ideal_original = {}
        for level in BLOOM_LEVELS:
            raw = total_questions * IDEAL_DISTRIBUTION[level]
            ideal_original[level] = max(0, int(round(raw)))
        
        # Adjust rounding so sum equals original total
        current_sum = sum(ideal_original.values())
        if current_sum != total_questions:
            diff = total_questions - current_sum
            if diff > 0:
                # Add to higher levels first
                for level in ['Creating', 'Evaluating', 'Analyzing', 'Applying', 'Understanding', 'Remembering']:
                    if diff <= 0:
                        break
                    ideal_original[level] += 1
                    diff -= 1
            else:
                # Remove from lower levels first
                for level in ['Remembering', 'Understanding', 'Applying', 'Analyzing', 'Evaluating', 'Creating']:
                    if diff >= 0:
                        break
                    if ideal_original[level] > 0:
                        ideal_original[level] -= 1
                        diff += 1
        
        # Compute how much we would like to remove/add relative to this original ideal.
        # This is deterministic: for each level we know exactly how many questions are
        # above or below the ideal count for the ORIGINAL exam size.
        excess_original: Dict[str, int] = {}
        deficit_original: Dict[str, int] = {}
        total_excess = 0
        total_deficit = 0
        for level in BLOOM_LEVELS:
            curr = current_counts.get(level, 0)
            tgt = ideal_original.get(level, 0)
            if curr > tgt:
                val = curr - tgt
                excess_original[level] = val
                total_excess += val
            elif curr < tgt:
                val = tgt - curr
                deficit_original[level] = val
                total_deficit += val

        # For this deterministic workflow, our ideal counts are simply the
        # rounded targets for the ORIGINAL exam size, already computed above.
        ideal_distribution_counts = dict(ideal_original)

        # ----------------------------------------------------------------------------------
        # STEP 3: Calculate removals to reach the ideal distribution
        # ----------------------------------------------------------------------------------
        calc_start = time.time()
        for level in BLOOM_LEVELS:
            current_count = current_counts.get(level, 0)
            ideal_count = ideal_distribution_counts.get(level, 0)

            if current_count > ideal_count:
                # Need to remove enough to reach ideal
                count_to_remove = current_count - ideal_count
                removals[level] = count_to_remove
                removal_changes.append({
                    'level': level,
                    'action': 'remove',
                    'count': count_to_remove,
                    'current': current_count,
                    'target': ideal_count
                })
                changes.append(removal_changes[-1])


        # ----------------------------------------------------------------------------------
        # STEP 4: Determine additions deterministically from exact deficits vs ideal
        # ----------------------------------------------------------------------------------
        current_counts_after_removal: Dict[str, int] = {}
        for level in BLOOM_LEVELS:
            current_count = current_counts.get(level, 0)
            if level in removals:
                current_counts_after_removal[level] = max(0, current_count - removals[level])
            else:
                current_counts_after_removal[level] = current_count

        # Exact deficits per level relative to the ideal distribution AFTER removals.
        deficits: Dict[str, int] = {}
        total_deficit_after_removal = 0
        for level in BLOOM_LEVELS:
            ideal_count = ideal_distribution_counts.get(level, 0)
            current_after = current_counts_after_removal.get(level, 0)
            deficit = max(0, ideal_count - current_after)
            deficits[level] = deficit
            total_deficit_after_removal += deficit

        add_changes = []
        for level in BLOOM_LEVELS:
            count_to_add = deficits.get(level, 0)
            if count_to_add > 0:
                current_after = current_counts_after_removal.get(level, 0)
                target = ideal_distribution_counts.get(level, 0)
                change = {
                    'level': level,
                    'action': 'add',
                    'count': count_to_add,
                    'current': current_after,
                    'target': target
                }
                add_changes.append(change)
                changes.append(change)

        effective_total_adds = sum(c.get('count', 0) for c in add_changes)
        
        # Generate new questions using RAG with batch Bloom verification for speed
        new_questions = []
        all_generated_questions = []  # Track all generated for deduplication
        global_question_index = 0  # Global index for context chunk rotation
        
        # Prepare context from original questions for improvement mode
        exam_content = "\n\n".join([f"Question {i+1}: {q}" for i, q in enumerate(original_questions)])
        
        # Generate questions for each needed Bloom level with batch verification
        generation_start = time.time()
        
        for change_idx, change in enumerate(add_changes):  # Only process 'add' actions for generation
            level = change['level']
            count = change['count']
            level_start = time.time()
            
            # Generate questions - SKIP VERIFICATION (trust Gemini generation)
            # VERIFICATION COMMENTED OUT - Gemini generates correct level, model verification causes false rejections
            verified_questions = []
            max_attempts = count * 4  # Reduced back to 4x for speed
            attempts = 0
            last_log_time = time.time()
            
            # Skip batch verification - accept questions directly after basic validation
            # batch_size = 5  # Verify 5 questions at a time - COMMENTED OUT
            # pending_questions = []  # Questions waiting for Bloom verification - COMMENTED OUT
            
            while len(verified_questions) < count and attempts < max_attempts:
                try:
                # Add delay between question generations to avoid rate limits (especially for Gemini)
                    if attempts > 0 and self.llm_api == "gemini":
                        delay = 1.5  # 1.5 second delay between Gemini requests to avoid rate limits
                        time.sleep(delay)
                    
                    # Generate one question at a time using RAG with improvement mode
                    gen_start = time.time()
                    generated_question = self._generate_question(
                        context=exam_content,  # Use original questions as context
                        bloom_level=level,
                        topic=topic or "the subject",
                        complexity="intermediate",
                        log_generation=False,
                        mode="improvement",  # Improvement mode: uses existing questions as context
                        previous_questions=all_generated_questions,  # Previously generated in this batch
                        question_index=global_question_index,  # Global index for context chunk rotation
                        all_previous_questions=original_questions  # Original questions for deduplication
                    )
                    gen_time = time.time() - gen_start
                    
                    attempts += 1
                    global_question_index += 1
                    
                    # Log progress every 5 attempts or every 10 seconds
                    if attempts % 5 == 0 or (time.time() - last_log_time) > 10:
                        last_log_time = time.time()
                    
                    # Pre-filter: quick validation before expensive Bloom check
                    if not generated_question or len(generated_question.strip()) < 10:
                        continue
                    
                    # Check for incomplete questions (quick pre-filter)
                    question_stripped = generated_question.strip()
                    if question_stripped.endswith((' such as', ' using', ' by', ' through', ' with', ' for', ' to', ' in', ' on', ' at', ' different', ' an', ' a')):
                        continue  # Skip incomplete questions
                    
                    # VERIFICATION SKIPPED - Trust Gemini generation, accept question directly
                    # No need to verify with model (82% accuracy causes false rejections)
                    verified_questions.append(generated_question)
                    all_generated_questions.append(generated_question)  # Track for deduplication
                    
                    # If we have enough, break early
                    if len(verified_questions) >= count:
                        break
                    
                    # OLD VERIFICATION CODE COMMENTED OUT:
                    # # Add to pending batch for Bloom verification
                    # pending_questions.append(generated_question)
                    # 
                    # # When batch is full or we have enough, verify in batch
                    # if len(pending_questions) >= batch_size or (len(verified_questions) + len(pending_questions) >= count):
                    #     # Batch Bloom prediction for speed
                    #     verify_start = time.time()
                    #     print(f"[TERMINAL] DEBUG: Verifying batch of {len(pending_questions)} questions for {level}...")
                    #     batch_analysis = analyze_exam_complete(pending_questions, analysis_model, analysis_tokenizer, f"Batch {level}")
                    #     batch_predictions = batch_analysis.get('predictions', [])
                    #     verify_time = time.time() - verify_start
                    #     print(f"[TERMINAL] DEBUG: Batch verification completed in {verify_time:.2f} seconds")
                    #     
                    #     # Process batch results
                    #     for i, pending_q in enumerate(pending_questions):
                    #         if i < len(batch_predictions):
                    #             predicted_level = batch_predictions[i]
                    #             
                    #             # If it matches the intended level, use it
                    #             if predicted_level == level:
                    #                 verified_questions.append(pending_q)
                    #                 all_generated_questions.append(pending_q)  # Track for deduplication
                    #             # If we're running out of attempts, be more lenient
                    #             elif attempts >= max_attempts - 2:
                    #                 # Accept questions that are close
                    #                 verified_questions.append(pending_q)
                    #                 all_generated_questions.append(pending_q)
                    #     
                    #     # Clear pending batch
                    #     pending_questions = []
                    #     
                    #     # If we have enough, break early
                    #     if len(verified_questions) >= count:
                    #         break
                            
                except Exception as e:
                    # If generation fails (including duplicate rejection), skip and try again
                    import traceback
                    error_msg = str(e)
                    error_full = traceback.format_exc()
                    
                    continue
            
            # VERIFICATION SKIPPED - No need to process pending questions
            # OLD CODE COMMENTED OUT:
            # # Process any remaining pending questions in final batch
            # if pending_questions and len(verified_questions) < count:
            #     batch_analysis = analyze_exam_complete(pending_questions, analysis_model, analysis_tokenizer, f"Final Batch {level}")
            #     batch_predictions = batch_analysis.get('predictions', [])
            #     
            #     for i, pending_q in enumerate(pending_questions):
            #         if i < len(batch_predictions) and len(verified_questions) < count:
            #             predicted_level = batch_predictions[i]
            #             if predicted_level == level:
            #                 verified_questions.append(pending_q)
            #                 all_generated_questions.append(pending_q)
            #             elif attempts >= max_attempts - 1:  # Very lenient on final batch
            #                 verified_questions.append(pending_q)
            #                 all_generated_questions.append(pending_q)
            
            # Add verified questions
            for q in verified_questions:
                new_questions.append({
                    'question': q,
                    'bloom_level': level,
                    'source': 'generated'
                })
        
        # Combine original and new questions (with removals)
        improved_questions = []
        
        # Step 0: Pre-compute simple per-question frequency within this exam.
        # This is used as one factor in the removal quality score so that
        # questions that appear multiple times are treated as "stronger" or
        # more established than one-off very short items.
        freq_counter = Counter()
        for q in original_questions:
            if isinstance(q, str):
                freq_counter[q.strip()] += 1
        
        # Step 1: Group original questions by their Bloom levels
        predictions = original_analysis.get('predictions', [])
        questions_by_level = {level: [] for level in BLOOM_LEVELS}
        
        for i, q in enumerate(original_questions):
            bloom_level = predictions[i] if i < len(predictions) else "Unknown"
            if bloom_level in BLOOM_LEVELS:
                questions_by_level[bloom_level].append({
                    'question': q,
                    'bloom_level': bloom_level,
                    'source': 'original',
                    'index': i
                })
        
        # Step 2: Remove excessive questions from over-represented levels
        for level, count_to_remove in removals.items():
            if level not in questions_by_level:
                continue
            
            questions = questions_by_level[level]
            original_count = len(questions)
            if count_to_remove <= 0 or original_count == 0:
                continue
            
            if original_count <= count_to_remove:
                # Remove all questions for this level if count_to_remove exceeds available
                questions_by_level[level] = []
                continue
            
            # ------------------------------------------------------------------
            # Multi-factor quality-based removal:
            #   - Length: longer questions tend to be richer (0..1, normalized)
            #   - Redundancy: questions highly similar to others in this level
            #                 are penalized; more unique ones are preferred.
            #   - Frequency: questions that appear multiple times in this exam
            #                are slightly preferred over one-off short items.
            # We remove the questions with the LOWEST overall quality score.
            # ------------------------------------------------------------------
            texts = [item['question'] for item in questions]
            lengths = [len(t) for t in texts]
            min_len = min(lengths) if lengths else 0
            max_len = max(lengths) if lengths else 0
            len_range = max(max_len - min_len, 1)
            
            # Tokenize once per question for redundancy estimates
            token_sets = []
            for t in texts:
                if isinstance(t, str):
                    token_sets.append(set(t.lower().split()))
                else:
                    token_sets.append(set())
            
            redundancy_raw = [0.0] * original_count
            for i in range(original_count):
                tokens_i = token_sets[i]
                if not tokens_i or original_count == 1:
                    redundancy_raw[i] = 0.0
                    continue
                sim_sum = 0.0
                compare_n = 0
                for j in range(original_count):
                    if i == j:
                        continue
                    tokens_j = token_sets[j]
                    if not tokens_j:
                        continue
                    inter = len(tokens_i & tokens_j)
                    union = len(tokens_i | tokens_j) or 1
                    sim_sum += inter / union
                    compare_n += 1
                redundancy_raw[i] = (sim_sum / compare_n) if compare_n > 0 else 0.0
            
            # Normalize redundancy into a "uniqueness" score where 1.0 = most unique
            min_red = min(redundancy_raw) if redundancy_raw else 0.0
            max_red = max(redundancy_raw) if redundancy_raw else 0.0
            red_range = max(max_red - min_red, 1e-6)
            uniqueness_scores = [
                1.0 - ((r - min_red) / red_range) for r in redundancy_raw
            ]
            
            # Frequency score: normalize counts across this level
            freq_values = []
            for t in texts:
                freq_values.append(freq_counter.get(t.strip() if isinstance(t, str) else "", 1))
            min_freq = min(freq_values) if freq_values else 1
            max_freq = max(freq_values) if freq_values else 1
            freq_range = max(max_freq - min_freq, 1)
            
            quality_scores = []
            for idx, t in enumerate(texts):
                # Length normalized 0..1
                length_norm = (lengths[idx] - min_len) / float(len_range) if len_range > 0 else 0.5
                # Uniqueness already 0..1
                uniq_norm = uniqueness_scores[idx] if uniqueness_scores else 0.5
                # Frequency normalized 0..1
                f_val = freq_values[idx]
                freq_norm = (f_val - min_freq) / float(freq_range) if freq_range > 0 else 0.5
                
                # Weights: favor richer, unique, and slightly more frequent questions
                quality = (0.5 * length_norm) + (0.3 * uniq_norm) + (0.2 * freq_norm)
                quality_scores.append((quality, idx))
            
            # Sort ascending by quality and remove the weakest items first
            quality_scores.sort(key=lambda x: x[0])
            to_remove_indices = {idx for _, idx in quality_scores[:count_to_remove]}
            
            filtered = [
                item for idx, item in enumerate(questions)
                if idx not in to_remove_indices
            ]
            questions_by_level[level] = filtered
        
        # Step 3: Add remaining original questions (after removals)
        for level in BLOOM_LEVELS:
            improved_questions.extend(questions_by_level[level])
        
        # Step 4: Add new generated questions
        improved_questions.extend(new_questions)
        
        # Re-analysis removed - trusting Gemini's bloom_level assignments
        # Distribution and quality score are now calculated from stored bloom_level values
        
        total_time = time.time() - start_time
        
        # improved_analysis is no longer needed - display function calculates from stored values
        # improvement_delta will be calculated in display function from stored bloom_level values
        
        return {
            'original_analysis': original_analysis,
            'improved_analysis': None,  # No longer used - calculated from stored bloom_level values
            'original_questions': original_questions,
            'improved_questions': improved_questions,
            'new_questions': new_questions,
            'changes_made': changes,
            'improvement_delta': None,  # Calculated in display function from stored values
            'total_questions': len(improved_questions)
        }
    
    def clear_content(self):
        """Clear all stored content."""
        self.chunks = []
        self.embeddings = None


def search_web_content(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web for content and return text chunks.
    Uses DuckDuckGo as a free search engine.
    
    Args:
        query: Search query
        num_results: Number of results to retrieve
    
    Returns:
        List of dictionaries with 'title', 'url', and 'text' keys
    """
    if not query or not query.strip():
        print("Error: Empty search query provided")
        return []
    
    try:
        from ddgs import DDGS
        
        results = []
        with DDGS() as ddgs:
            # Try text search first
            try:
                search_results = ddgs.text(query, max_results=num_results)
                
                for result in search_results:
                    title = result.get('title', '')
                    text = result.get('body', '')
                    url = result.get('href', '')
                    
                    # Only add results with actual content
                    if title or text:
                        results.append({
                            'title': title,
                            'url': url,
                            'text': text
                        })
            except Exception as text_search_error:
                print(f"Text search failed: {text_search_error}")
                # Try news search as fallback
                try:
                    search_results = ddgs.news(query, max_results=num_results)
                    for result in search_results:
                        title = result.get('title', '')
                        text = result.get('body', '')
                        url = result.get('href', '')
                        if title or text:
                            results.append({
                                'title': title,
                                'url': url,
                                'text': text
                            })
                except Exception as news_search_error:
                    print(f"News search also failed: {news_search_error}")
        
        if not results:
            print(f"No results found for query: '{query}'. This might be due to:")
            print("  1. Network connectivity issues")
            print("  2. DuckDuckGo rate limiting")
            print("  3. The search query being too specific or unusual")
            print("  4. Temporary service unavailability")
        
        return results
    except ImportError:
        # Fallback: return empty or use requests
        print("ERROR: ddgs not installed. Install with: pip install ddgs")
        return []
    except Exception as e:
        print(f"Error in web search: {e}")
        print(f"Query was: '{query}'")
        import traceback
        traceback.print_exc()
        return []

