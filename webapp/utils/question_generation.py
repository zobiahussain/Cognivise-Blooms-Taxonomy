import os
import pandas as pd
from utils.bloom_analyzer_complete import BLOOM_LEVELS, IDEAL_DISTRIBUTION, analyze_exam
import numpy as np

def _generate_questions_template(bloom_level, topic, num_questions=1):
    templates = {
        'Remembering': [
            "Define the term {concept} in {topic}.",
            "What is {concept} in {topic}?",
            "List the main components of {concept}.",
            "Identify the key features of {concept}.",
            "State the purpose of {concept} in {topic}.",
        ],
        'Understanding': [
            "Explain how {concept} works in {topic}.",
            "Describe the purpose of {concept}.",
            "Why is {concept} important in {topic}?",
            "Summarize the main idea of {concept}.",
            "Explain the difference between {concept1} and {concept2}.",
        ],
        'Applying': [
            "Implement {concept} to solve a problem in {topic}.",
            "Use {concept} to demonstrate {task}.",
            "Apply {concept} in a practical scenario.",
            "How would you use {concept} in real-world {topic}?",
            "Write a program using {concept} in {topic}.",
        ],
        'Analyzing': [
            "Compare {concept1} and {concept2} in {topic}.",
            "Analyze the relationship between {concept1} and {concept2}.",
            "What are the differences between {concept1} and {concept2}?",
            "Examine the advantages and disadvantages of {concept}.",
            "Differentiate between {concept1} and {concept2} in {topic}.",
        ],
        'Evaluating': [
            "Evaluate the effectiveness of {concept} in {topic}.",
            "Justify why {concept} is better than {alternative}.",
            "Assess the impact of {concept} on {topic}.",
            "Which is more efficient: {concept1} or {concept2}? Justify.",
            "Critique the use of {concept} in {topic}.",
        ],
        'Creating': [
            "Design a new {artifact} using {concept}.",
            "Develop a solution to {problem} in {topic}.",
            "Create a {artifact} that demonstrates {concept}.",
            "Construct a system using {concept} in {topic}.",
            "Formulate a new approach to {task} in {topic}.",
        ]
    }
    # Group concepts into comparable categories to avoid nonsensical comparisons
    # Data structures (can be compared with each other)
    data_structures = ['array', 'list', 'linked list', 'stack', 'queue', 'tree', 'graph', 'hash table', 'binary search tree', 'heap']
    # Algorithms (can be compared with each other)
    algorithms = ['sorting algorithm', 'searching', 'breadth-first search', 'depth-first search', 'dynamic programming', 'greedy algorithm']
    # Programming concepts (can be compared with each other)
    programming_concepts = ['variable', 'function', 'loop', 'recursion', 'string']
    # General concepts (use alone or with same category)
    general_concepts = ['algorithm']
    
    # Combined list for single-concept questions
    concepts = data_structures + algorithms + programming_concepts + general_concepts
    
    # Concept pairs that make sense to compare
    comparable_pairs = [
        # Data structure comparisons
        ('array', 'linked list'), ('stack', 'queue'), ('tree', 'graph'),
        ('hash table', 'binary search tree'), ('array', 'list'),
        # Algorithm comparisons
        ('breadth-first search', 'depth-first search'),
        ('dynamic programming', 'greedy algorithm'),
        ('sorting algorithm', 'searching'),
        # Programming concept comparisons
        ('loop', 'recursion'), ('array', 'string'),
    ]
    tasks = ['data sorting', 'searching', 'data management', 'problem-solving', 'optimization']
    problems = ['data organization', 'efficiency', 'scalability', 'performance']
    artifacts = ['algorithm', 'program', 'system', 'data structure', 'solution']
    alternatives = ['traditional approaches', 'other methods', 'conventional techniques']
    generated_questions = []
    for i in range(num_questions):
        template = np.random.choice(templates[bloom_level])
        question = template
        if '{concept}' in question:
            concept = np.random.choice(concepts)
            question = question.replace('{concept}', concept)
        if '{concept1}' in question:
            if '{concept2}' in question:
                # For comparison questions, use predefined comparable pairs
                if comparable_pairs:
                    pair = comparable_pairs[np.random.randint(len(comparable_pairs))]
                    concept1, concept2 = pair
                    question = question.replace('{concept1}', concept1)
                    question = question.replace('{concept2}', concept2)
                else:
                    # Fallback: pick from same category
                    category = np.random.choice([data_structures, algorithms, programming_concepts])
                    if len(category) >= 2:
                        concept1, concept2 = np.random.choice(category, size=2, replace=False)
                        question = question.replace('{concept1}', concept1)
                        question = question.replace('{concept2}', concept2)
                    else:
                        # Single concept fallback
                        concept1 = np.random.choice(concepts)
                        question = question.replace('{concept1}', concept1)
                        question = question.replace('{concept2}', 'related concept')
            else:
                # Single concept
                concept1 = np.random.choice(concepts)
                question = question.replace('{concept1}', concept1)
        if '{topic}' in question:
            question = question.replace('{topic}', topic)
        if '{task}' in question:
            question = question.replace('{task}', np.random.choice(tasks))
        if '{problem}' in question:
            question = question.replace('{problem}', np.random.choice(problems))
        if '{artifact}' in question:
            question = question.replace('{artifact}', np.random.choice(artifacts))
        if '{alternative}' in question:
            question = question.replace('{alternative}', np.random.choice(alternatives))
        generated_questions.append(question)
    return generated_questions


def generate_questions_improved(bloom_level, topic, num_questions=1, model=None, tokenizer=None):
    """
    Generate questions for a single Bloom level and topic.
    
    Primary path:
        - Uses the same RAG + Gemini (or local generator) stack as the main exam
          generation flows, via RAGExamGenerator.generate_exam_from_content.
    Fallback:
        - If RAG or Gemini/local generation fails for any reason, falls back to the
          lightweight template-based generator (_generate_questions_template).
    """
    from utils.rag_exam_generator import RAGExamGenerator
    import os
    
    bloom_level = bloom_level or "Understanding"
    topic = topic or "the subject"
    num_questions = max(1, int(num_questions or 1))
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    gemini_available = bool(gemini_key)
    
    try:
        # Initialize RAG generator - use Gemini if available, otherwise local model
        if gemini_available:
            rag_generator = RAGExamGenerator(
                llm_api="gemini",
                use_local_vector_store=True,
                api_key=gemini_key
            )
        else:
            rag_generator = RAGExamGenerator(
                llm_api="local",
                use_local_vector_store=True,
                use_optimized_generation=True
            )
        
        # Treat the topic/description as content for RAG; this keeps the API consistent
        # with other flows but with a minimal "corpus".
        content_text = f"Topic: {topic.strip()}\n\nGenerate exam questions for this topic."
        rag_generator.add_content(
            content_text,
            source_type="text",
            metadata={"topic": topic, "source": "manual_generate"}
        )
        
        # Use the same generation pipeline as full exams, but constrained to a single level
        questions = rag_generator.generate_exam_from_content(
            total_questions=num_questions,
            topic=topic,
            specific_bloom_level=bloom_level
        )
        
        # Extract plain question texts
        return [q.get("question", "").strip() for q in questions if q.get("question")]
    
    except Exception as e:
        # Log and fall back to the original template-based generator
        return _generate_questions_template(bloom_level, topic, num_questions)

def improve_exam_smart(original_questions, model, tokenizer, topic="Computer Science", exam_name="Exam", analysis_result=None):
    """
    Smart exam improvement using LLM with original questions as content context.
    
    This function uses RAG to generate new questions based on the concepts and content
    from the original exam questions, ensuring new questions are relevant and related.
    
    Args:
        original_questions: List of original exam questions (used as content)
        model: Model for analysis
        tokenizer: Tokenizer for analysis
        topic: Subject topic
        exam_name: Name of the exam
        analysis_result: Optional pre-computed analysis result to avoid duplicate analysis
    
    Returns:
        Dictionary with improvement results
    """
    from utils.rag_exam_generator import RAGExamGenerator
    import os
    
    # Use provided analysis result if available to avoid duplicate analysis
    if analysis_result is not None:
        analysis = analysis_result
    else:
        analysis = analyze_exam(original_questions, model, tokenizer)
    
    # Improve_exam_with_rag now handles all distribution logic
    # based on analysis recommendations, so we don't need to pre-calculate changes here
    
    # Use RAG with original questions as content instead of templates
    # Initialize RAG generator - use Gemini if available, otherwise local model
    gemini_key = os.getenv("GEMINI_API_KEY")
    gemini_available = bool(gemini_key)
    
    if gemini_available:
        rag_generator = RAGExamGenerator(
            llm_api="gemini",
            use_local_vector_store=True,
            api_key=gemini_key
        )
    else:
        # Fallback to local model
        rag_generator = RAGExamGenerator(
            llm_api="local",
            use_local_vector_store=True,
            use_optimized_generation=True
        )
    
    # Add original questions as content so the LLM can generate related questions
    # Combine all questions into content text
    questions_text = "\n\n".join([f"Question {i+1}: {q}" for i, q in enumerate(original_questions)])
    rag_generator.add_content(
        questions_text,
        source_type="text",
        metadata={"topic": topic, "source": "original_exam_questions", "exam_name": exam_name}
    )
    
    # Use RAG to generate improved exam based on original questions
    # Pass analysis_result to avoid duplicate analysis
    improvement = rag_generator.improve_exam_with_rag(
        original_questions,
        model,
        tokenizer,
        topic=topic,
        exam_name=exam_name,
        use_analysis_model_for_gen=False,  # Use Gemini/local model for generation, not analysis model
        analysis_result=analysis  # Pass pre-computed analysis
    )
    
    return improvement
def export_improved_exam(improvement_result, export_dir, filename="improved_exam"):
    df_improved = pd.DataFrame(improvement_result['improved_questions'])
    csv_path = os.path.join(export_dir, f'{filename}.csv')
    df_improved.to_csv(csv_path, index=False)
    text_path = os.path.join(export_dir, f'{filename}.txt')
    with open(text_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("IMPROVED EXAM PAPER\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total Questions: {len(improvement_result['improved_questions'])}\n")
        f.write(f"Quality Score: {improvement_result['improved_analysis']['quality_score']:.1f}/100\n\n")
        by_level = {}
        for item in improvement_result['improved_questions']:
            level = item['bloom_level']
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(item['question'])
        question_number = 1
        for level in BLOOM_LEVELS:
            if level in by_level:
                f.write("-"*70 + "\n")
                f.write(f"{level.upper()} ({len(by_level[level])} questions)\n")
                f.write("-"*70 + "\n\n")
                for q in by_level[level]:
                    f.write(f"{question_number}. {q}\n\n")
                    question_number += 1
        f.write("="*70 + "\nEND OF EXAM\n" + "="*70 + "\n")
    return csv_path, text_path

