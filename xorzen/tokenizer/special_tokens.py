"""
XORZENX Ultimate Special Tokens - For AGI-Level Intelligence
Comprehensive token system for advanced reasoning, multimodal, tool use, and more.
"""

# Core tokens
CORE_TOKENS = [
    "<pad>",           # 0 - Padding
    "<unk>",           # 1 - Unknown token
    "<s>",             # 2 - Start of sequence
    "</s>",            # 3 - End of sequence
    "<mask>",          # 4 - Masked token (for MLM)
    "<|endoftext|>",   # 5 - Document separator
]

# Reasoning & Chain-of-Thought tokens
REASONING_TOKENS = [
    "<|think|>",           # Start thinking/reasoning
    "<|/think|>",          # End thinking
    "<|step|>",            # Reasoning step marker
    "<|conclusion|>",      # Final conclusion
    "<|hypothesis|>",      # Hypothesis formation
    "<|evidence|>",        # Evidence citation
    "<|analysis|>",        # Analysis section
    "<|reflect|>",         # Self-reflection
    "<|critique|>",        # Self-critique
    "<|verify|>",          # Verification step
    "<|alternative|>",     # Alternative solution
    "<|confidence|>",      # Confidence level marker
]

# Role tokens (multi-agent)
ROLE_TOKENS = [
    "<|system|>",          # System message
    "<|user|>",            # User input
    "<|assistant|>",       # Assistant response
    "<|function|>",        # Function call result
    "<|tool|>",            # Tool output
    "<|agent|>",           # Agent action
    "<|planner|>",         # Planning agent
    "<|critic|>",          # Critic agent
    "<|executor|>",        # Executor agent
    "<|researcher|>",      # Research agent
    "<|coder|>",           # Coding agent
    "<|teacher|>",         # Teaching agent
]

# Tool use & function calling
TOOL_TOKENS = [
    "<|call|>",            # Function call start
    "<|/call|>",           # Function call end
    "<|args|>",            # Arguments
    "<|result|>",          # Result
    "<|error|>",           # Error message
    "<|search|>",          # Web search
    "<|code|>",            # Code execution
    "<|python|>",          # Python code
    "<|bash|>",            # Bash command
    "<|sql|>",             # SQL query
    "<|api|>",             # API call
    "<|file|>",            # File operation
    "<|browse|>",          # Web browsing
    "<|math|>",            # Math calculation
]

# Multimodal tokens
MULTIMODAL_TOKENS = [
    "<|image|>",           # Image input
    "<|/image|>",          # Image end
    "<|audio|>",           # Audio input
    "<|/audio|>",          # Audio end
    "<|video|>",           # Video input
    "<|/video|>",          # Video end
    "<|vision|>",          # Visual understanding
    "<|caption|>",         # Image caption
    "<|ocr|>",             # OCR text
    "<|detect|>",          # Object detection
    "<|segment|>",         # Segmentation
    "<|3d|>",              # 3D understanding
]

# Memory & context tokens
MEMORY_TOKENS = [
    "<|remember|>",        # Store in memory
    "<|recall|>",          # Retrieve from memory
    "<|forget|>",          # Clear memory
    "<|context|>",         # Context window marker
    "<|summary|>",         # Summary of previous
    "<|history|>",         # Conversation history
    "<|reference|>",       # Reference to previous
    "<|cache|>",           # Cache this
]

# Instruction & control tokens
CONTROL_TOKENS = [
    "<|instruction|>",     # Instruction following
    "<|task|>",            # Task definition
    "<|goal|>",            # Goal specification
    "<|constraint|>",      # Constraint definition
    "<|priority|>",        # Priority level
    "<|urgent|>",          # Urgent task
    "<|background|>",      # Background task
    "<|wait|>",            # Wait for input
    "<|continue|>",        # Continue generation
    "<|stop|>",            # Stop generation
    "<|retry|>",           # Retry action
]

# Structured output tokens
STRUCTURE_TOKENS = [
    "<|json|>",            # JSON output
    "<|/json|>",           # JSON end
    "<|yaml|>",            # YAML output
    "<|xml|>",             # XML output
    "<|markdown|>",        # Markdown output
    "<|latex|>",           # LaTeX output
    "<|table|>",           # Table data
    "<|list|>",            # List data
    "<|code_block|>",      # Code block
    "<|quote|>",           # Quotation
]

# Emotional & personality tokens
PERSONALITY_TOKENS = [
    "<|friendly|>",        # Friendly tone
    "<|professional|>",    # Professional tone
    "<|casual|>",          # Casual tone
    "<|empathetic|>",      # Empathetic response
    "<|humorous|>",        # Humorous tone
    "<|serious|>",         # Serious tone
    "<|creative|>",        # Creative mode
    "<|analytical|>",      # Analytical mode
]

# Language & translation
LANGUAGE_TOKENS = [
    "<|en|>",              # English
    "<|es|>",              # Spanish
    "<|fr|>",              # French
    "<|de|>",              # German
    "<|zh|>",              # Chinese
    "<|ja|>",              # Japanese
    "<|ar|>",              # Arabic
    "<|translate|>",       # Translation marker
    "<|detect_lang|>",     # Language detection
]

# Meta & safety tokens
META_TOKENS = [
    "<|safe|>",            # Safe content
    "<|unsafe|>",          # Unsafe content
    "<|fact|>",            # Factual statement
    "<|opinion|>",         # Opinion
    "<|uncertain|>",       # Uncertainty marker
    "<|citation|>",        # Citation needed
    "<|source|>",          # Source attribution
    "<|disclaimer|>",      # Disclaimer
    "<|warning|>",         # Warning
]

# Advanced reasoning tokens
ADVANCED_REASONING = [
    "<|analogical|>",      # Analogical reasoning
    "<|causal|>",          # Causal reasoning
    "<|deductive|>",       # Deductive reasoning
    "<|inductive|>",       # Inductive reasoning
    "<|abductive|>",       # Abductive reasoning
    "<|counterfactual|>",  # Counterfactual thinking
    "<|metacognitive|>",   # Metacognition
    "<|bayesian|>",        # Bayesian reasoning
]

# Planning & execution tokens
PLANNING_TOKENS = [
    "<|plan|>",            # Plan generation
    "<|/plan|>",           # Plan end
    "<|step1|>",           # Step 1
    "<|step2|>",           # Step 2
    "<|step3|>",           # Step 3
    "<|substep|>",         # Sub-step
    "<|checkpoint|>",      # Checkpoint
    "<|milestone|>",       # Milestone
    "<|progress|>",        # Progress update
    "<|complete|>",        # Task complete
]

# Learning & adaptation tokens
LEARNING_TOKENS = [
    "<|learn|>",           # Learning signal
    "<|feedback|>",        # Feedback incorporation
    "<|correct|>",         # Correction
    "<|improve|>",         # Improvement suggestion
    "<|pattern|>",         # Pattern recognition
    "<|generalize|>",      # Generalization
    "<|specialize|>",      # Specialization
]

# Dialogue & interaction tokens
DIALOGUE_TOKENS = [
    "<|question|>",        # Question
    "<|answer|>",          # Answer
    "<|clarify|>",         # Clarification request
    "<|confirm|>",         # Confirmation
    "<|acknowledge|>",     # Acknowledgment
    "<|suggest|>",         # Suggestion
    "<|recommend|>",       # Recommendation
    "<|agree|>",           # Agreement
    "<|disagree|>",        # Disagreement
]

# Domain-specific tokens
DOMAIN_TOKENS = [
    "<|medical|>",         # Medical domain
    "<|legal|>",           # Legal domain
    "<|finance|>",         # Finance domain
    "<|science|>",         # Science domain
    "<|math|>",            # Mathematics domain
    "<|history|>",         # History domain
    "<|literature|>",      # Literature domain
    "<|tech|>",            # Technology domain
]

# Combine all tokens
ALL_SPECIAL_TOKENS = (
    CORE_TOKENS +
    REASONING_TOKENS +
    ROLE_TOKENS +
    TOOL_TOKENS +
    MULTIMODAL_TOKENS +
    MEMORY_TOKENS +
    CONTROL_TOKENS +
    STRUCTURE_TOKENS +
    PERSONALITY_TOKENS +
    LANGUAGE_TOKENS +
    META_TOKENS +
    ADVANCED_REASONING +
    PLANNING_TOKENS +
    LEARNING_TOKENS +
    DIALOGUE_TOKENS +
    DOMAIN_TOKENS
)

# Total: 150+ special tokens for AGI-level capabilities

def get_all_special_tokens():
    """Get all special tokens."""
    return ALL_SPECIAL_TOKENS

def get_token_categories():
    """Get tokens organized by category."""
    return {
        'core': CORE_TOKENS,
        'reasoning': REASONING_TOKENS,
        'roles': ROLE_TOKENS,
        'tools': TOOL_TOKENS,
        'multimodal': MULTIMODAL_TOKENS,
        'memory': MEMORY_TOKENS,
        'control': CONTROL_TOKENS,
        'structure': STRUCTURE_TOKENS,
        'personality': PERSONALITY_TOKENS,
        'language': LANGUAGE_TOKENS,
        'meta': META_TOKENS,
        'advanced_reasoning': ADVANCED_REASONING,
        'planning': PLANNING_TOKENS,
        'learning': LEARNING_TOKENS,
        'dialogue': DIALOGUE_TOKENS,
        'domain': DOMAIN_TOKENS,
    }

def get_tokens_for_capability(capability: str):
    """Get tokens needed for a specific capability."""
    categories = get_token_categories()
    return categories.get(capability, [])


if __name__ == "__main__":
    print(f"Total special tokens: {len(ALL_SPECIAL_TOKENS)}")
    print("\nToken categories:")
    for category, tokens in get_token_categories().items():
        print(f"  {category}: {len(tokens)} tokens")
    
    print("\n Sample tokens:")
    print("  Reasoning:", REASONING_TOKENS[:5])
    print("  Tools:", TOOL_TOKENS[:5])
    print("  Multimodal:", MULTIMODAL_TOKENS[:5])
