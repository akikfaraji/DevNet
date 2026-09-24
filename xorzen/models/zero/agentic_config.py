
from dataclasses import dataclass, field
from xorzen.config import ModelConfig, ModelSize
from enum import Enum
from typing import List, Optional

@dataclass
class ZeroAgenticConfig(ModelConfig):
    """
    zero Agentic Configuration.
    Extends XORZENX config with recursive thinking and persistent memory capabilities.
    Formerly known as IGRIS (Intelligent General Recursive Inference System),
    now merged as an agentic sub-variant within the zero model family.
    """
    # --- Thinking Capabilities ---
    recurrence_depth: int = 1  # How many times to re-process difficult tokens
    adaptive_depth: bool = True

    # --- Agentic Capabilities (Bit Upgrades) ---
    use_latent_cot: bool = True      # Internal Monologue in latent space
    latent_cot_dim: int = 128        # Dimension of the "Reasoning State"
    num_action_slots: int = 16       # Native heads for tool use/actions
    use_self_critique: bool = True   # Enable feedback loop for refinement

    # --- Memory Capabilities ---
    memory_slots: int = 64
    memory_dim: int = 128

    # --- Advanced Bit Upgrades ---
    use_flash_ssm: bool = True
    use_gla: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.hidden_size > 2048:
            self.recurrence_depth = max(self.recurrence_depth, 2)

class ZeroAgenticSize(str, Enum):
    NANO = "zero-Agentic-Nano"
    MICRO = "zero-Agentic-Micro"
    MINI = "zero-Agentic-Mini"
    BASE = "zero-Agentic-Base"
    LARGE = "zero-Agentic-Large"
