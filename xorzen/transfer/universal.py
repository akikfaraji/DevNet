"""
XORZENX Universal Transfer System
Transfer from ANY model: GPT-2, LLaMA, Mistral, Qwen, Gemma, etc.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple
import warnings


# List of recommended models by size/quality
RECOMMENDED_MODELS = {
    'tiny': [
        'gpt2',                              # 124M - Fast baseline
        'distilgpt2',                        # 82M - Distilled GPT-2
    ],
    'small': [
        'gpt2-medium',                       # 355M
        'microsoft/phi-2',                   # 2.7B - Very strong small model
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0', # 1.1B - LLaMA architecture
    ],
    'medium': [
        'gpt2-large',                        # 774M
        'microsoft/phi-1_5',                 # 1.3B
        'stabilityai/stablelm-3b-4e1t',     # 3B
    ],
    'large': [
        'gpt2-xl',                           # 1.5B
        'meta-llama/Llama-2-7b-hf',         # 7B - Need access token
        'mistralai/Mistral-7B-v0.1',        # 7B - Very strong
        'Qwen/Qwen-7B',                      # 7B - Multilingual
        'google/gemma-7b',                   # 7B - Google's model
    ],
    'huge': [
        'meta-llama/Llama-2-13b-hf',        # 13B
        'mistralai/Mixtral-8x7B-v0.1',      # 47B - MoE model
        'meta-llama/Meta-Llama-3-70B',      # 70B - SOTA
    ]
}


class UniversalTransfer:
    """Transfer from ANY HuggingFace model to XORZENX."""
    
    def __init__(
        self,
        teacher_name: str = 'microsoft/phi-2',
        device: str = 'auto',
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize with any teacher model.
        
        Args:
            teacher_name: Any HuggingFace model name
            device: 'auto', 'cuda', 'cpu'
            load_in_8bit: Load large models in 8-bit (saves memory)
            load_in_4bit: Load large models in 4-bit (saves more memory)
        """
        print(f"🚀 Loading teacher: {teacher_name}")
        
        self.teacher_name = teacher_name
        
        # Auto-detect device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Load with quantization if requested
        load_kwargs = {}
        if load_in_8bit:
            load_kwargs['load_in_8bit'] = True
            print("  Using 8-bit quantization (saves memory)")
        elif load_in_4bit:
            load_kwargs['load_in_4bit'] = True
            print("  Using 4-bit quantization (saves lots of memory)")
        
        try:
            # Try loading as causal LM first (most common)
            self.teacher = AutoModelForCausalLM.from_pretrained(
                teacher_name,
                trust_remote_code=True,
                **load_kwargs
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                teacher_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if not load_in_8bit and not load_in_4bit:
                self.teacher.to(self.device)
            
            self.teacher.eval()
            
            # Detect architecture
            self.architecture = self._detect_architecture()
            print(f"✓ Loaded {teacher_name}")
            print(f"  Architecture: {self.architecture}")
            print(f"  Params: {sum(p.numel() for p in self.teacher.parameters()):,}")
            
        except Exception as e:
            print(f"✗ Failed to load {teacher_name}: {e}")
            print("\nTrying fallback to gpt2...")
            self.__init__('gpt2', device='auto')
    
    def _detect_architecture(self) -> str:
        """Auto-detect model architecture."""
        model_type = self.teacher.config.model_type
        
        arch_map = {
            'gpt2': 'GPT-2',
            'gpt_neo': 'GPT-Neo',
            'gptj': 'GPT-J',
            'llama': 'LLaMA',
            'mistral': 'Mistral',
            'mixtral': 'Mixtral (MoE)',
            'phi': 'Phi',
            'qwen': 'Qwen',
            'gemma': 'Gemma',
            'opt': 'OPT',
            'bloom': 'BLOOM',
            'falcon': 'Falcon',
        }
        
        return arch_map.get(model_type, f'Unknown ({model_type})')
    
    def extract_embeddings(self) -> Dict[str, torch.Tensor]:
        """Extract embeddings - works for any architecture. Returns clones (safe to modify)."""
        embeddings = {}
        
        # Try different embedding locations
        if hasattr(self.teacher, 'transformer'):  # GPT-2, GPT-Neo
            if hasattr(self.teacher.transformer, 'wte'):
                embeddings['token'] = self.teacher.transformer.wte.weight.data.clone()
            if hasattr(self.teacher.transformer, 'wpe'):
                embeddings['position'] = self.teacher.transformer.wpe.weight.data.clone()
        
        elif hasattr(self.teacher, 'model'):  # LLaMA, Mistral, etc.
            if hasattr(self.teacher.model, 'embed_tokens'):
                embeddings['token'] = self.teacher.model.embed_tokens.weight.data.clone()
        
        elif hasattr(self.teacher, 'gpt_neox'):  # GPT-NeoX
            embeddings['token'] = self.teacher.gpt_neox.embed_in.weight.data.clone()
        
        elif hasattr(self.teacher, 'embeddings'):  # BERT-style
            embeddings['token'] = self.teacher.embeddings.word_embeddings.weight.data.clone()
        
        else:
            # Generic fallback - search for embedding layers
            for name, module in self.teacher.named_modules():
                if isinstance(module, nn.Embedding) and 'token' in name.lower():
                    embeddings['token'] = module.weight.data.clone()
                    break
        
        if 'token' not in embeddings:
            raise ValueError(f"Could not find token embeddings in {self.teacher_name}")
        
        return embeddings
    
    def extract_layers(self) -> List[nn.Module]:
        """Extract transformer layers - works for any architecture."""
        layers = []
        
        # Try different layer locations
        if hasattr(self.teacher, 'transformer'):  # GPT-2
            if hasattr(self.teacher.transformer, 'h'):
                layers = list(self.teacher.transformer.h)
        
        elif hasattr(self.teacher, 'model'):  # LLaMA, Mistral
            if hasattr(self.teacher.model, 'layers'):
                layers = list(self.teacher.model.layers)
        
        elif hasattr(self.teacher, 'gpt_neox'):  # GPT-NeoX
            if hasattr(self.teacher.gpt_neox, 'layers'):
                layers = list(self.teacher.gpt_neox.layers)
        
        elif hasattr(self.teacher, 'encoder'):  # BERT
            if hasattr(self.teacher.encoder, 'layer'):
                layers = list(self.teacher.encoder.layer)
        
        if not layers:
            raise ValueError(f"Could not find transformer layers in {self.teacher_name}")
        
        return layers
    
    def extract_mlp_weights(self, layer) -> Dict[str, torch.Tensor]:
        """Extract MLP/FFN weights - architecture-agnostic. Returns clones."""
        weights = {}
        
        # Try different MLP names
        mlp_attrs = ['mlp', 'feed_forward', 'ffn', 'fc', 'intermediate']
        
        for attr in mlp_attrs:
            if hasattr(layer, attr):
                mlp = getattr(layer, attr)
                
                # Get input projection
                for name in ['up_proj', 'w1', 'gate_proj', 'fc1', 'c_fc', 'dense', 'wi']:
                    if hasattr(mlp, name):
                        weights['w_in'] = getattr(mlp, name).weight.data.clone()
                        break
                
                # Get output projection
                for name in ['down_proj', 'w2', 'fc2', 'c_proj', 'wo']:
                    if hasattr(mlp, name):
                        weights['w_out'] = getattr(mlp, name).weight.data.clone()
                        break
                
                if 'w_in' in weights and 'w_out' in weights:
                    return weights
        
        raise ValueError(f"Could not extract MLP weights from layer")
    
    def smart_transfer(self, xorzen_model, verbose: bool = True):
        """Universal transfer that works with any teacher model."""
        if verbose:
            print(f"\n⚡ SMART TRANSFER: {self.teacher_name} → XORZENX")
        
        # 1. Transfer embeddings
        if verbose:
            print("  [1/3] Embeddings...", end='')
        
        teacher_emb = self.extract_embeddings()
        teacher_vocab_size = teacher_emb['token'].shape[0]
        teacher_hidden_dim = teacher_emb['token'].shape[1]
        xorzen_hidden_dim = xorzen_model.config.hidden_size
        
        # Resize XORZENX vocab
        xorzen_model.token_embedding = nn.Embedding(teacher_vocab_size, xorzen_hidden_dim)
        xorzen_model.lm_head = nn.Linear(xorzen_hidden_dim, teacher_vocab_size, bias=False)
        
        # Project if dimensions differ
        if teacher_hidden_dim != xorzen_hidden_dim:
            proj = nn.Linear(teacher_hidden_dim, xorzen_hidden_dim, bias=False)
            nn.init.orthogonal_(proj.weight)
            # Move projection to same device as teacher embeddings
            proj = proj.to(teacher_emb['token'].device)
            with torch.no_grad():
                projected = proj(teacher_emb['token'])
            xorzen_model.token_embedding.weight.data = projected.cpu()
        else:
            xorzen_model.token_embedding.weight.data = teacher_emb['token'].cpu()
        
        xorzen_model.lm_head.weight = xorzen_model.token_embedding.weight
        xorzen_model.config.vocab_size = teacher_vocab_size
        
        if verbose:
            print(f" ✓ ({teacher_vocab_size:,} tokens)")
        
        # 2. Transfer to experts
        if verbose:
            print("  [2/3] Experts...", end='')
        
        teacher_layers = self.extract_layers()
        num_experts = xorzen_model.moe.num_experts
        experts_per_layer = max(1, num_experts // len(teacher_layers))
        
        # Skip expert seeding in test_mode (no disk manager or experts list)
        if xorzen_model.moe.test_mode:
            if verbose:
                print(f" ✓ (skipped in test_mode)")
        else:
            expert_id = 0
            for layer in teacher_layers:
                if expert_id >= num_experts:
                    break
                
                try:
                    mlp_weights = self.extract_mlp_weights(layer)
                    
                    for variant in range(experts_per_layer):
                        if expert_id >= num_experts:
                            break
                        
                        expert = xorzen_model.moe.cache.get(expert_id)
                        if expert is None:
                            expert = xorzen_model.moe.disk_manager.load_expert(expert_id)

                        noise = 0.01 * variant
                        
                        # w_in shape: [intermediate, hidden] — already correct for Linear weight
                        expert.gate_proj.weight.data = self._adapt_weight(
                            mlp_weights['w_in'],
                            expert.gate_proj.weight.shape,
                            noise
                        )
                        expert.up_proj.weight.data = self._adapt_weight(
                            mlp_weights['w_in'],
                            expert.up_proj.weight.shape,
                            noise * 0.5
                        )
                        # w_out shape: [hidden, intermediate] — correct for Linear weight
                        expert.down_proj.weight.data = self._adapt_weight(
                            mlp_weights['w_out'],
                            expert.down_proj.weight.shape,
                            noise
                        )
                        
                        xorzen_model.moe.disk_manager.save_expert(expert_id, expert)
                        xorzen_model.moe.cache.put(expert_id, expert)
                        expert_id += 1
                except Exception:
                    continue
            
            if verbose:
                print(f" ✓ ({expert_id} experts)")
        
        # 3. Smart router init
        if verbose:
            print("  [3/3] Router...", end='')
        
        router = xorzen_model.router
        
        # Balanced depth routing
        router.depth_router.weight.data *= 0.1
        router.expert_router.weight.data *= 0.1
        
        if verbose:
            print(" ✓")
        
        if verbose:
            print(f"\n✅ Transfer complete from {self.architecture}")
        
        return xorzen_model, self.tokenizer
    
    def _adapt_weight(self, source: torch.Tensor, target_shape: Tuple[int, int], noise: float = 0.0) -> torch.Tensor:
        """Adapt weight matrix to different dimensions."""
        src_h, src_w = source.shape
        tgt_h, tgt_w = target_shape
        
        result = torch.zeros(target_shape, dtype=source.dtype, device='cpu')
        min_h, min_w = min(src_h, tgt_h), min(src_w, tgt_w)
        result[:min_h, :min_w] = source[:min_h, :min_w].cpu()
        
        if tgt_h > src_h or tgt_w > src_w:
            result += torch.randn_like(result) * 0.01
        
        if noise > 0:
            result += torch.randn_like(result) * noise
        
        return result


def list_recommended_models():
    """Print recommended models by category."""
    print("\n📚 RECOMMENDED TEACHER MODELS")
    print("="*80)
    
    for category, models in RECOMMENDED_MODELS.items():
        print(f"\n{category.upper()}:")
        for model in models:
            print(f"  • {model}")
    
    print("\n💡 TIP: Larger teachers = better transfer, but slower")
    print("   Recommended: microsoft/phi-2 (2.7B, very strong)")
    print("   Alternative: mistralai/Mistral-7B-v0.1 (7B, SOTA)")


if __name__ == "__main__":
    list_recommended_models()
    
    print("\n🧪 Test transfer from Phi-2...")
    import xorzen
    
    model = xorzen.zero_1M()
    transfer = UniversalTransfer('microsoft/phi-2')
    model, tokenizer = transfer.smart_transfer(model)
    
    print("\n✅ Transfer successful!")
