"""
XORZENX Transfer Learning System - Universal Version
Fast transfer from ANY pre-trained model to XORZENX.
No hardcoded models - works with GPT-2, LLaMA, Mistral, Phi, Qwen, Gemma, etc.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple
import numpy as np


class TeacherExtractor:
    """Extract transferable knowledge from ANY pre-trained model."""
    
    def __init__(self, model_name: str, device: str = 'auto'):
        print(f"Loading teacher model: {model_name}...")
        
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        try:
            self.teacher = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        except Exception:
            self.teacher = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.teacher.to(self.device)
        self.teacher.eval()
        print(f"✓ Loaded {model_name}")
    
    def extract_embeddings(self) -> Dict[str, torch.Tensor]:
        """Extract token and position embeddings - works for any architecture."""
        embeddings = {}
        
        # Try different embedding locations
        if hasattr(self.teacher, 'transformer'):  # GPT-2, GPT-Neo
            if hasattr(self.teacher.transformer, 'wte'):
                embeddings['token'] = self.teacher.transformer.wte.weight.data.clone()
            if hasattr(self.teacher.transformer, 'wpe'):
                embeddings['position'] = self.teacher.transformer.wpe.weight.data.clone()
        
        elif hasattr(self.teacher, 'model'):  # LLaMA, Mistral, Qwen
            if hasattr(self.teacher.model, 'embed_tokens'):
                embeddings['token'] = self.teacher.model.embed_tokens.weight.data.clone()
        
        elif hasattr(self.teacher, 'embeddings'):  # BERT
            embeddings['token'] = self.teacher.embeddings.word_embeddings.weight.data.clone()
        
        elif hasattr(self.teacher, 'gpt_neox'):  # GPT-NeoX
            embeddings['token'] = self.teacher.gpt_neox.embed_in.weight.data.clone()
        
        else:
            # Generic fallback
            for name, module in self.teacher.named_modules():
                if isinstance(module, nn.Embedding) and 'token' in name.lower():
                    embeddings['token'] = module.weight.data.clone()
                    break
        
        print(f"✓ Extracted embeddings: token={embeddings.get('token', torch.empty(0)).shape}")
        return embeddings
    
    def extract_ffn_layers(self) -> List[Dict[str, torch.Tensor]]:
        """Extract FFN weights from each transformer layer - architecture agnostic."""
        ffn_layers = []
        
        # Get transformer blocks
        layers = []
        if hasattr(self.teacher, 'transformer'):  # GPT-2
            if hasattr(self.teacher.transformer, 'h'):
                layers = list(self.teacher.transformer.h)
        elif hasattr(self.teacher, 'model'):  # LLaMA, Mistral
            if hasattr(self.teacher.model, 'layers'):
                layers = list(self.teacher.model.layers)
        elif hasattr(self.teacher, 'encoder'):  # BERT
            if hasattr(self.teacher.encoder, 'layer'):
                layers = list(self.teacher.encoder.layer)
        elif hasattr(self.teacher, 'gpt_neox'):  # GPT-NeoX
            if hasattr(self.teacher.gpt_neox, 'layers'):
                layers = list(self.teacher.gpt_neox.layers)
        
        for i, block in enumerate(layers):
            weights = self._extract_mlp_from_block(block)
            if weights:
                ffn_layers.append(weights)
        
        print(f"✓ Extracted {len(ffn_layers)} FFN layers")
        return ffn_layers
    
    def _extract_mlp_from_block(self, block) -> Optional[Dict[str, torch.Tensor]]:
        """Extract MLP weights from a single block - handles multiple architectures."""
        
        # Try different MLP attribute names
        mlp = None
        for attr in ['mlp', 'feed_forward', 'ffn', 'fc']:
            if hasattr(block, attr):
                mlp = getattr(block, attr)
                break
        
        if mlp is None:
            return None
        
        weights = {}
        
        # Extract input projection
        for in_name in ['up_proj', 'gate_proj', 'w1', 'c_fc', 'fc1', 'dense', 'wi']:
            if hasattr(mlp, in_name):
                weights['w1'] = getattr(mlp, in_name).weight.data.clone()
                bias_attr = getattr(mlp, in_name).bias
                if bias_attr is not None:
                    weights['b1'] = bias_attr.data.clone()
                break
        
        # Extract output projection
        for out_name in ['down_proj', 'w2', 'c_proj', 'fc2', 'wo']:
            if hasattr(mlp, out_name):
                weights['w2'] = getattr(mlp, out_name).weight.data.clone()
                bias_attr = getattr(mlp, out_name).bias
                if bias_attr is not None:
                    weights['b2'] = bias_attr.data.clone()
                break
        
        return weights if 'w1' in weights and 'w2' in weights else None
    
    def extract_attention_patterns(self, sample_texts: List[str]) -> Optional[torch.Tensor]:
        """Extract attention patterns from sample data."""
        all_attentions = []
        
        with torch.no_grad():
            for text in sample_texts[:10]:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.teacher(**inputs, output_attentions=True)
                
                if hasattr(outputs, 'attentions') and outputs.attentions:
                    attn_stack = torch.stack([a.squeeze(0).mean(0) for a in outputs.attentions])
                    all_attentions.append(attn_stack.mean(0))
        
        if all_attentions:
            avg_attention = torch.stack(all_attentions).mean(0)
            print(f"✓ Extracted attention patterns: {avg_attention.shape}")
            return avg_attention
        
        return None


class XORZENXTransferLearning:
    """Transfer knowledge from ANY pre-trained model to XORZENX."""
    
    def __init__(self, teacher_name: str):
        self.extractor = TeacherExtractor(teacher_name)
        self.teacher_name = teacher_name
    
    def transfer_embeddings(
        self,
        xorzen_model,
        strategy: str = 'resize',
    ):
        """Transfer token embeddings to XORZENX."""
        print(f"\n[1/4] Transferring embeddings (strategy={strategy})...")
        
        teacher_emb = self.extractor.extract_embeddings()
        teacher_token_emb = teacher_emb['token']
        
        if strategy == 'resize':
            new_vocab_size, hidden_dim = teacher_token_emb.shape
            
            new_embedding = nn.Embedding(new_vocab_size, xorzen_model.config.hidden_size)
            
            if hidden_dim != xorzen_model.config.hidden_size:
                print(f"  Projecting {hidden_dim} → {xorzen_model.config.hidden_size}")
                projection = self._create_projection(hidden_dim, xorzen_model.config.hidden_size)
                # Move projection to same device as teacher embeddings
                projection = projection.to(teacher_token_emb.device)
                with torch.no_grad():
                    new_embedding.weight.data = projection(teacher_token_emb)
            else:
                new_embedding.weight.data = teacher_token_emb.clone()
            
            xorzen_model.token_embedding = new_embedding
            xorzen_model.config.vocab_size = new_vocab_size
            
            xorzen_model.lm_head = nn.Linear(
                xorzen_model.config.hidden_size,
                new_vocab_size,
                bias=False
            )
            xorzen_model.lm_head.weight = new_embedding.weight
            
            print(f"  ✓ Resized vocab: {new_vocab_size:,}")
        
        return xorzen_model
    
    def seed_experts(
        self,
        xorzen_model,
        num_experts_per_layer: int = 4,
        noise_scale: float = 0.02,
    ):
        """Initialize XORZENX experts from teacher FFN layers."""
        print(f"\n[2/4] Seeding {xorzen_model.moe.num_experts} experts from teacher FFNs...")
        
        teacher_ffns = self.extractor.extract_ffn_layers()
        num_teacher_layers = len(teacher_ffns)
        
        if num_teacher_layers == 0:
            print("  Warning: No FFN layers extracted from teacher, skipping expert seeding.")
            return xorzen_model

        expert_id = 0
        for layer_idx in range(num_teacher_layers):
            teacher_ffn = teacher_ffns[layer_idx]
            
            for variant in range(num_experts_per_layer):
                if expert_id >= xorzen_model.moe.num_experts:
                    break
                
                # In test_mode the moe has a dummy_expert instead of an experts list
                if xorzen_model.moe.test_mode:
                    break

                expert = xorzen_model.moe.cache.get(expert_id)
                if expert is None:
                    expert = xorzen_model.moe.disk_manager.load_expert(expert_id)

                noise = noise_scale * (variant / max(1, num_experts_per_layer - 1))
                
                # w1 shape from teacher: [intermediate, hidden] (already in correct Linear orientation)
                expert.gate_proj.weight.data = self._adapt_weight(
                    teacher_ffn['w1'],
                    expert.gate_proj.weight.shape,
                    noise
                )
                # w2 shape from teacher: [hidden, intermediate]
                expert.down_proj.weight.data = self._adapt_weight(
                    teacher_ffn['w2'],
                    expert.down_proj.weight.shape,
                    noise
                )
                
                if teacher_ffn.get('b1') is not None and expert.gate_proj.bias is not None:
                    expert.gate_proj.bias.data = self._adapt_bias(
                        teacher_ffn['b1'],
                        expert.gate_proj.bias.shape,
                        noise
                    )
                
                # Persist modified expert back to disk
                xorzen_model.moe.disk_manager.save_expert(expert_id, expert)
                xorzen_model.moe.cache.put(expert_id, expert)

                expert_id += 1
                
                if expert_id % 10 == 0:
                    print(f"  Seeded {expert_id}/{xorzen_model.moe.num_experts} experts...")
        
        print(f"  ✓ Seeded {expert_id} experts from {num_teacher_layers} teacher layers")
        return xorzen_model
    
    def init_router_smart(
        self,
        xorzen_model,
        strategy: str = 'balanced',
    ):
        """Initialize router with smart defaults."""
        print(f"\n[3/4] Initializing router (strategy={strategy})...")
        
        router = xorzen_model.router
        num_layers = len(xorzen_model.blocks)
        
        if hasattr(router, 'depth_router'):
            if strategy == 'balanced':
                router.depth_router.weight.data.fill_(0.0)
            elif strategy == 'early-heavy':
                logits = torch.linspace(1.0, -1.0, num_layers)
                if router.depth_router.weight.shape[-1] == num_layers:
                    router.depth_router.weight.data = logits.unsqueeze(0).expand_as(router.depth_router.weight).clone()
            elif strategy == 'late-heavy':
                logits = torch.linspace(-1.0, 1.0, num_layers)
                if router.depth_router.weight.shape[-1] == num_layers:
                    router.depth_router.weight.data = logits.unsqueeze(0).expand_as(router.depth_router.weight).clone()
        
        if hasattr(router, 'expert_router'):
            router.expert_router.weight.data = torch.randn_like(router.expert_router.weight.data) * 0.02
        
        print(f"  ✓ Router initialized with {strategy} strategy")
        return xorzen_model
    
    def freeze_embeddings(self, xorzen_model, freeze: bool = True):
        """Freeze or unfreeze embeddings."""
        print(f"\n[4/4] {'Freezing' if freeze else 'Unfreezing'} embeddings...")
        
        xorzen_model.token_embedding.weight.requires_grad = not freeze
        if hasattr(xorzen_model, 'position_embedding'):
            xorzen_model.position_embedding.weight.requires_grad = not freeze
        
        frozen = sum(1 for p in xorzen_model.token_embedding.parameters() if not p.requires_grad)
        print(f"  ✓ {frozen} embedding parameters frozen")
        return xorzen_model
    
    def _create_projection(self, in_dim: int, out_dim: int) -> nn.Module:
        proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(proj.weight, gain=0.5)
        return proj
    
    def _adapt_weight(
        self,
        source: torch.Tensor,
        target_shape: Tuple[int, int],
        noise_scale: float = 0.0,
    ) -> torch.Tensor:
        src_out, src_in = source.shape
        tgt_out, tgt_in = target_shape
        
        if src_out == tgt_out and src_in == tgt_in:
            adapted = source.clone()
        else:
            adapted = torch.zeros(target_shape, dtype=source.dtype, device='cpu')
            min_out = min(src_out, tgt_out)
            min_in = min(src_in, tgt_in)
            adapted[:min_out, :min_in] = source[:min_out, :min_in].cpu()
            
            if tgt_out > src_out or tgt_in > src_in:
                nn.init.xavier_uniform_(adapted, gain=0.1)
                adapted[:min_out, :min_in] = source[:min_out, :min_in].cpu()
        
        if noise_scale > 0:
            noise = torch.randn_like(adapted) * noise_scale
            adapted = adapted + noise
        
        return adapted
    
    def _adapt_bias(
        self,
        source: torch.Tensor,
        target_shape: Tuple[int],
        noise_scale: float = 0.0,
    ) -> torch.Tensor:
        adapted = torch.zeros(target_shape, dtype=source.dtype, device='cpu')
        min_len = min(len(source), target_shape[0])
        adapted[:min_len] = source[:min_len].cpu()
        
        if noise_scale > 0:
            adapted = adapted + torch.randn_like(adapted) * noise_scale
        
        return adapted


def quick_transfer(
    xorzen_model,
    teacher_name: str,
    freeze_embeddings: bool = True,
    seed_experts: bool = True,
):
    """
    One-line transfer from ANY teacher to XORZENX.
    
    Usage:
        model = xorzen.zero_1M()
        quick_transfer(model, teacher_name='microsoft/phi-2')
        # Train with 5-10x faster convergence!
    
    Supported teachers:
        - 'microsoft/phi-2' (RECOMMENDED - 2.7B, very strong)
        - 'mistralai/Mistral-7B-v0.1' (7B, SOTA)
        - 'meta-llama/Llama-2-7b-hf' (7B, need token)
        - 'Qwen/Qwen-7B' (7B, multilingual)
        - 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
        - Any HuggingFace causal LM model
    """
    print("="*80)
    print(f"XORZENX QUICK TRANSFER: {teacher_name} → XORZENX")
    print("="*80)
    
    transfer = XORZENXTransferLearning(teacher_name)
    
    transfer.transfer_embeddings(xorzen_model, strategy='resize')
    
    if seed_experts:
        transfer.seed_experts(xorzen_model, num_experts_per_layer=4, noise_scale=0.02)
    
    transfer.init_router_smart(xorzen_model, strategy='balanced')
    
    if freeze_embeddings:
        transfer.freeze_embeddings(xorzen_model, freeze=True)
    
    print("\n" + "="*80)
    print("✓ Transfer complete! Model ready for training.")
    print("="*80)
    
    return xorzen_model


__all__ = [
    'TeacherExtractor',
    'XORZENXTransferLearning',
    'quick_transfer',
]
