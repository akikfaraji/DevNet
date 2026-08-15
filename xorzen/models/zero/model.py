"""
This module provides the core implementation of the `zeroModel`, representing
the flagship architecture of the XORZENX framework. It is designed as a production-grade,
highly efficient, and scalable deep learning model that orchestrates
various innovative components for state-of-the-art performance.

The `zeroModel` exemplifies the framework's commitment to optimizing
computational resources while achieving performance comparable to significantly
larger dense models. Its architecture integrates adaptive sparse activation
through intelligent routing, enabling a balance between model capacity and
computational cost.

Key Architectural Components:
- **Token + Position Embeddings**: Standard input processing layers.
- **Internal CoT Vector Initialization**: Establishes the initial state for
  the Chain-of-Thought reasoning mechanism.
- **Adaptive Router**: A critical component responsible for dynamic decision-making
  regarding depth, width, pathway selection, and expert allocation.
- **HASS Blocks (Hybrid Attention-Shard Switch)**: Modular building blocks that
  process information through multiple parallel pathways (Local, Global, SSM).
- **Disk-Sharded MoE Experts**: A Mixture-of-Experts system where a large number
  of experts are managed efficiently through disk-sharding and top-K routing.
- **Merger Gate**: Responsible for integrating outputs from different pathways
  and components into a unified representation.
- **LM Head**: The final layer that projects the model's internal representation
  to a vocabulary-sized output for next-token prediction.

The primary innovation lies in its adaptive sparse activation, allowing for
dynamic resource allocation per token, leading to substantial gains in
inference speed and reduction in training costs compared to traditional dense models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
import warnings
from typing import Optional, Tuple, Dict, List, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import time

from xorzen.config import ModelConfig
from xorzen.utils.logger import get_logger
from xorzen.utils.math_utils import (
    TensorStability, 
    InformationTheory,
    count_parameters,
    get_device_memory_stats,
    RMSNorm
)
from xorzen.model.components.cot_vector import InternalLatentCoT
from xorzen.model.components.routing import AdaptiveRouter, RoutingDecision, RoutingRegularizer
from xorzen.model.components.hass_block import HASSBlock
from xorzen.model.components.merger import xorzenMergerGate

logger = get_logger()



# ==================== MAIN MODEL ====================

from xorzen.model.base import BaseModel, ModelOutput, GenerationConfig


class zeroModel(BaseModel):
    """
    Implements the full XORZENX-zero model architecture, integrating all core components
    such as embeddings, adaptive routing, HASS blocks, Mixture-of-Experts (MoE) fabric,
    Chain-of-Thought (CoT) reasoning, and a final language model head.
    
    This class orchestrates the dynamic interaction between these components to
    achieve adaptive sparse activation, leading to high computational efficiency
    and state-of-the-art performance. The model supports various sizes and
    configurations as defined by the `ModelConfig`.
    
    Args:
        config (`ModelConfig`): An instance of `ModelConfig` containing all
                                hyperparameters and architectural specifications
                                for building the `zeroModel`.
        test_mode (`bool`, optional): If `True`, the model may operate in a
                                      simplified mode suitable for testing or
                                      debugging, potentially altering behaviors
                                      like expert loading or caching. Defaults to `False`.
    """
    
    def __init__(self, config: ModelConfig, test_mode: bool = False):
        """
        Initializes the `zeroModel` with the given configuration.
        This sets up all architectural components from embeddings to the LM head.
        """
        super().__init__(config)
        
        if not isinstance(config, ModelConfig):
            raise TypeError(f"config must be zeroConfig, got {type(config)}")
        
        self.config = config
        self.test_mode = test_mode # Store test_mode
        
        # Validate configuration
        self._validate_config()
        
        logger.info("core", f"Initializing {config.model_name} with {config.num_layers} layers...")
        
        # ========== EMBEDDINGS ==========
        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None)
        )
        
        # Position embeddings (learned for now, can switch to RoPE)
        self.position_embedding = nn.Embedding(
            num_embeddings=config.context_length,
            embedding_dim=config.hidden_size
        )
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(config.dropout)
        
        # ========== INTERNAL REASONING ==========
        # Internal CoT vector module (6 components).
        # During pre-training the CoT is fully frozen (requires_grad=False).
        # This prevents the 4 CoT output params from showing up as "missing gradient"
        # and avoids gradient noise from an untrained reasoning module.
        # Enable CoT for fine-tuning by calling model.enable_cot().
        self.cot = InternalLatentCoT(config)
        self._freeze_cot()  # Freeze for pre-training
        logger.info("core", f"Internal CoT: {config.cot_components} components x {config.cot_dim} dim = {config.cot_dim * config.cot_components} total (frozen for pre-training)")
        
        # ========== ADAPTIVE ROUTING ==========
        # Router makes all decisions: depth, width, path, expert
        self.router = AdaptiveRouter(config)
        self.routing_regularizer = RoutingRegularizer(config)
        logger.info("core", "Adaptive Router initialized (depth/width/path/expert routing)")
        
        # ========== HASS BLOCKS ==========
        # 16 transformer blocks with 3 parallel pathways each
        self.blocks = nn.ModuleList([
            HASSBlock(config, layer_idx=i) 
            for i in range(config.num_layers)
        ])
        logger.info("core", f"Created {config.num_layers} HASS blocks (Local/Global/SSM pathways)")
        
        # ========== MOE EXPERT FABRIC ==========
        # Import here to avoid circular dependency
        from xorzen.model.zmoe import ShardedExpertFabric
        import shutil as _shutil, json as _json
        # Auto-clear stale expert shards if config changed (hidden_size or multiplier mismatch)
        _shard_dir = Path(config.expert_shard_dir)
        _meta_path = _shard_dir / ".meta.json"
        if _meta_path.exists():
            try:
                _meta = _json.loads(_meta_path.read_text())
                _changed = (
                    _meta.get("hidden_size") != config.hidden_size or
                    _meta.get("expert_hidden_multiplier") != config.expert_hidden_multiplier or
                    _meta.get("expert_count") != config.expert_count
                )
                if _changed:
                    logger.info("core", f"Stale expert shards (hidden={_meta.get('hidden_size')}->{config.hidden_size}). Auto-clearing.")
                    _shutil.rmtree(str(_shard_dir), ignore_errors=True)
            except Exception:
                pass
        logger.info("core", f"zeroModel passing test_mode={self.test_mode} to ShardedExpertFabric")
        self.moe = ShardedExpertFabric(config, test_mode=self.test_mode)
        # Write meta for future stale-shard detection
        _shard_dir.mkdir(parents=True, exist_ok=True)
        try:
            _meta_path.write_text(_json.dumps({"hidden_size": config.hidden_size, "expert_hidden_multiplier": config.expert_hidden_multiplier, "expert_count": config.expert_count}))
        except Exception:
            pass
        logger.info("core", f"MoE: {config.expert_count} experts (Top-{config.top_k_experts} routing, disk-sharded)")
        
        # ========== MERGER GATE ==========
        # Fuses HASS output + MoE output + CoT vector
        self.merger = xorzenMergerGate(config)
        logger.info("core", "Merger gate initialized (fuses HASS + MoE + CoT)")
        
        # ========== OUTPUT ==========
        # Final layer norm before LM head
        self.final_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Language model head (can be tied with token_embedding)
        self.lm_head = nn.Linear(
            config.hidden_size, 
            config.vocab_size, 
            bias=False
        )
        
        # Optionally tie weights
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
            logger.info("core", "Tied word embeddings (embedding ↔ lm_head)")
        
        # ========== INITIALIZATION ==========
        self.apply(self._init_weights)
        
        # Special initialization for specific layers
        self._init_special_layers()
        
        # ========== GRADIENT CHECKPOINTING ==========
        self.gradient_checkpointing = config.gradient_checkpointing
        if self.gradient_checkpointing:
            logger.info("core", "Gradient checkpointing enabled (saves memory)")
        
        # ========== PERFORMANCE TRACKING ==========
        self._step_count = 0
        self._total_tokens_processed = 0
        self._active_params_history = []
        
        # Print model summary
        total_params = self.count_parameters()
        trainable_params = self.count_parameters(only_trainable=True)
        logger.info("core", f"Model initialized: {total_params:,} total params, {trainable_params:,} trainable")
        # Active-parameter estimate: use the SAME accounting as
        # `config.estimate_active_parameters()` and the runtime estimator
        # `_estimate_active_params()`. The previous ad-hoc formula only
        # counted `top_k_experts * hidden_size * expert_hidden_multiplier`,
        # which omits the *2 for input+output projections of each expert and
        # ignores every always-on component (embeddings, HASS blocks, router,
        # merger, CoT, LM head). On zero_1M that gave ~0.03% instead of the
        # real ~7.8%, which printed as "~0.0%". Use the proper estimator so
        # the init-time log matches the runtime active-params metric.
        _active_est = int(self.config.estimate_active_parameters())
        _active_pct = 100.0 * _active_est / max(1, trainable_params)
        logger.info(
            "core",
            f"Top-{self.config.top_k_experts}/{self.config.expert_count} experts active per token "
            f"(~{_active_pct:.1f}% of {trainable_params:,} trainable params | "
            f"active~{_active_est:,} / total~{total_params:,})",
        )
    
    def _validate_config(self):
        """
        Validates the provided `ModelConfig` instance to ensure all essential
        parameters are correctly set and adhere to architectural constraints
        before the model's layers are constructed. This prevents runtime errors
        due to malformed configurations.
        
        Raises:
            AssertionError: If any critical configuration parameter is invalid
                            or inconsistent.
        """
        config = self.config
        
        # Check basic requirements
        assert config.vocab_size > 0, "vocab_size must be positive"
        assert config.hidden_size > 0, "n_embd must be positive"
        assert config.num_layers > 0, "n_layer must be positive"
        assert config.num_attention_heads > 0, "n_head must be positive"
        assert config.hidden_size % config.num_attention_heads == 0, "n_embd must be divisible by n_head"
        
        # Check adaptive routing
        assert len(config.width_choices) > 0, "width_choices cannot be empty"
        assert max(config.width_choices) == config.hidden_size, "max width choice must equal n_embd"
        assert config.max_depth <= config.num_layers, "max_depth must be <= n_layer"
        
        # Check MoE
        assert config.expert_count > 0, "num_experts must be positive"
        assert config.top_k_experts <= config.expert_count, "top_k must be <= num_experts"
        
        # Check CoT
        assert config.cot_components == 6, "cot_components must be 6 (intention, decomposition, confidence, contradiction, direction, summary)"
        assert config.cot_dim > 0, "cot_dim must be positive"
        
        logger.debug("core", "Configuration validation passed")
    
    def _init_weights(self, module):
        """
        Initializes the weights of the model's sub-modules using recommended
        best practices. Linear layers and embeddings are typically initialized
        with a normal distribution, while biases are set to zeros. Layer normalization
        weights are generally initialized to ones.
        
        Args:
            module: The `torch.nn.Module` to initialize its weights.
        """        
        if isinstance(module, nn.Linear):
            # Use scaled initialization for stability
            std = 0.02
            if hasattr(module, 'scale_init'):
                std *= module.scale_init
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if hasattr(module, 'weight'):
                torch.nn.init.ones_(module.weight)
    
    def _init_special_layers(self):
        """
        Applies specific initialization routines to particular layers
        or components within the model that require non-standard setup.
        For instance, the language model head might have its weights
        scaled down to improve initial training stability if not tied
        to the token embeddings.
        """        # Scale down LM head for better initialization
        if not self.config.tie_word_embeddings:
            self.lm_head.weight.data.mul_(1.0 / math.sqrt(self.config.hidden_size))
        
        logger.debug("core", "Special layer initialization complete")
    
    # ==================== COT LIFECYCLE ====================

    def _freeze_cot(self):
        """Freeze all CoT parameters for pre-training."""
        for param in self.cot.parameters():
            param.requires_grad = False

    def enable_cot(self):
        """
        Enable CoT for fine-tuning phase.
        Call this after pre-training completes to unfreeze CoT parameters
        and enable CoT injection into the hidden states.
        """
        for param in self.cot.parameters():
            param.requires_grad = True
        logger.info("core", "CoT enabled for fine-tuning")

    # ==================== FORWARD PASS ====================
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        output_routing_info: bool = False,
        output_cot_vector: bool = True,
        return_dict: bool = True,
        use_cache: bool = False,
    ) -> Union[ModelOutput, Tuple]:
        """
        Defines the forward pass logic for the `zeroModel`.
        Input tokens are processed sequentially through a series of stages:
        embeddings, adaptive routing, HASS blocks, Mixture-of-Experts (MoE),
        merger gate, and finally the language model head to produce logits.
        Auxiliary losses for routing and CoT consistency are also computed.
        
        The process involves:
        1.  **Embeddings**: Token and positional embeddings are applied.
        2.  **Internal CoT Initialization/Update**: CoT vector is initialized and
            recurrently updated across layers.
        3.  **Adaptive Routing**: The router dynamically decides which components
            (depth, width, pathway, experts) are activated for each token.
        4.  **HASS Blocks**: Processes hidden states through hybrid attention
            and State Space Model pathways.
        5.  **MoE Experts**: Processes tokens through dynamically selected
            Mixture-of-Experts layers.
        6.  **Merger Gate**: Combines outputs from HASS blocks, MoE, and CoT.
        7.  **LM Head**: Projects the final hidden states to vocabulary logits.
        
        Args:
            input_ids (`torch.LongTensor`): Tensor of input token IDs, shape `(batch_size, sequence_length)`.
            attention_mask (`torch.Tensor`, *optional*): Mask to avoid performing attention
                                                           on padding tokens, shape `(batch_size, sequence_length)`.
            position_ids (`torch.LongTensor`, *optional*): Specific positional indices
                                                            for each token, shape `(batch_size, sequence_length)`.
                                                            If `None`, sequential indices are generated.
            labels (`torch.LongTensor`, *optional*): Labels for computing the language modeling loss,
                                                     shape `(batch_size, sequence_length)`.
            output_hidden_states (`bool`, *optional*): If `True`, intermediate hidden states
                                                        from each HASS block are returned. Defaults to `False`.
            output_routing_info (`bool`, *optional*): If `True`, the routing decisions made by
                                                       the Adaptive Router are returned. Defaults to `False`.
            output_cot_vector (`bool`, *optional*): If `True`, the final Chain-of-Thought
                                                     vector is returned. Defaults to `True`.
            return_dict (`bool`, *optional*): If `True`, a `ModelOutput` object is returned.
                                               Otherwise, a tuple is returned. Defaults to `True`.
            use_cache (`bool`, *optional*): If `True`, enables KV caching for faster sequential
                                             decoding during generation. Defaults to `False`.
        
        Returns:
            `Union[ModelOutput, Tuple]`: A `ModelOutput` object or a tuple, depending on `return_dict`.
                                         The output contains:
                                         - `logits` (`torch.Tensor`): Raw prediction scores for each vocabulary token.
                                         - `loss` (`torch.Tensor`, *optional*): Language modeling loss.
                                         - `cot_vector` (`torch.Tensor`, *optional*): Final CoT vector.
                                         - `routing_info` (`RoutingDecision`, *optional*): Details of routing decisions.
                                         - `layer_outputs` (`List[torch.Tensor]`, *optional*): Intermediate hidden states.
        
        Raises:
            ValueError: If `input_ids` sequence length exceeds the model's configured `context_length`
                        or if `labels` shape does not match `input_ids` shape.
        """

        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        # NOTE: global RNG seeding removed — it broke sampling across batches.

        # Validate inputs
        if seq_length > self.config.context_length:
            raise ValueError(
                f"Sequence length {seq_length} exceeds maximum "
                f"{self.config.context_length}"
            )
        
        # ========== STEP 1: EMBEDDINGS ==========
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)  # [B, T, H]
        
        # Position embeddings
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, 
                dtype=torch.long, 
                device=device
            ).unsqueeze(0).expand(batch_size, -1)
        
        position_embeds = self.position_embedding(position_ids)  # [B, T, H]
        
        # Combine embeddings
        hidden_states = hidden_states + position_embeds
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length),
                dtype=torch.bool,
                device=device
            )
        
        # ========== STEP 2: INITIALIZE COT ==========
        # CoT is DISABLED during pre-training to ensure clean gradient flow.
        # It will be enabled as a fine-tuning phase after the base model converges.
        # We still pass a zeroed cot_vector_seq so downstream components (merger,
        # router) keep their expected input shapes without receiving any signal.
        cot_total_dim = self.config.cot_dim * self.config.cot_components
        cot_vector_seq = torch.zeros(
            batch_size, seq_length, cot_total_dim,
            device=device, dtype=hidden_states.dtype
        )  # [B, T, cot_dim] — zeros, no gradient
        
        # ========== STEP 3: ADAPTIVE ROUTING ==========
        # Router makes ALL decisions for the forward pass
        routing_decision = self.router(
            x=hidden_states,
            cot_features=cot_vector_seq,
            training=self.training,
        )
        
        # Extract routing decisions
        depth_mask = routing_decision.depth_mask  # [B, T, n_layer]
        width_multiplier = routing_decision.width_multiplier  # [B, T, 1]
        path_probs = routing_decision.path_probs  # [B, T, num_paths]
        expert_indices = routing_decision.expert_indices  # [B, T, top_k]
        expert_weights = routing_decision.expert_weights  # [B, T, top_k]
        
        # ========== STEP 4: HASS BLOCKS ==========
        # Process through transformer blocks with adaptive depth
        layer_outputs = [] if output_hidden_states else None
        
        for layer_idx, block in enumerate(self.blocks):
            # depth_mask[:, :, layer_idx] is the soft STE mask [B, T] with values in (0,1)
            # during training (gradient flows through it) and hard 0/1 at inference.
            layer_mask = depth_mask[:, :, layer_idx]  # [B, T]  — soft during training

            # GENUINE SPARSE DEPTH (inference): if some tokens have mask=0,
            # gather only the active tokens, run the block on them, and scatter
            # back. Tokens with mask=0 are NOT processed by the block. This is
            # the "gather active -> block -> scatter" pattern required by the
            # conditional-compute principle; the previous "block(x) * mask + x
            # * (1-mask)" computed everything then masked, which is forbidden.
            #
            # At training, we use the masked blend because the STE needs the
            # block output in the autograd graph. The block is still skipped
            # entirely if NO token is active.
            if not self.training:
                # Hard 0/1 mask at inference
                active = (layer_mask > 0.5)
                n_active = int(active.sum().item())
                if n_active == 0:
                    # No tokens active — skip block entirely
                    if output_hidden_states:
                        layer_outputs.append(hidden_states)
                    continue
                if n_active < batch_size * seq_length:
                    # PARTIAL skip: use the block's forward_with_depth method,
                    # which gathers active tokens, slices the routing decision,
                    # runs the block on the active subset, and scatters back.
                    hidden_states = block.forward_with_depth(
                        x=hidden_states,
                        depth_mask=active.float(),
                        routing_decision=routing_decision,
                        attention_mask=attention_mask,
                    )
                    if output_hidden_states:
                        layer_outputs.append(hidden_states)
                    continue
                # else: all tokens active — fall through to normal path

            # Training path (or all-active inference): compute block on full input
            # Apply gradient checkpointing if enabled
            if self.gradient_checkpointing and self.training:
                def create_forward_func(block_module, routing_decision_obj, attention_mask_obj):
                    def forward_func(h_states):
                        return block_module(
                            x=h_states,
                            routing_decision=routing_decision_obj,
                            attention_mask=attention_mask_obj
                        )
                    return forward_func

                block_out = checkpoint(
                    create_forward_func(block, routing_decision, attention_mask),
                    hidden_states,
                    use_reentrant=False
                )
            else:
                block_out = block(
                    x=hidden_states,
                    routing_decision=routing_decision,
                    attention_mask=attention_mask
                )

            # Training: use STE mask blend (gradient flows through mask).
            # block_out is the FULL block output (already has its own internal residual),
            # so we select between block_out (active) and hidden_states (skipped) per token.
            layer_mask_3d = layer_mask.unsqueeze(-1)          # [B, T, 1]
            hidden_states = block_out * layer_mask_3d + hidden_states * (1.0 - layer_mask_3d)

            # CoT update skipped during pre-training (cot_vector_seq stays zero).
            if output_hidden_states:
                layer_outputs.append(hidden_states)
        
        # ========== STEP 5: MOE EXPERTS ==========
        # Flatten for expert processing
        batch_seq = batch_size * seq_length
        hidden_flat = hidden_states.reshape(batch_seq, -1)  # [B*T, H]
        expert_indices_flat = expert_indices.reshape(batch_seq, -1)  # [B*T, top_k]
        expert_weights_flat = expert_weights.reshape(batch_seq, -1)  # [B*T, top_k]
        
        # Process through MoE fabric
        moe_output_flat, expert_stats = self.moe(
            hidden_flat,
            expert_indices_flat,
            expert_weights_flat,
            attention_mask=attention_mask.reshape(batch_seq) if attention_mask is not None else None
        )
        
        # Reshape back
        moe_output = moe_output_flat.reshape(batch_size, seq_length, -1)  # [B, T, H]
        
        # ========== STEP 6: MERGER GATE ==========
        # Fuse HASS output + MoE output + CoT reasoning
        merged_output = self.merger(
            hass_output=hidden_states,
            moe_output=moe_output,
            cot_vector=cot_vector_seq,
            attention_mask=attention_mask
        )
        
        # ========== STEP 7: FINAL PROCESSING ==========
        # Final layer norm
        hidden_states = self.final_norm(merged_output)
        
        # LM head -> logits
        logits = self.lm_head(hidden_states)  # [B, T, vocab_size]
        
        # ========== STEP 8: LOSS COMPUTATION ==========
        loss = None
        if labels is not None:
            # Loud failure: label shape must exactly match input shape
            if labels.shape != input_ids.shape:
                raise ValueError(
                    f"labels shape {tuple(labels.shape)} must match input_ids shape {tuple(input_ids.shape)}"
                )
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=getattr(self.config, "pad_token_id", -100)
            )
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # ========== STEP 9: AUXILIARY LOSSES ==========
        # Routing regularization (uncertainty keeps uncertainty_estimator in graph)
        routing_loss = self.routing_regularizer(routing_decision)
        
        # Add router auxiliary losses (z_loss + lb_loss + path_div_loss from router internals)
        if hasattr(routing_decision, 'auxiliary') and routing_decision.auxiliary:
            for aux_key, aux_loss_val in routing_decision.auxiliary.items():
                if isinstance(aux_loss_val, torch.Tensor) and aux_loss_val.requires_grad:
                    routing_loss = routing_loss + aux_loss_val
        
        if routing_loss.numel() > 1:
            routing_loss = routing_loss.mean()

        # Expert load balancing
        load_balance_loss = self._compute_load_balance_loss(
            expert_indices_flat,
            expert_weights_flat
        )
        
        # CoT consistency loss zeroed during pre-training
        cot_consistency_loss = torch.tensor(0.0, device=device)
        
        # Accumulate auxiliary losses — but ONLY if LM loss exists.
        # The aux weights in config are now 0.0001 so they won't drown LM signal.
        # We track lm_loss separately on ModelOutput for debugging.
        lm_loss = loss  # pure cross-entropy before aux
        if loss is not None:
            loss = loss + routing_loss + load_balance_loss
        else:
            loss = routing_loss + load_balance_loss
        
        if loss is not None and loss.numel() > 1:
            loss = loss.mean()
        
        # ========== STEP 10: COMPUTE METRICS ==========
        active_params = self._estimate_active_params(routing_decision)
        compute_cost = self._estimate_compute_cost(routing_decision, batch_size, seq_length)
        
        # Update tracking
        self._step_count += 1
        self._total_tokens_processed += batch_size * seq_length
        self._active_params_history.append(active_params)
        
        # ========== RETURN OUTPUTS ==========
        if not return_dict:
            outputs = (logits, loss, cot_vector_seq if output_cot_vector else None)
            if output_hidden_states:
                outputs += (layer_outputs,)
            if output_routing_info:
                outputs += (routing_decision,)
            return outputs
        
        return ModelOutput(
            logits=logits,
            loss=loss,
            lm_loss=lm_loss,
            cot_vector=cot_vector_seq if output_cot_vector else None,
            routing_info=routing_decision if output_routing_info else None,
            layer_outputs=layer_outputs,
            expert_stats=expert_stats,
            routing_loss=routing_loss,
            load_balance_loss=load_balance_loss,
            cot_consistency_loss=cot_consistency_loss,
            active_params=active_params,
            compute_cost=compute_cost
        )
    
    # ==================== AUXILIARY LOSS FUNCTIONS ====================
    
    def _compute_load_balance_loss(
        self,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes a load balancing loss for the Mixture-of-Experts (MoE) component.
        This loss term encourages a more uniform distribution of workload across
        all experts, actively preventing certain experts from becoming over-utilized
        while others remain under-utilized or 'dead'.
        
        Args:
            expert_indices (`torch.Tensor`): A tensor of shape `(num_tokens, top_k)`
                                             containing the indices of the top-k experts
                                             selected for each token.
            expert_weights (`torch.Tensor`): A tensor of shape `(num_tokens, top_k)`
                                             containing the weights corresponding to
                                             the selected top-k experts for each token.
        
        Returns:
            `torch.Tensor`: A scalar tensor representing the computed load balancing loss.
        """
        num_experts = self.config.expert_count
        num_tokens = expert_indices.size(0)
        
        # Count how many tokens route to each expert
        expert_counts = torch.zeros(num_experts, device=expert_indices.device, dtype=expert_weights.dtype)
        for k in range(self.config.top_k_experts):
            expert_counts.scatter_add_(
                0,
                expert_indices[:, k],
                expert_weights[:, k]
            )
        
        # Normalize to get probabilities
        expert_probs = expert_counts / (num_tokens * self.config.top_k_experts + 1e-8)
        
        # Target: uniform distribution
        target_prob = 1.0 / num_experts
        
        # L2 loss
        loss = torch.sum((expert_probs - target_prob) ** 2)
        
        return self.config.load_balancing_weight * loss
    
    def _compute_cot_consistency_loss(self, cot_vector: torch.Tensor) -> torch.Tensor:
        """
        Calculates a consistency loss for the Chain-of-Thought (CoT) vector.
        This loss penalizes large, erratic changes in the CoT vector across
        sequential tokens, thereby encouraging a smoother and more stable
        evolution of the model's internal reasoning trace.
        
        Args:
            cot_vector (`torch.Tensor`): A tensor of shape `(batch_size, sequence_length, cot_dim)`
                                         representing the CoT vector for each token in the sequence.
        
        Returns:
            `torch.Tensor`: A scalar tensor representing the computed CoT consistency loss.
        """
        if cot_vector.size(1) < 2:
            return torch.tensor(0.0, device=cot_vector.device)
        
        # Compute difference between consecutive steps
        cot_diff = cot_vector[:, 1:, :] - cot_vector[:, :-1, :]  # [B, T-1, cot_dim]
        
        # Penalize large jumps (L2 norm)
        consistency_loss = torch.mean(cot_diff ** 2)
        
        return self.config.cot_consistency_weight * consistency_loss
    
    def _estimate_active_params(self, routing_decision: RoutingDecision) -> int:
        """
        Estimates the number of active parameters engaged during the current
        forward pass, based on the dynamic routing decisions made for the batch.
        This provides an approximate measure of the computational sparsity and
        efficiency for a given input.
        
        Args:
            routing_decision (`RoutingDecision`): An object containing the
                                                  dynamic routing decisions
                                                  (e.g., active layers, experts)
                                                  for the current batch.
        
        Returns:
            An integer representing the estimated number of active parameters.
        """
        active_params = 0
        
        # Embeddings (always active)
        active_params += self.config.vocab_size * self.config.hidden_size
        
        # Average depth (number of layers executed)
        avg_depth = routing_decision.depth_mask.float().sum(dim=-1).mean().item()
        
        # Parameters per HASS block (approximate)
        params_per_block = count_parameters(self.blocks[0]) if self.blocks else 0
        active_params += int(avg_depth * params_per_block)
        
        # Active experts (top-k)
        intermediate_dim = int(self.config.hidden_size * self.config.expert_hidden_multiplier)
        active_params += self.config.top_k_experts * (
            self.config.hidden_size * intermediate_dim * 2
        )
        
        # CoT module (always active)
        active_params += count_parameters(self.cot)
        
        # Router (always active)
        active_params += count_parameters(self.router)
        
        # Merger (always active)
        active_params += count_parameters(self.merger)
        
        # LM head (always active)
        active_params += self.config.hidden_size * self.config.vocab_size
        
        return active_params
    
    def _estimate_compute_cost(
        self, 
        routing_decision: RoutingDecision,
        batch_size: int,
        seq_length: int
    ) -> float:
        """
        Provides a rough estimation of the computational cost (FLOPs) for a single
        forward pass, considering the dynamic routing decisions, batch size, and
        sequence length. This metric offers an indicative measure of the
        computational demand for the current processing step.
        
        Args:
            routing_decision (`RoutingDecision`): An object encapsulating the
                                                  dynamic routing choices for
                                                  the current batch.
            batch_size (`int`): The number of sequences processed in parallel.
            seq_length (`int`): The length of each sequence in the batch.
        
        Returns:
            A float representing the estimated FLOPs (in GFLOPs) for the forward pass.
        """
        # This is a rough estimate
        flops = 0.0
        
        # Embeddings: B * T * H
        flops += batch_size * seq_length * self.config.hidden_size
        
        # HASS blocks: depends on depth and width
        avg_depth = routing_decision.depth_mask.float().mean().item() * self.config.num_layers
        avg_width = routing_decision.width_multiplier.mean().item() * self.config.hidden_size
        
        # Attention: B * T^2 * H
        flops += avg_depth * batch_size * seq_length * seq_length * avg_width
        
        # MLP: B * T * H * (4H)
        flops += avg_depth * batch_size * seq_length * avg_width * (4 * avg_width)
        
        # MoE: B * T * top_k * expert_dim
        flops += batch_size * seq_length * self.config.top_k_experts * int(self.config.hidden_size * self.config.expert_hidden_multiplier)
        
        # LM head: B * T * H * vocab
        flops += batch_size * seq_length * self.config.hidden_size * self.config.vocab_size
        
        # Convert to GFLOPs
        return flops / 1e9
    
    # ==================== GENERATION ====================
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> torch.LongTensor:
        """
        Generates a sequence of tokens autoregressively based on the provided
        `input_ids` and generation strategy. This method supports greedy decoding,
        sampling-based generation (with top-k and top-p filtering), and beam search.
        
        Args:
            input_ids (`torch.LongTensor`): A tensor of starting token IDs,
                                             shape `(batch_size, initial_sequence_length)`.
            generation_config (`GenerationConfig`, *optional*): An instance of `GenerationConfig`
                                                                specifying the parameters for
                                                                the generation process (e.g.,
                                                                `max_new_tokens`, `temperature`,
                                                                `do_sample`, `num_beams`).
                                                                If `None`, a default configuration is used.
            **kwargs: Arbitrary keyword arguments to override parameters within
                      the `generation_config`. These overrides take precedence
                      over values set in `generation_config`.
        
        Returns:
            `torch.LongTensor`: A tensor of generated token IDs,
                                 shape `(batch_size, initial_sequence_length + max_new_tokens)`.
        """
        # Setup generation config
        if generation_config is None:
            generation_config = GenerationConfig()
        
        # Override with kwargs
        for key, value in kwargs.items():
            if hasattr(generation_config, key):
                setattr(generation_config, key, value)
        
        # Determine generation method
        if generation_config.num_beams > 1:
            return self._beam_search_generate(input_ids, generation_config)
        elif generation_config.do_sample:
            return self._sample_generate(input_ids, generation_config)
        else:
            return self._greedy_generate(input_ids, generation_config)
    
    def _greedy_generate(
        self,
        input_ids: torch.LongTensor,
        config: GenerationConfig
    ) -> torch.LongTensor:
        """
        Performs greedy decoding to generate a sequence of tokens.
        At each step, the token with the highest probability is deterministically
        selected as the next token. This method also applies temperature scaling
        and an optional repetition penalty.
        
        Args:
            input_ids (`torch.LongTensor`): A tensor of starting token IDs,
                                             shape `(batch_size, current_sequence_length)`.
            config (`GenerationConfig`): An instance of `GenerationConfig`
                                         specifying parameters like `max_new_tokens`,
                                         `temperature`, `repetition_penalty`, and `eos_token_id`.
                                         
        Returns:
            `torch.LongTensor`: A tensor of generated token IDs,
                                 shape `(batch_size, initial_sequence_length + max_new_tokens)`.
        """        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Track which sequences are finished
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        
        for _ in range(config.max_new_tokens):
            # Forward pass
            outputs = self(input_ids, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]  # [B, vocab]
            
            # Apply temperature
            if config.temperature != 1.0:
                next_token_logits = next_token_logits / config.temperature
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits,
                    input_ids,
                    config.repetition_penalty
                )
            
            # Greedy selection
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Update which sequences are finished
            if config.eos_token_id is not None:
                pad_token_id = config.pad_token_id if config.pad_token_id is not None else 0
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.ne(config.eos_token_id).long()
                )
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Stop if all sequences are finished
            if unfinished_sequences.max() == 0:
                break
        
        return input_ids
    
    def _sample_generate(
        self,
        input_ids: torch.LongTensor,
        config: GenerationConfig
    ) -> torch.LongTensor:
        """
        Generates a sequence of tokens using sampling-based decoding, which introduces
        stochasticity to the generation process. This method incorporates temperature
        scaling, an optional repetition penalty, and supports top-k and top-p (nucleus)
        filtering to control the diversity and quality of generated text.
        
        Args:
            input_ids (`torch.LongTensor`): A tensor of starting token IDs,
                                             shape `(batch_size, current_sequence_length)`.

            config (`GenerationConfig`): An instance of `GenerationConfig`
                                         specifying parameters like `max_new_tokens`,
                                         `temperature`, `repetition_penalty`, `top_k`, `top_p`,
                                         and `eos_token_id`.
                                         
        Returns:
            `torch.LongTensor`: A tensor of generated token IDs,
                                 shape `(batch_size, initial_sequence_length + max_new_tokens)`.
        """        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        
        for _ in range(config.max_new_tokens):
            # Forward pass
            outputs = self(input_ids, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]  # [B, vocab]
            
            # Apply temperature
            next_token_logits = next_token_logits / config.temperature
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits,
                    input_ids,
                    config.repetition_penalty
                )
            
            # Apply top-k filtering
            if config.top_k is not None and config.top_k > 0:
                next_token_logits = self._top_k_filtering(
                    next_token_logits,
                    config.top_k
                )
            
            # Apply top-p (nucleus) filtering
            if config.top_p is not None and config.top_p < 1.0:
                next_token_logits = self._top_p_filtering(
                    next_token_logits,
                    config.top_p
                )
            
            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Update sequences
            if config.eos_token_id is not None:
                pad_token_id = config.pad_token_id if config.pad_token_id is not None else 0
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.ne(config.eos_token_id).long()
                )
            
            # Append
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Early stopping
            if unfinished_sequences.max() == 0:
                break
        
        return input_ids
    
    def _beam_search_generate(
        self,
        input_ids: torch.LongTensor,
        config: GenerationConfig
    ) -> torch.LongTensor:
        """
        Generates a sequence of tokens using beam search decoding.
        This method explores multiple promising sequences simultaneously
        to find a high-probability output sequence, often leading to
        higher quality generation compared to greedy decoding.
        
        Args:
            input_ids (`torch.LongTensor`): A tensor of starting token IDs,
                                             shape `(batch_size, current_sequence_length)`.
            config (`GenerationConfig`): An instance of `GenerationConfig`
                                         specifying parameters like `max_new_tokens`,
                                         `num_beams`, `temperature`, and `eos_token_id`.
                                         
        Returns:
            `torch.LongTensor`: A tensor of generated token IDs,
                                 shape `(batch_size, initial_sequence_length + max_new_tokens)`.
        """        # Simplified beam search implementation
        batch_size = input_ids.size(0)
        device = input_ids.device
        num_beams = config.num_beams
        
        # Expand input for beam search
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, -1)
        input_ids = input_ids.reshape(batch_size * num_beams, -1)
        
        # Beam scores
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_scores[:, 1:] = -1e9  # Only first beam is active initially
        beam_scores = beam_scores.view(-1)
        
        unfinished_sequences = torch.ones(batch_size * num_beams, dtype=torch.long, device=device)
        
        for _ in range(config.max_new_tokens):
            outputs = self(input_ids, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]  # [B*num_beams, vocab]
            
            # Apply temperature
            next_token_logits = next_token_logits / config.temperature
            
            # Get log probabilities
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # [B*num_beams, vocab]
            
            # Add beam scores
            next_token_scores = next_token_scores + beam_scores[:, None]  # [B*num_beams, vocab]
            
            # Reshape for beam selection
            vocab_size = next_token_scores.size(-1)
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            
            # Select top 2*num_beams
            next_scores, next_tokens = torch.topk(
                next_token_scores, 
                2 * num_beams, 
                dim=1, 
                largest=True, 
                sorted=True
            )
            
            # Get beam indices and tokens
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode='floor')
            next_tokens = next_tokens % vocab_size
            
            # Create new beams
            beam_outputs = []
            beam_scores_new = []
            
            for batch_idx in range(batch_size):
                beams = []
                for beam_idx in range(num_beams):
                    # Get top beams for this batch
                    for idx in range(2 * num_beams):
                        score = next_scores[batch_idx, idx]
                        token = next_tokens[batch_idx, idx]
                        beam_id = next_indices[batch_idx, idx]
                        
                        # Get original beam
                        orig_idx = batch_idx * num_beams + beam_id
                        new_seq = torch.cat([
                            input_ids[orig_idx],
                            token.unsqueeze(0)
                        ], dim=0)
                        
                        beams.append((score, new_seq))
                        
                        if len(beams) >= num_beams:
                            break
                    
                    if len(beams) >= num_beams:
                        break
                
                # Sort and select top beams
                beams = sorted(beams, key=lambda x: x[0], reverse=True)[:num_beams]
                
                for score, seq in beams:
                    beam_outputs.append(seq)
                    beam_scores_new.append(score)
            
            # Update
            input_ids = torch.stack(beam_outputs, dim=0)
            beam_scores = torch.tensor(beam_scores_new, device=device)
            
            # Check for EOS
            if config.eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    input_ids[:, -1].ne(config.eos_token_id).long()
                )
            
            if unfinished_sequences.max() == 0:
                break
        
        # Return best beam for each batch
        input_ids = input_ids.view(batch_size, num_beams, -1)
        return input_ids[:, 0, :]  # Return first beam
    
    # ==================== HELPER FUNCTIONS ====================
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """
        Applies a penalty to the logits of previously generated tokens to
        discourage the model from repeating itself during text generation.
        Tokens that have already appeared in the `input_ids` will have their
        probabilities adjusted based on the `penalty` factor.
        
        Args:
            logits (`torch.Tensor`): The raw prediction scores for the next token,
                                      shape `(batch_size, vocab_size)`.
            input_ids (`torch.Tensor`): The sequence of tokens generated so far,
                                        shape `(batch_size, current_sequence_length)`.
            penalty (`float`): The repetition penalty factor. A value greater
                               than 1.0 will penalize repetitions, while a value
                               less than 1.0 will encourage them.
        
        Returns:
            `torch.Tensor`: The adjusted logits after applying the repetition penalty,
                             shape `(batch_size, vocab_size)`.
        """        
        batch_size, vocab_size = logits.shape
        
        for i in range(batch_size):
            for token_id in set(input_ids[i].tolist()):
                # Lower probability of repeated tokens
                if logits[i, token_id] < 0:
                    logits[i, token_id] *= penalty
                else:
                    logits[i, token_id] /= penalty
        
        return logits
    
    def _top_k_filtering(
        self,
        logits: torch.Tensor,
        top_k: int
    ) -> torch.Tensor:
        """
        Filters the logits to retain only the `top_k` highest probability
        tokens, setting the scores of all other tokens to negative infinity.
        This technique is used in sampling-based text generation to
        reduce the vocabulary space from which tokens are selected.
        
        Args:
            logits (`torch.Tensor`): The raw prediction scores for the next token,
                                      shape `(batch_size, vocab_size)`.
            top_k (`int`): The number of highest probability tokens to keep.
                           If `top_k` is 0, no filtering is applied.
        
        Returns:
            `torch.Tensor`: The filtered logits tensor.
        """        
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('inf')
        return logits
    
    def _top_p_filtering(
        self,
        logits: torch.Tensor,
        top_p: float
    ) -> torch.Tensor:
        """
        Applies Top-P (Nucleus) filtering to the logits, retaining a dynamically
        sized set of tokens whose cumulative probability mass exceeds `top_p`.
        This method helps to maintain diversity in generated text while preventing
        the selection of very low-probability tokens.
        
        Args:
            logits (`torch.Tensor`): The raw prediction scores for the next token,
                                      shape `(batch_size, vocab_size)`.
            top_p (`float`): The cumulative probability threshold. Only tokens
                             whose cumulative probability sum up to at least `top_p`
                             are retained. Must be between 0.0 and 1.0.
        
        Returns:
            `torch.Tensor`: The filtered logits tensor.
        """        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 0] = False
        
        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float('inf')
        return logits
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def count_parameters(self, only_trainable: bool = False) -> int:
        """
        Counts the total number of parameters in the model or, optionally,
        only the trainable parameters. This is a fundamental metric for
        understanding model complexity and memory footprint.
        
        Args:
            only_trainable (`bool`, *optional*): If `True`, only parameters
                                                 that require gradients are counted.
                                                 Defaults to `False`.
        
        Returns:
            An integer representing the count of (trainable) parameters.
        """        
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_num_params(self, only_trainable: bool = False) -> int:
        """
        An alias for `count_parameters()`, providing an alternative name
        for retrieving the count of total or trainable parameters.
        
        Args:
            only_trainable (`bool`, *optional*): If `True`, only parameters
                                                 that require gradients are counted.
                                                 Defaults to `False`.
        
        Returns:
            An integer representing the count of (trainable) parameters.
        """        
        return self.count_parameters(only_trainable)
    
    def get_memory_footprint(self) -> Dict[str, Any]:
        """
        Calculates and returns a detailed breakdown of the model's memory footprint,
        including the count and size (in MB) of parameters and buffers.
        This is essential for optimizing memory usage and ensuring the model
        fits within available hardware constraints.
        
        Returns:
            A dictionary containing:
            - `param_count`: Total number of parameters.
            - `param_size_mb`: Memory consumed by parameters in MB.
            - `buffer_count`: Total number of buffers.
            - `buffer_size_mb`: Memory consumed by buffers in MB.
            - `total_size_mb`: Total memory consumed by the model in MB.
            - `total_size_gb`: Total memory consumed by the model in GB.
        """        
        param_size = 0
        param_count = 0
        buffer_size = 0
        buffer_count = 0
        
        for param in self.parameters():
            param_count += param.numel()
            param_size += param.numel() * param.element_size()
        
        for buffer in self.buffers():
            buffer_count += buffer.numel()
            buffer_size += buffer.numel() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        return {
            'param_count': param_count,
            'param_size_mb': param_size / (1024 ** 2),
            'buffer_count': buffer_count,
            'buffer_size_mb': buffer_size / (1024 ** 2),
            'total_size_mb': total_size / (1024 ** 2),
            'total_size_gb': total_size / (1024 ** 3)
        }
    
    def print_model_summary(self, verbose: bool = True):
        """
        Prints a comprehensive summary of the model's configuration,
        architectural details, parameter counts (total and trainable),
        estimated memory footprint, and optional runtime statistics.
        This provides a high-level overview for quick inspection.
        
        Args:
            verbose (`bool`, *optional*): If `True`, includes additional runtime
                                          statistics (steps, tokens processed,
                                          average active parameters) if available.
                                          Defaults to `True`.
        """        
        print(f"  {self.config.model_name} - Model Summary")
        
        print(f"\n[INFO] Configuration:")
        print(f"  Vocabulary Size: {self.config.vocab_size:,}")
        print(f"  Hidden Dimension: {self.config.hidden_size}")
        print(f"  Number of Layers: {self.config.num_layers}")
        print(f"  Number of Heads: {self.config.num_attention_heads}")
        print(f"  Context Length: {self.config.context_length:,}")
        
        print(f"\n🧠 Adaptive Features:")
        print(f"  Width Choices: {self.config.width_choices}")
        print(f"  Depth Range: {self.config.min_depth}-{self.config.max_depth}")
        print(f"  Adaptive Routing: Depth + Width + Path + Expert")
        
        print(f"\n🔀 MoE Configuration:")
        print(f"  Total Experts: {self.config.expert_count}")
        print(f"  Top-K Active: {self.config.top_k_experts}")
        print(f"  Disk Sharded: {self.config.shard_experts}")
        print(f"  Cache Size: {self.config.max_expert_cache} experts")
        
        print(f"\n💭 CoT Configuration:")
        print(f"  Components: {self.config.cot_components}")
        print(f"  Dim per Component: {self.config.cot_dim}")
        print(f"  Total CoT Dimension: {self.config.cot_dim * self.config.cot_components}")
        
        print(f"\n📈 Parameters:")
        total_params = self.count_parameters()
        trainable_params = self.count_parameters(only_trainable=True)
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        # active_params not available at summary time; skip this line
        
        memory = self.get_memory_footprint()
        print(f"\n[INFO] Memory Footprint:")
        print(f"  Parameters: {memory['param_size_mb']:.2f} MB")
        print(f"  Total: {memory['total_size_mb']:.2f} MB ({memory['total_size_gb']:.3f} GB)")
        
        if verbose and len(self._active_params_history) > 0:
            avg_active = sum(self._active_params_history) / len(self._active_params_history)
            print(f"\n[INFO] Runtime Statistics:")
            print(f"  Steps: {self._step_count:,}")
            print(f"  Tokens Processed: {self._total_tokens_processed:,}")
            print(f"  Avg Active Params: {avg_active/1e6:.1f}M")

    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """
        Retrieves statistics pertaining to the model's dynamic routing decisions
        and active parameter usage during its operational lifetime. These statistics
        provide insights into the model's adaptive behavior and efficiency.
        
        Returns:
            A dictionary containing:
            - `avg_active_params`: Average number of active parameters over time.
            - `min_active_params`: Minimum number of active parameters recorded.
            - `max_active_params`: Maximum number of active parameters recorded.
            - `total_steps`: Total number of forward passes executed.
            - `total_tokens`: Total number of tokens processed.
            - `avg_params_per_token`: Average parameters activated per token.
            Returns an empty dictionary if no data has been collected yet.
        """        
        if len(self._active_params_history) == 0:
            return {}
        
        return {
            'avg_active_params': sum(self._active_params_history) / len(self._active_params_history),
            'min_active_params': min(self._active_params_history),
            'max_active_params': max(self._active_params_history),
            'total_steps': self._step_count,
            'total_tokens': self._total_tokens_processed,
            'avg_params_per_token': sum(self._active_params_history) / max(self._total_tokens_processed, 1)
        }
    
    # ==================== CHECKPOINT MANAGEMENT ====================
    
    def save_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        **kwargs
    ):
        """
        Saves the current state of the model and optionally its associated
        optimizer and learning rate scheduler to a checkpoint file. This
        allows for resuming training or deploying the model at a later stage.
        
        Args:
            path (`Union[str, Path]`): The file path where the checkpoint will be saved.
                                        Intermediate directories will be created if needed.
            optimizer (`torch.optim.Optimizer`, *optional*): The optimizer whose state
                                                              should be saved. Defaults to `None`.
            scheduler (`Any`, *optional*): The learning rate scheduler whose state
                                           should be saved. Can be any object with a `state_dict()` method.
                                           Defaults to `None`.
            epoch (`int`, *optional*): The current training epoch number to be recorded
                                       in the checkpoint. Defaults to `None`.
            step (`int`, *optional*): The current training step number to be recorded
                                      in the checkpoint. Defaults to `None`.
            **kwargs: Additional arbitrary keyword arguments to be included
                      in the checkpoint dictionary.
        """        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else None,
            'step_count': self._step_count,
            'total_tokens': self._total_tokens_processed,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if step is not None:
            checkpoint['step'] = step
        
        # Add any additional kwargs
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        strict: bool = True,
        map_location: Optional[Union[str, torch.device]] = None
    ) -> Dict[str, Any]:
        """
        Loads a model checkpoint from a specified file path, restoring the
        model's state, and optionally the optimizer and scheduler states.
        It also updates internal performance tracking statistics if present
        in the checkpoint.
        
        Args:
            path (`Union[str, Path]`): The file path to the checkpoint to be loaded.
            optimizer (`torch.optim.Optimizer`, *optional*): The optimizer whose state
                                                              should be loaded. Defaults to `None`.
            scheduler (`Any`, *optional*): The learning rate scheduler whose state
                                           should be loaded. Defaults to `None`.
            strict (`bool`, *optional*): If `True`, the `state_dict` keys must
                                         exactly match the keys of this module. Defaults to `True`.
            map_location (`Union[str, torch.device]`, *optional*): Specifies how to
                                                                    remap storage locations.
                                                                    Defaults to `None`.
        
        Returns:
            `Dict[str, Any]`: The loaded checkpoint dictionary, which includes
                              model, optimizer, and scheduler states, along with
                              any recorded metadata like step count or epoch.
        
        Raises:
            FileNotFoundError: If the specified checkpoint file does not exist.
        """        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=map_location)
        
        # Load model state
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load tracking stats
        if 'step_count' in checkpoint:
            self._step_count = checkpoint['step_count']
        if 'total_tokens' in checkpoint:
            self._total_tokens_processed = checkpoint['total_tokens']
        
        return checkpoint
    
    # ==================== SPECIAL METHODS ====================
    
    def __repr__(self) -> str:
        """
        Provides a concise string representation of the `zeroModel`,
        summarizing key characteristics such as the total number of parameters,
        number of layers, hidden dimension size, and expert count.
        
        Returns:
            A string representation of the model instance.
        """        
        total_params = self.count_parameters()
        return (
            f"{self.__class__.__name__}("
            f"params={total_params:,}, "
            f"layers={self.config.num_layers}, "
            f"hidden={self.config.hidden_size}, "
            f"experts={self.config.expert_count}"
            f")"
        )
    
    def to(self, *args, **kwargs):
        """
        Overrides the standard `torch.nn.Module.to()` method to ensure that
        the model's components, including the `ShardedExpertFabric` (MoE experts),
        are correctly moved to the specified device (CPU/GPU).
        
        Args:
            *args: Positional arguments typically passed to `torch.nn.Module.to()`,
                   e.g., `device` or `dtype`.
            **kwargs: Keyword arguments typically passed to `torch.nn.Module.to()`,
                      e.g., `device` or `dtype`.
        
        Returns:
            The model instance, moved to the specified device.
        """        # Move main model
        super().to(*args, **kwargs)
        
        # Notify MoE fabric of device change
        if hasattr(self, 'moe'):
            device = args[0] if args else kwargs.get('device')
            if device is not None:
                self.moe.set_device(device)
        
        return self



# ==================== TESTING ====================







