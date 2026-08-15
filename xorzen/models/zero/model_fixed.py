# This file shows ONLY the fixed section - lines 698-728

        # ========== STEP 9: AUXILIARY LOSSES ==========
        # Routing regularization (includes uncertainty loss -> gives uncertainty_estimator a gradient)
        routing_loss = self.routing_regularizer(routing_decision)
        
        # Add router auxiliary losses from routing decision
        if hasattr(routing_decision, 'auxiliary') and routing_decision.auxiliary:
            for aux_loss_val in routing_decision.auxiliary.values():
                if isinstance(aux_loss_val, torch.Tensor):
                    # Ensure auxiliary loss is scalar before adding
                    if aux_loss_val.dim() > 0:
                        aux_loss_val = aux_loss_val.mean()
                    routing_loss = routing_loss + aux_loss_val
        
        # Expert load balancing
        load_balance_loss = self._compute_load_balance_loss(
            expert_indices_flat,
            expert_weights_flat
        )
        
        # CoT consistency loss zeroed during pre-training
        cot_consistency_loss = torch.tensor(0.0, device=device)
        
        # Ensure all losses are scalars before accumulation
        if routing_loss.dim() > 0:
            routing_loss = routing_loss.mean()
        if load_balance_loss.dim() > 0:
            load_balance_loss = load_balance_loss.mean()
        if cot_consistency_loss.dim() > 0:
            cot_consistency_loss = cot_consistency_loss.mean()
        
        # Accumulate all auxiliary losses into the main loss so a single
        # loss.backward() covers ALL parameters (including uncertainty_estimator).
        if loss is not None:
            loss = loss + routing_loss + load_balance_loss
        else:
            # No labels: auxiliary losses still need to be scalar for .backward()
            loss = routing_loss + load_balance_loss
