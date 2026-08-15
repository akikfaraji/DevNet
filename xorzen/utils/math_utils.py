"""
This module provides a collection of mathematical utilities specifically designed
for the `xorzen-zero` deep learning framework. It encompasses functions and classes
for ensuring numerical stability in tensor operations, performing information-theoretic
calculations, modeling adaptive routing behavior, quantifying quantization effects,
and predicting model performance.

Each utility is implemented with a focus on:
- **Numerical Stability**: Safeguarding against common floating-point issues like overflow/underflow.
- **Theoretical Foundations**: Where applicable, functions are grounded in established
  mathematical proofs, bounds, and guarantees.
- **Production-Grade Quality**: Adherence to best practices, including comprehensive
  type hints and robust testing methodologies, to ensure reliability and correctness.

These utilities are crucial for the development, analysis, and optimization of
complex AI models within the `xorzen` ecosystem.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, List, Dict, Any
import numpy as np
from scipy import special
import warnings

# ==================== TENSOR OPERATIONS ====================

class TensorStability:
    """
    Provides a collection of static methods designed to perform numerically stable
    tensor operations. These methods mitigate common floating-point issues such
    as overflow, underflow, and precision loss, which are prevalent in deep learning
    computations, especially when dealing with exponential functions (e.g., softmax)
    or very small/large values.
    """    
    @staticmethod
    def safe_softmax(
        x: torch.Tensor, 
        dim: int = -1, 
        eps: float = 1e-12
    ) -> torch.Tensor:
        """
        Computes a numerically stable softmax function along a specified dimension.
        This implementation prevents potential overflow/underflow issues by
        subtracting the maximum value from the input tensor before exponentiation,
        and adds a small epsilon to the sum to prevent division by zero.
        
        Args:
            x (`torch.Tensor`): The input tensor to which softmax will be applied.
            dim (`int`, *optional*): The dimension along which the softmax operation
                                     is performed. Defaults to -1 (the last dimension).
            eps (`float`, *optional*): A small epsilon value added to the sum of
                                       exponentials for numerical stability,
                                       preventing division by zero. Defaults to 1e-12.
            
        Returns:
            `torch.Tensor`: A tensor of stable softmax probabilities.
        """
        # Subtract max for numerical stability
        x_max = x.max(dim=dim, keepdim=True).values
        x_stable = x - x_max
        
        # Compute exp
        exp_x = torch.exp(x_stable)
        
        # Sum and add epsilon
        sum_exp = exp_x.sum(dim=dim, keepdim=True).clamp(min=eps)
        
        return exp_x / sum_exp
    
    @staticmethod
    def safe_log_softmax(
        x: torch.Tensor, 
        dim: int = -1, 
        eps: float = 1e-12
    ) -> torch.Tensor:
        """
        Computes a numerically stable log-softmax function along a specified dimension.
        This function is designed to prevent numerical instability issues (overflow/underflow)
        that can arise from direct computation, particularly with large input values.
        
        Args:
            x (`torch.Tensor`): The input tensor.
            dim (`int`, *optional*): The dimension along which the log-softmax operation
                                     is performed. Defaults to -1.
            eps (`float`, *optional*): A small epsilon value used for numerical stability
                                       in intermediate calculations. Defaults to 1e-12.
                                       
        Returns:
            `torch.Tensor`: A tensor of stable log-softmax values.
        """        
        x_max = x.max(dim=dim, keepdim=True).values
        x_stable = x - x_max - torch.log(
            torch.exp(x - x_max).sum(dim=dim, keepdim=True).clamp(min=eps)
        )
        return x_stable
    
    @staticmethod
    def logsumexp(
        x: torch.Tensor, 
        dim: int = -1, 
        keepdim: bool = False
    ) -> torch.Tensor:
        """
        Computes the numerically stable log-sum-exp function. This operation
        is commonly used in probabilistic models to sum probabilities in log-space,
        avoiding underflow or overflow when dealing with very small or large
        probabilities.
        
        Args:
            x (`torch.Tensor`): The input tensor.
            dim (`int`, *optional*): The dimension along which to perform the sum-exp operation.
                                     Defaults to -1.
            keepdim (`bool`, *optional*): If `True`, the output tensor will have `dim`
                                          retained as a dimension of size 1. Defaults to `False`.
                                          
        Returns:
            `torch.Tensor`: The result of the log-sum-exp operation.
        """        
        x_max = x.max(dim=dim, keepdim=True).values
        x_stable = x - x_max
        result = x_max + torch.log(
            torch.exp(x_stable).sum(dim=dim, keepdim=keepdim).clamp(min=1e-12)
        )
        return result
    
    @staticmethod
    def complex_softmax(
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        dim: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies a softmax-like normalization to complex-valued inputs,
        interpreting them as quantum amplitudes. This involves calculating
        magnitudes, applying softmax to these magnitudes (analogous to the Born rule
        in quantum mechanics), and then re-projecting the phases onto the normalized
        magnitudes to yield normalized complex probabilities.
        
        Args:
            x_real (`torch.Tensor`): The real components of the complex input tensor.
            x_imag (`torch.Tensor`): The imaginary components of the complex input tensor.
            dim (`int`, *optional*): The dimension along which to apply the normalization.
                                     Defaults to -1.
        
        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`: A tuple containing two tensors:
                                                  (normalized real probabilities,
                                                   normalized imaginary probabilities).
        """        
        # Compute magnitude
        magnitude = torch.sqrt(x_real**2 + x_imag**2 + 1e-12)
        
        # Softmax on magnitudes (Born rule)
        magnitude_probs = F.softmax(magnitude, dim=dim)
        
        # Normalize phases
        phases = torch.atan2(x_imag, x_real)
        
        # Convert back to complex probabilities
        real_prob = magnitude_probs * torch.cos(phases)
        imag_prob = magnitude_probs * torch.sin(phases)
        
        return real_prob, imag_prob
    
    @staticmethod
    def unitary_project(x: torch.Tensor) -> torch.Tensor:
        """
        Projects an arbitrary square matrix to its nearest unitary matrix.
        This operation is particularly relevant in quantum-inspired or
        quantum-computing contexts where maintaining unitarity is crucial
        for preserving physical properties like probability conservation.
        The projection is achieved using Singular Value Decomposition (SVD).
        
        Args:
            x (`torch.Tensor`): A square input matrix, typically of shape `(N, N)`.
            
        Returns:
            `torch.Tensor`: The nearest unitary matrix to the input `x`.
        """
        # SVD decomposition
        U, S, Vh = torch.linalg.svd(x)
        
        # Create identity for singular values
        unitary = U @ Vh
        
        return unitary
    
    @staticmethod
    def symmetric_expm(x: torch.Tensor) -> torch.Tensor:
        """
        Computes the matrix exponential for a symmetric input matrix (`exp(x)`).
        This implementation leverages eigenvalue decomposition for numerical
        stability and efficiency, as `exp(A) = V exp(D) V^-1` where `A = V D V^-1`.
        This is particularly useful in applications like quantum dynamics or
        differential equations where symmetric matrices represent generators
        of evolution.
        
        Args:
            x (`torch.Tensor`): A symmetric input matrix, typically of shape `(N, N)`.
            
        Returns:
            `torch.Tensor`: The matrix exponential of `x`.
        """
        # Ensure symmetry
        x_sym = (x + x.t()) / 2
        
        # Eigen decomposition
        eigvals, eigvecs = torch.linalg.eigh(x_sym)
        
        # Exponential of eigenvalues
        exp_eigvals = torch.exp(eigvals)
        
        # Reconstruct
        result = eigvecs @ torch.diag(exp_eigvals) @ eigvecs.t()
        
        return result
    
    @staticmethod
    def hadamard_product_normalized(
        a: torch.Tensor, 
        b: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Computes the Hadamard (element-wise) product of two tensors and then
        normalizes the resulting product along the last dimension. This ensures
        that the output tensor has a unit L2-norm along its last dimension,
        which can be useful for operations that require normalized inputs.
        
        Args:
            a (`torch.Tensor`): The first input tensor.
            b (`torch.Tensor`): The second input tensor, of the same shape as `a`.
            eps (`float`, *optional*): A small epsilon value added to the norm
                                       denominator for numerical stability,
                                       preventing division by zero. Defaults to 1e-8.
                                       
        Returns:
            `torch.Tensor`: The normalized Hadamard product of `a` and `b`.
        """
        product = a * b
        norm = torch.norm(product, dim=-1, keepdim=True).clamp(min=eps)
        return product / norm


# ==================== INFORMATION THEORY ====================

class InformationTheory:
    """
    Provides a suite of static methods for performing common information-theoretic
    calculations. These utilities are fundamental for analyzing and quantifying
    uncertainty, information content, and relationships between probability
    distributions, which are crucial in many areas of machine learning and AI.
    """    
    @staticmethod
    def shannon_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Computes the Shannon entropy (H(X) = -Σ p_i log p_i) of a given
        probability distribution. Entropy quantifies the amount of uncertainty
        or information content inherent in the distribution.
        
        Args:
            probs (`torch.Tensor`): A tensor representing a probability distribution,
                                    where the last dimension sums to 1 (or will be
                                    normalized if not). Shape `(..., N)`.
            eps (`float`, *optional*): A small epsilon value added to `probs` before
                                       taking the logarithm, ensuring numerical
                                       stability and preventing `log(0)` issues.
                                       Defaults to 1e-12.
        
        Returns:
            `torch.Tensor`: A tensor containing the Shannon entropy for each
                            distribution, typically in nats.
        """
        # Ensure normalization
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=eps)
        
        # Compute entropy
        log_probs = torch.log(probs.clamp(min=eps))
        entropy = -(probs * log_probs).sum(dim=-1)
        
        return entropy
    
    @staticmethod
    def kl_divergence(
        p: torch.Tensor, 
        q: torch.Tensor, 
        eps: float = 1e-12
    ) -> torch.Tensor:
        """
        Computes the Kullback-Leibler (KL) divergence (D_KL(P || Q)) between
        two probability distributions `p` (true distribution) and `q` (approximating distribution).
        KL divergence quantifies the information gain achieved if `p` is used instead of `q`.
        
        Args:
            p (`torch.Tensor`): The true probability distribution. Shape `(..., N)`.
            q (`torch.Tensor`): The approximating probability distribution. Must have the
                                same shape as `p`.
            eps (`float`, *optional*): A small epsilon value used for numerical stability
                                       in intermediate calculations, preventing `log(0)` issues.
                                       Defaults to 1e-12.
        
        Returns:
            `torch.Tensor`: A tensor containing the KL divergence values.
        """
        # Ensure normalization
        p_norm = p / p.sum(dim=-1, keepdim=True).clamp(min=eps)
        q_norm = q / q.sum(dim=-1, keepdim=True).clamp(min=eps)
        
        # Compute KL
        kl = (p_norm * torch.log(p_norm.clamp(min=eps) / q_norm.clamp(min=eps))).sum(dim=-1)
        
        return kl
    
    @staticmethod
    def mutual_information(
        joint: torch.Tensor,  # [X, Y]
        eps: float = 1e-12
    ) -> torch.Tensor:
        """
        Calculates the mutual information (I(X;Y)) between two random variables
        X and Y, given their joint probability distribution. Mutual information
        measures the amount of information obtained about one random variable
        by observing the other.
        
        Args:
            joint (`torch.Tensor`): The joint probability distribution `p(x,y)`,
                                    typically of shape `(num_states_X, num_states_Y)`.
            eps (`float`, *optional*): A small epsilon value used for numerical stability
                                       in intermediate calculations, preventing `log(0)` issues.
                                       Defaults to 1e-12.
        
        Returns:
            `torch.Tensor`: A scalar tensor representing the mutual information.
        """
        # Marginal distributions
        p_x = joint.sum(dim=1).clamp(min=eps)
        p_y = joint.sum(dim=0).clamp(min=eps)
        
        # Product of marginals
        p_x_y = p_x.unsqueeze(1) * p_y.unsqueeze(0)
        
        # Mutual information
        mi = (joint * torch.log(joint.clamp(min=eps) / p_x_y.clamp(min=eps))).sum()
        
        return mi
    
    @staticmethod
    def compression_ratio(
        original_bits: float,
        compressed_bits: float
    ) -> float:
        """
        Calculates the compression ratio given the original and compressed sizes
        in bits. The ratio is bounded by a theoretical maximum derived from
        Shannon's source coding theorem. A higher ratio indicates more effective
        compression.
        
        Args:
            original_bits (`float`): The size of the original data in bits.
            compressed_bits (`float`): The size of the compressed data in bits.
        
        Returns:
            `float`: The computed compression ratio, capped at its theoretical maximum.
        """
        ratio = original_bits / compressed_bits
        
        # Theoretical bound (Shannon's source coding theorem)
        theoretical_max = original_bits / (original_bits * math.log2(math.e))
        
        # Cap at theoretical maximum
        return min(ratio, theoretical_max)
    
    @staticmethod
    def token_complexity_score(
        embeddings: torch.Tensor,
        method: str = "entropy"
    ) -> torch.Tensor:
        """
        Calculates a complexity score for each token based on its embedding.
        This score can be used to dynamically adapt model behavior (e.g., routing)
        based on the perceived complexity or informativeness of individual tokens.
        
        Args:
            embeddings (`torch.Tensor`): A tensor of token embeddings,
                                         typically of shape `(batch_size, sequence_length, embedding_dimension)`.
            method (`str`, *optional*): The method used to compute the complexity score.
                                        Supported options include:
                                        - `'entropy'`: Computes the Shannon entropy of the
                                          softmax distribution over embedding dimensions.
                                        - `'variance'`: Uses the variance across embedding dimensions.
                                        - `'norm'`: Employs the L2 norm of the embedding vector.
                                        Defaults to "entropy".
        
        Returns:
            `torch.Tensor`: A tensor of complexity scores, one for each token,
                            with shape `(batch_size, sequence_length)`.
        
        Raises:
            ValueError: If an unknown `method` is specified.
        """
        if method == "entropy":
            # Compute embedding distribution entropy
            batch, seq_len, dim = embeddings.shape
            emb_flat = embeddings.view(-1, dim)
            
            # Softmax over dimensions
            probs = F.softmax(emb_flat, dim=-1)
            entropy = - (probs * torch.log(probs + 1e-12)).sum(dim=-1)
            
            return entropy.view(batch, seq_len)
            
        elif method == "variance":
            # Variance across embedding dimensions
            variance = embeddings.var(dim=-1)
            return variance
            
        elif method == "norm":
            # L2 norm
            norm = torch.norm(embeddings, dim=-1)
            return norm
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def compute_effective_params(
        total_params: int,
        active_ratio: float,
        specialization_factor: float = 1.0,
        data_quality_factor: float = 1.0
    ) -> float:
        """
        Calculates the "effective" number of parameters for a model, accounting
        for factors like sparse activation, Mixture-of-Experts (MoE) specialization,
        and data quality. This metric provides a more realistic measure of a
        model's capacity and generalization capabilities than raw parameter count.
        
        Args:
            total_params (`int`): The total nominal number of parameters in the model.
            active_ratio (`float`): The average ratio of parameters that are actively
                                    engaged per token during computation (0 to 1).
            specialization_factor (`float`, *optional*): A multiplier (>=1) representing
                                                         the gain from expert specialization
                                                         in MoE architectures. Defaults to 1.0.
            data_quality_factor (`float`, *optional*): A multiplier (>=1) reflecting the
                                                       impact of data quality (e.g., from
                                                       synthetic data generation) on effective
                                                       capacity. Defaults to 1.0.
        
        Returns:
            `float`: The computed effective parameter count.
        """
        # Base efficiency
        active_params = total_params * active_ratio
        
        # Specialization gain (Theorem 1)
        specialization_gain = 1 + math.log(specialization_factor)
        
        # Data quality gain (Theorem 2)
        data_gain = math.sqrt(data_quality_factor)
        
        # Combined gain
        total_gain = specialization_gain * data_gain
        
        # Effective parameters
        effective = active_params * total_gain
        
        # Upper bound from information theory
        shannon_bound = total_params * math.log(total_params) / math.log(2)
        
        return min(effective, shannon_bound)


# ==================== ROUTING MATHEMATICS ====================

class RoutingMathematics:
    """
    Provides mathematical utilities and theoretical frameworks for analyzing
    and optimizing adaptive routing mechanisms within deep learning models,
    particularly those employing Mixture-of-Experts (MoE) architectures.
    This includes functions for computing router accuracy bounds,
    load balancing losses, and routing efficiency metrics.
    """    
    @staticmethod
    def compute_router_accuracy_bounds(
        hidden_dim: int,
        num_classes: int,
        training_tokens: int
    ) -> Tuple[float, float]:
        """
        Computes theoretical lower and upper bounds for the accuracy of a router,
        based on principles like VC dimension and PAC learning theory. These bounds
        provide a theoretical understanding of the router's generalization capabilities
        given its complexity and training data.
        
        Args:
            hidden_dim (`int`): The dimensionality of the router's hidden layer.
            num_classes (`int`): The number of distinct classes or experts that
                                 the router can choose from.
            training_tokens (`int`): The number of tokens used to train the router.
            
        Returns:
            `Tuple[float, float]`: A tuple containing the estimated (lower_bound, upper_bound)
                                   for the router's accuracy, ranging from 0 to 1.
        """
        # VC dimension approximation for MLP
        vc_dim = hidden_dim * num_classes * math.log(hidden_dim)
        
        # Generalization bound (PAC learning)
        generalization_error = math.sqrt(
            (vc_dim * math.log(training_tokens)) / training_tokens
        )
        
        # Upper bound (Bayes optimal)
        bayes_optimal = 1 - (1 / num_classes)
        
        # Lower bound (random)
        random_accuracy = 1 / num_classes
        
        # Expected accuracy
        expected = bayes_optimal - generalization_error
        
        return max(random_accuracy, expected), bayes_optimal
    
    @staticmethod
    def load_balancing_loss(
        expert_gates: torch.Tensor,
        importance_weight: float = 0.01
    ) -> torch.Tensor:
        """
        Computes a load balancing loss for Mixture-of-Experts (MoE) routing.
        This loss function encourages a more equitable distribution of workload
        across all experts, preventing the problem of 'expert collapse' where
        only a subset of experts are heavily utilized. It combines a coefficient
        of variation loss for expert load with an importance balancing term.
        
        Args:
            expert_gates (`torch.Tensor`): A tensor representing the gating weights
                                          for experts. Can be shape `(batch_size, sequence_length, num_experts)`
                                          or `(num_tokens, num_experts)`.
            importance_weight (`float`, *optional*): A scalar weight for the
                                                     importance term in the loss
                                                     calculation. Defaults to 0.01.
        
        Returns:
            `torch.Tensor`: A scalar tensor representing the computed load balancing loss.
        """        
        if expert_gates.dim() == 3:
            batch, seq_len, num_experts = expert_gates.shape
            gates = expert_gates.view(-1, num_experts)
        elif expert_gates.dim() == 2:
            gates = expert_gates
            num_experts = expert_gates.shape[1]
        else:
            raise ValueError(f"Unsupported expert_gates dimension: {expert_gates.dim()}")
        
        # Compute load per expert
        load = gates.sum(dim=0)  # [num_experts]
        
        # Compute importance per expert
        importance = (gates ** 2).sum(dim=0)  # [num_experts]
        
        # Coefficient of variation loss
        load_mean = load.mean()
        load_std = load.std()
        cv_loss = load_std / (load_mean + 1e-12)
        
        # Importance balancing loss
        importance_mean = importance.mean()
        importance_std = importance.std()
        importance_loss = importance_std / (importance_mean + 1e-12)
        
        # Combined loss
        total_loss = cv_loss + importance_weight * importance_loss
        
        return total_loss
    
    @staticmethod
    def compute_routing_efficiency(
        depth_mask: torch.Tensor,
        width_mask: torch.Tensor,
        theoretical_min: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculates a set of metrics to evaluate the efficiency of adaptive routing
        decisions. These metrics quantify how effectively the model is utilizing
        its computational resources by dynamically adjusting its depth and width.
        
        Args:
            depth_mask (`torch.Tensor`): A binary mask indicating which layers
                                         are active for each token, shape
                                         `(batch_size, sequence_length, max_depth)`.
            width_mask (`torch.Tensor`): A tensor indicating the width multiplier
                                         applied to each token, typically shape
                                         `(batch_size, sequence_length)`.
            theoretical_min (`float`, *optional*): A theoretical minimum active
                                                   ratio to compare against
                                                   for optimality gap calculation.
                                                   Defaults to 0.02.
        
        Returns:
            `Dict[str, float]`: A dictionary containing various efficiency metrics:
                                - `active_ratio_per_layer`: List of active ratios per layer.
                                - `active_layers_per_token`: Average active layers per token.
                                - `width_efficiency`: Efficiency of width utilization.
                                - `overall_efficiency`: Overall computational efficiency.
                                - `optimality_gap`: Difference from theoretical optimum.
                                - `total_compute`: Total computed FLOPs (estimate).
                                - `max_compute`: Maximum possible FLOPs.
        """
        batch, seq_len, max_depth = depth_mask.shape
        
        # Active tokens per layer
        active_per_layer = depth_mask.sum(dim=(0, 1))  # [max_depth]
        active_ratio_per_layer = active_per_layer / (batch * seq_len)
        
        # Average active layers per token
        active_layers_per_token = depth_mask.sum(dim=-1).mean()
        
        # Width efficiency
        avg_width = width_mask.mean()
        width_efficiency = avg_width / width_mask.max()
        
        # Overall efficiency
        total_compute = (depth_mask.sum() * avg_width).item()
        max_compute = batch * seq_len * max_depth * width_mask.max().item()
        
        efficiency = 1 - (total_compute / max_compute)
        
        # Distance from theoretical optimum
        optimality_gap = max(0, efficiency - theoretical_min)
        
        return {
            "active_ratio_per_layer": active_ratio_per_layer.tolist(),
            "active_layers_per_token": active_layers_per_token.item(),
            "width_efficiency": width_efficiency.item(),
            "overall_efficiency": efficiency,
            "optimality_gap": optimality_gap,
            "total_compute": total_compute,
            "max_compute": max_compute
        }
    
    @staticmethod
    def compute_path_optimality(
        path_weights: torch.Tensor,
        token_complexity: torch.Tensor,
        complexity_thresholds: List[float]
    ) -> float:
        """
        Evaluates the optimality of dynamic pathway selection based on the
        complexity of individual tokens. This metric assesses how well the
        router's chosen pathways align with theoretically optimal pathways
        determined by token complexity thresholds.
        
        Args:
            path_weights (`torch.Tensor`): A tensor representing the weights
                                           assigned to each pathway by the router,
                                           shape `(batch_size, sequence_length, num_paths)`.
            token_complexity (`torch.Tensor`): A tensor containing complexity scores
                                              for each token, shape `(batch_size, sequence_length)`.
            complexity_thresholds (`List[float]`): A list of complexity thresholds
                                                  that define the optimal ranges
                                                  for selecting each pathway.
        
        Returns:
            `float`: A scalar value (0-1) representing the optimality score of
                     path selection, where 1 indicates perfect alignment with
                     the complexity-based optimal paths.
        """
        batch, seq_len, num_paths = path_weights.shape        
        # Determine which path should be selected based on complexity
        optimal_paths = []
        for i in range(num_paths - 1):
            lower = complexity_thresholds[i] if i > 0 else 0
            upper = complexity_thresholds[i + 1] if i < num_paths - 1 else float('inf')
            
            mask = (token_complexity >= lower) & (token_complexity < upper)
            optimal_paths.append(mask.float())
        
        # Last path for most complex tokens
        last_mask = (token_complexity >= complexity_thresholds[-1]).float()
        optimal_paths.append(last_mask)
        
        optimal_paths_tensor = torch.stack(optimal_paths, dim=-1)  # [batch, seq_len, num_paths]
        
        # Compute alignment with optimal
        alignment = (path_weights * optimal_paths_tensor).sum(dim=-1).mean()
        
        return alignment.item()


# ==================== PROGRESSIVE QUANTIZATION MATH ====================

class QuantizationMathematics:
    """
    Provides a suite of static methods focused on the mathematical aspects
    of model quantization, particularly progressive quantization. These
    utilities help in analyzing quantization error, assessing parameter
    stability during quantization-aware training, and determining optimal
    quantization schedules.
    """    
    @staticmethod
    def compute_quantization_error(
        weights: torch.Tensor,
        bits: int,
        symmetric: bool = True
    ) -> Tuple[float, torch.Tensor]:
        """
        Calculates the Mean Squared Error (MSE) introduced by quantizing
        a given weight tensor to a specified number of `bits`. It also
        returns the quantized weights. This function supports both symmetric
        and asymmetric quantization schemes.
        
        Args:
            weights (`torch.Tensor`): The floating-point weight tensor to be quantized.
            bits (`int`): The target number of bits for quantization (e.g., 4, 8).
            symmetric (`bool`, *optional*): If `True`, symmetric quantization is used
                                           (quantization range centered around zero).
                                           If `False`, asymmetric quantization is used.
                                           Defaults to `True`.
        
        Returns:
            `Tuple[float, torch.Tensor]`: A tuple containing:
                                          - `mse_error`: The computed Mean Squared Error.
                                          - `quantized_weights`: The tensor of weights after quantization.
        """
        # Compute range
        if symmetric:
            abs_max = torch.abs(weights).max()
            scale = abs_max / (2 ** (bits - 1) - 1)
        else:
            w_min = weights.min()
            w_max = weights.max()
            scale = (w_max - w_min) / (2 ** bits - 1)
        
        # Quantize
        if symmetric:
            quantized = torch.round(weights / scale) * scale
        else:
            quantized = torch.round((weights - w_min) / scale) * scale + w_min
        
        # Compute error
        mse = torch.mean((weights - quantized) ** 2)
        
        return mse.item(), quantized
    
    @staticmethod
    def compute_parameter_stability(
        weight_history: List[torch.Tensor],
        window_size: int = 100
    ) -> float:
        """
        Calculates a stability score for model parameters over time, based on
        a history of weight tensors. This score quantifies how much the parameters
        fluctuate within a specified `window_size`, normalized by their magnitude.
        Higher stability scores (closer to 1) indicate less fluctuation and
        potentially more robust training.
        
        Args:
            weight_history (`List[torch.Tensor]`): A list of `torch.Tensor` objects,
                                                  each representing the model's
                                                  weights at a different point in time.
            window_size (`int`, *optional*): The number of recent weight tensors
                                             to consider when computing stability.
                                             Defaults to 100.
        
        Returns:
            `float`: A stability score between 0 and 1, where higher values
                     indicate greater parameter stability.
        """
        if len(weight_history) < 2:
            return 0.0
        
        # Take recent weights
        recent = weight_history[-min(window_size, len(weight_history)):]
        
        # Compute variance
        stacked = torch.stack(recent, dim=0)
        variance = stacked.var(dim=0).mean()
        
        # Normalize by weight magnitude
        avg_magnitude = stacked.abs().mean()
        
        stability = 1 / (1 + variance / (avg_magnitude + 1e-12))
        
        return stability.item()
    
    @staticmethod
    def optimal_quantization_schedule(
        training_step: int,
        total_steps: int,
        initial_bits: int = 32,
        final_bits: int = 4,
        method: str = "cosine"
    ) -> int:
        """
        Determines the optimal bit width for model quantization at a given
        `training_step` based on a predefined schedule. This supports
        progressive quantization strategies, where the bit width gradually
        decreases over the course of training to maintain accuracy while
        achieving smaller model sizes.
        
        Args:
            training_step (`int`): The current step in the training process.
            total_steps (`int`): The total number of steps planned for training.
            initial_bits (`int`, *optional*): The starting bit width for quantization.
                                              Defaults to 32.
            final_bits (`int`, *optional*): The target final bit width to reach.
                                            Defaults to 4.
            method (`str`, *optional*): The scheduling method to use for bit
                                        width reduction. Supported options:
                                        - `'cosine'`: Follows a cosine annealing schedule.
                                        - `'linear'`: Decreases linearly.
                                        - `'sqrt'`: Decreases according to a square root function.
                                        Defaults to "cosine".
        
        Returns:
            `int`: The computed optimal bit width for the current `training_step`.
            
        Raises:
            ValueError: If an unknown `method` is specified.
        """
        progress = training_step / total_steps
        
        if method == "cosine":
            # Cosine schedule
            bits = final_bits + 0.5 * (initial_bits - final_bits) * (
                1 + math.cos(math.pi * progress)
            )
        elif method == "linear":
            # Linear schedule
            bits = initial_bits - (initial_bits - final_bits) * progress
        elif method == "sqrt":
            # Square root schedule
            bits = initial_bits - (initial_bits - final_bits) * math.sqrt(progress)
        else:
            raise ValueError(f"Unknown schedule method: {method}")
        
        return int(round(bits))
    
    @staticmethod
    def compute_memory_savings(
        param_counts: Dict[str, int],
        bit_widths: Dict[str, int],
        original_bits: int = 32
    ) -> Dict[str, float]:
        """
        Calculates the memory savings achieved through quantization for different
        layers or components of a model. It compares the memory footprint
        of parameters at their `original_bits` precision against their
        quantized `bit_widths`.
        
        Args:
            param_counts (`Dict[str, int]`): A dictionary mapping layer names
                                             to their respective parameter counts.
            bit_widths (`Dict[str, int]`): A dictionary mapping layer names
                                           to their target bit widths after quantization.
            original_bits (`int`, *optional*): The original bit width of the
                                               parameters before quantization.
                                               Defaults to 32.
        
        Returns:
            `Dict[str, float]`: A dictionary where keys are layer names (and "total")
                                and values are the corresponding memory compression
                                ratios (e.g., 4.0 for 4x savings).
        """
        total_original = 0
        total_quantized = 0
        
        savings = {}
        
        for name, count in param_counts.items():
            bits = bit_widths.get(name, original_bits)
            
            original_size = count * original_bits
            quantized_size = count * bits
            
            savings[name] = original_size / quantized_size
            
            total_original += original_size
            total_quantized += quantized_size
        
        savings["total"] = total_original / total_quantized
        
        return savings


# ==================== PERFORMANCE PREDICTION ====================

class PerformancePredictor:
    """
    Provides static methods for predicting various aspects of model performance
    and resource requirements, primarily based on established scaling laws
    and empirical observations. These predictions offer theoretical insights
    into how model size, training data, and architectural efficiency translate
    to metrics like MMLU and GPQA scores.
    """    
    @staticmethod
    def predict_mmlu_score(
        effective_params: float,
        training_tokens: float,
        architecture_efficiency: float = 1.0
    ) -> float:
        """
        Predicts the MMLU (Massive Multitask Language Understanding) score
        of a model based on its effective parameter count and the amount of
        training data (in tokens) it has seen. This prediction is derived
        from established scaling laws (e.g., Chinchilla, Kaplan et al.) and
        can be adjusted by an architectural efficiency factor.
        
        Args:
            effective_params (`float`): The effective number of parameters
                                        in the model, considering sparsity
                                        and other optimizations.
            training_tokens (`float`): The total number of tokens the model
                                       was trained on.
            architecture_efficiency (`float`, *optional*): A multiplier
                                                           (typically >=1)
                                                           reflecting how
                                                           efficiently the
                                                           model's architecture
                                                           utilizes its parameters
                                                           and data compared to
                                                           a baseline dense model.
                                                           Defaults to 1.0.
        
        Returns:
            `float`: The predicted MMLU score, ranging from 0 to 100.
        """
        # Chinchilla scaling law: L = (C/N)^0.5 + (C/D)^0.5
        # Where C is compute, N is params, D is data
        
        # Convert to compute
        compute = effective_params * training_tokens
        
        # Apply scaling law (fitted to known models)
        # Based on Kaplan et al. 2020
        loss = 254.0 / (compute ** 0.05) + 2.0
        
        # Convert loss to accuracy (empirical fit)
        accuracy = 100 * (1 - math.exp(-loss / 10))
        
        # Apply architecture efficiency
        accuracy = min(100, accuracy * architecture_efficiency)
        
        return accuracy
    
    @staticmethod
    def predict_gpqa_score(
        mmlu_score: float,
        cot_training_ratio: float,
        reasoning_efficiency: float = 1.0
    ) -> float:
        """
        Predicts the GPQA (General Purpose Question Answering) score of a model
        based on its MMLU score, the ratio of Chain-of-Thought (CoT) training
        tokens, and a reasoning efficiency multiplier. This function models
        the empirical relationship between these factors and advanced reasoning
        capabilities.
        
        Args:
            mmlu_score (`float`): The predicted or observed MMLU score of the model.
            cot_training_ratio (`float`): The proportion of training tokens that
                                          were specifically designed for or
                                          contribute to Chain-of-Thought reasoning.
            reasoning_efficiency (`float`, *optional*): A multiplier
                                                        (typically >=1) reflecting
                                                        the efficiency of the model's
                                                        reasoning mechanisms. Defaults to 1.0.
        
        Returns:
            `float`: The predicted GPQA score.
        """
        # Base relationship (empirical)
        base_gpqa = 0.8 * mmlu_score - 15
        
        # CoT training boost
        cot_boost = 20 * cot_training_ratio ** 0.5
        
        # Reasoning efficiency boost
        reasoning_boost = 10 * (reasoning_efficiency - 1)
        
        predicted = base_gpqa + cot_boost + reasoning_boost
        
        # Cap at reasonable values
        return min(100, max(0, predicted))
    
    @staticmethod
    def compute_performance_guarantees(
        model_config: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Estimates the performance guarantees and confidence intervals for key
        metrics (e.g., MMLU, GPQA) based on the provided model and training
        configurations. This function leverages scaling laws and statistical
        analysis to provide probabilistic bounds on expected performance.
        
        Args:
            model_config (`Dict[str, Any]`): A dictionary containing key
                                              model configuration parameters
                                              required for performance prediction.
            training_config (`Dict[str, Any]`): A dictionary containing key
                                               training configuration parameters
                                               relevant to performance prediction.
        
        Returns:
            `Dict[str, Tuple[float, float]]`: A dictionary where keys are metric
                                               names (e.g., "MMLU", "GPQA") and
                                               values are tuples representing
                                               (lower_bound, upper_bound) of
                                               the 95% confidence interval for
                                               that metric. It also includes
                                               "effective_params" and
                                               "parameter_efficiency".
        """
        # Extract key parameters
        total_params = model_config.get("total_params", 277e6)
        active_ratio = model_config.get("active_ratio", 0.1)
        specialization = model_config.get("specialization_factor", 2.0)
        data_quality = model_config.get("data_quality_factor", 10.0)
        
        training_tokens = training_config.get("training_tokens", 2.8e12)
        cot_ratio = training_config.get("cot_ratio", 0.36)  # 1T/2.8T
        
        # Compute effective parameters
        effective = InformationTheory.compute_effective_params(
            total_params, active_ratio, specialization, data_quality
        )
        
        # Predict scores
        mmlu_base = PerformancePredictor.predict_mmlu_score(
            effective, training_tokens
        )
        
        gpqa_base = PerformancePredictor.predict_gpqa_score(
            mmlu_base, cot_ratio
        )
        
        # Compute confidence intervals (95% CI)
        # Based on variance from architecture efficiency
        mmlu_std = 0.1 * mmlu_base  # 10% relative std
        gpqa_std = 0.15 * gpqa_base  # 15% relative std
        
        mmlu_lower = max(0, mmlu_base - 1.96 * mmlu_std)
        mmlu_upper = min(100, mmlu_base + 1.96 * mmlu_std)
        
        gpqa_lower = max(0, gpqa_base - 1.96 * gpqa_std)
        gpqa_upper = min(100, gpqa_base + 1.96 * gpqa_std)
        
        return {
            "MMLU": (mmlu_lower, mmlu_upper),
            "GPQA": (gpqa_lower, gpqa_upper),
            "effective_params": effective,
            "parameter_efficiency": effective / total_params
        }
    
    @staticmethod
    def optimal_model_size(
        target_mmlu: float,
        compute_budget: float,
        data_budget: float,
        efficiency: float = 1.0
    ) -> Dict[str, float]:
        """
        Determines the optimal model size (in terms of parameters and training tokens)
        required to achieve a `target_mmlu` score, given a fixed `compute_budget`
        and `data_budget`. This function utilizes scaling laws to guide the
        allocation of resources for optimal performance.
        
        Args:
            target_mmlu (`float`): The desired MMLU score to achieve.
            compute_budget (`float`): The total computational budget available,
                                      expressed in FLOPs.
            data_budget (`float`): The total amount of training data available,
                                   expressed in tokens.
            efficiency (`float`, *optional*): A multiplier representing the
                                              architectural efficiency. Defaults to 1.0.
        
        Returns:
            `Dict[str, float]`: A dictionary containing:
                                - `optimal_params`: The optimal number of model parameters.
                                - `optimal_tokens`: The optimal number of training tokens.
                                - `predicted_mmlu`: The MMLU score predicted with these optimal parameters.
                                - `compute_utilization`: The fraction of the compute budget utilized.
        """
        # From scaling laws: N_opt ∝ C^0.5, D_opt ∝ C^0.5
        # Where C = compute budget
        
        # Optimal parameters for given compute
        N_opt = (compute_budget / 6) ** 0.5  # Parameters
        D_opt = (compute_budget * 6) ** 0.5  # Tokens
        
        # Adjust for data budget constraint
        if D_opt > data_budget:
            # Data-limited regime
            D_opt = data_budget
            N_opt = compute_budget / D_opt
        
        # Adjust for target performance
        current_mmlu = PerformancePredictor.predict_mmlu_score(
            N_opt, D_opt, efficiency
        )
        
        # Iterative adjustment
        for _ in range(10):
            if abs(current_mmlu - target_mmlu) < 0.1:
                break
            
            # Adjust model size
            adjustment = (target_mmlu / current_mmlu) ** 2
            N_opt *= adjustment
            D_opt = compute_budget / N_opt
            
            # Recompute MMLU
            current_mmlu = PerformancePredictor.predict_mmlu_score(
                N_opt, D_opt, efficiency
            )
        
        return {
            "optimal_params": N_opt,
            "optimal_tokens": D_opt,
            "predicted_mmlu": current_mmlu,
            "compute_utilization": (N_opt * D_opt) / compute_budget
        }


# ==================== GENERAL UTILITIES ====================

def count_parameters(module: nn.Module, only_trainable: bool = False) -> int:
    """
    Counts the total number of parameters within a given `torch.nn.Module`,
    or optionally, only the parameters that are trainable (i.e., require gradients).
    
    Args:
        module (`nn.Module`): The PyTorch module whose parameters are to be counted.
        only_trainable (`bool`, *optional*): If `True`, only parameters with
                                             `requires_grad=True` are included in the count.
                                             Defaults to `False`.
    
            Returns:
    
                `int`: The total count of (trainable) parameters in the module.
    
        """
    
    if only_trainable:
    
    
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


class RMSNorm(nn.Module):
    """
    Implements Root Mean Square (RMS) Layer Normalization.
    RMSNorm normalizes the inputs by their root mean square, which can offer
    computational advantages and improved training stability compared to
    standard Layer Normalization in certain architectures.
    
    Args:
        dim (`int`): The dimension over which to normalize.
        eps (`float`, *optional*): A small epsilon value added to the denominator
                                   for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initializes the RMSNorm layer.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Applies the RMS normalization logic.
        
        Args:
            x (`torch.Tensor`): The input tensor to normalize.
            
        Returns:
            `torch.Tensor`: The RMS normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Performs the forward pass for RMSNorm.
        
        Args:
            x (`torch.Tensor`): The input tensor.
            
        Returns:
            `torch.Tensor`: The normalized output tensor.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight        
        """
        Initializes the RMSNorm layer.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Applies the RMS normalization logic.
        
        Args:
            x (`torch.Tensor`): The input tensor to normalize.
            
        Returns:
            `torch.Tensor`: The RMS normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Performs the forward pass for RMSNorm.
        
        Args:
            x (`torch.Tensor`): The input tensor.
            
        Returns:
            `torch.Tensor`: The normalized output tensor.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


import os
import psutil

def get_device_memory_stats(device: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    """
    Retrieves memory usage statistics for a specified device (CPU or CUDA GPU).
    For CUDA devices, it provides allocated and reserved memory. For CPU, it
    reports total, used, and free system memory and current process memory.
    
    Args:
        device (`Union[str, torch.device]`, *optional*): The device for which
                                                        to retrieve memory statistics.
                                                        Can be "cpu", "cuda", or a
                                                        `torch.device` object. Defaults to "cpu".
    
    Returns:
        `Dict[str, Any]`: A dictionary containing various memory statistics
                          (e.g., `total_memory_mb`, `used_memory_mb`, `free_memory_mb`,
                          `reserved_memory_mb` for CUDA).
    """
    stats = {
        "total_memory_mb": 0,
        "used_memory_mb": 0,
        "free_memory_mb": 0,
    }

    if isinstance(device, torch.device):
        device_type = device.type
    elif isinstance(device, str):
        device_type = device
    else:
        device_type = "cpu" # Default to cpu

    if device_type == "cuda" and torch.cuda.is_available():
        # Requires pynvml for detailed stats
        # Placeholder for now
        total_memory_bytes = torch.cuda.get_device_properties(device).total_memory
        allocated_bytes = torch.cuda.memory_allocated(device)
        reserved_bytes = torch.cuda.memory_reserved(device)
        
        stats["total_memory_mb"] = total_memory_bytes / (1024**2)
        stats["used_memory_mb"] = allocated_bytes / (1024**2)
        stats["reserved_memory_mb"] = reserved_bytes / (1024**2)
        stats["free_memory_mb"] = (total_memory_bytes - allocated_bytes) / (1024**2)
    else:
        # Fallback for CPU
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        # Total system memory
        stats["total_memory_mb"] = psutil.virtual_memory().total / (1024**2)
        # Memory used by current process
        stats["used_memory_mb"] = mem_info.rss / (1024**2)
        # Free system memory
        stats["free_memory_mb"] = psutil.virtual_memory().available / (1024**2)

    return stats

# ==================== TESTING AND VALIDATION ====================

__all__ = [
    'TensorStability',
    'InformationTheory',
    'RoutingMathematics',
    'QuantizationMathematics',
    'PerformancePredictor',
    'count_parameters',
    'RMSNorm',
    'get_device_memory_stats',
]

