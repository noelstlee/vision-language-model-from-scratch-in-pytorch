"""
Vision-Language Model from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - split_image_into_patches
import torch

def split_image_into_patches(image, patch_size):
    """Split an image tensor (B, C, H, W) into a sequence of (patch_size, patch_size) patches.

    Returns a tensor of shape (B, num_patches, C, patch_size, patch_size) in row-major order.
    """
    # TODO: split the (B, C, H, W) image into (B, num_patches, C, patch_size, patch_size).
    B, C, H, W = image.shape
    num_patches = (H//patch_size) * (W//patch_size)
    # reshape
    reshaped = image.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    # permute
    permuted = reshaped.permute(0, 2, 4, 1, 3, 5) # (B, H/P, W/P, C, P, P)
    # reshape to flatten the grid axes into N patches
    out = permuted.reshape(B, num_patches, C, patch_size, patch_size)
    return out

# Step 2 - flatten_patches
def flatten_patches(patches):
    # TODO: flatten each patch's channel and spatial dims into one vector, keep (B, N) leading dims.
    B, N, C, P1, P2 = patches.shape
    return patches.reshape(B, N, C * P1 * P2)

# Step 3 - linear_projection
import torch

def linear_projection(x, weight, bias):
    """Apply y = x @ weight.T + bias with arbitrary leading dims on x."""
    # TODO: compute the affine map y = x @ weight.T + bias
    return x @ weight.T + bias

# Step 4 - project_patches_to_embeddings
import torch

def project_patches_to_embeddings(flat_patches, patch_proj_weight, patch_proj_bias):
    # TODO: Linearly project flattened image patches into the ViT embedding dimension.
    return linear_projection(flat_patches, patch_proj_weight, patch_proj_bias)

# Step 5 - prepend_class_token
import torch

def prepend_class_token(patch_embeddings, class_token):
    """Prepend a learnable [CLS] token to the patch embedding sequence.

    patch_embeddings: (B, num_patches, embed_dim)
    class_token:      (1, 1, embed_dim)
    returns:          (B, num_patches+1, embed_dim)
    """
    # TODO: prepend the [CLS] token to every sequence in the batch
    B, N, D = patch_embeddings.shape
    ex_class_token = class_token.expand(B, -1, -1) # (B, 1, D)
    return torch.cat((ex_class_token, patch_embeddings), dim=1) # (B, N, D) concat (B, 1, D) = (B, N+1, D)

# Step 6 - add_position_embeddings
import torch

def add_position_embeddings(tokens, position_embeddings):
    """Add learnable position embeddings to a (B, S, D) token sequence."""
    # TODO: combine tokens (B, S, D) with position_embeddings (1, S, D) via broadcasting.
    return tokens + position_embeddings # (B, S, D)

# Step 7 - compute_attention_scores
import torch

def compute_attention_scores(q, k):
    """Compute raw attention scores Q @ K^T.

    q: (..., Sq, d_head)
    k: (..., Sk, d_head)
    returns: (..., Sq, Sk)
    """
    # TODO: compute the raw attention scores as Q times K-transpose
    return torch.matmul(q, k.transpose(-2, -1))

# Step 8 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_head):
    """Scale raw attention scores so softmax inputs stay well-conditioned."""
    # TODO: Divide raw attention scores by a constant derived from d_head.
    return scores / torch.sqrt(torch.tensor(d_head))

# Step 9 - apply_attention_mask
def apply_attention_mask(scores, mask):
    # TODO: add an additive mask (0 = allowed, -inf = blocked) to attention scores.
    return scores + mask if mask is not None else scores

# Step 10 - attention_softmax
import torch

def attention_softmax(masked_scores):
    """Softmax over the last (key) axis of attention scores."""
    # TODO: convert masked attention scores into normalized weights over the key axis
    return torch.softmax(masked_scores, dim=-1)

# Step 11 - attention_context
import torch

def attention_context(attn_weights, v):
    """Combine attention weights with values to produce context vectors."""
    # TODO: return a tensor of shape (..., Sq, d_head) from attn_weights and v
    return torch.matmul(attn_weights, v)

# Step 12 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(q, k, v, mask=None):
    """Compose score, scale, mask, softmax, and context into full attention."""
    # TODO: compose the five attention primitives into a single forward pass.
    # (q*k^t) / sqrt(d_k):
    d_head = q.shape[-1]
    pre_norm_scores = scale_attention_scores(compute_attention_scores(q,k), d_head)
    # apply masking
    masked_scores = apply_attention_mask(pre_norm_scores, mask)
    # apply softmax
    attn_weights = attention_softmax(masked_scores)
    # apply values to produce context vectors
    output = attention_context(attn_weights, v)

    return output

# Step 13 - split_into_heads
import torch

def split_into_heads(x, num_heads):
    """Reshape (B, S, d_model) into (B, num_heads, S, d_head)."""
    # TODO: split the last dim into (num_heads, d_head) and move heads next to batch
    B, S, d_model = x.shape
    d_head = d_model // num_heads
    # split last axis into two: (B, S, nums_head, d_head)
    x = x.reshape(B, S, num_heads, d_head)
    x = x.transpose(-2, -3) # (B, nums_head, S, d_head)
    return x

# Step 14 - merge_heads
import torch

def merge_heads(x):
    """Merge (B, num_heads, S, d_head) back to (B, S, num_heads*d_head)."""
    # TODO: merge the multi-head dimension back into the model dimension
    B, num_heads, S, d_head = x.shape
    # transpose num_heads axis and S (sequence):
    x = x.transpose(-2, -3) # (B, S, num_heads, d_head)
    x = x.reshape(B, S, num_heads * d_head)
    return x

# Step 15 - project_qkv
def project_qkv(x, wq, bq, wk, bk, wv, bv):
    """
    produce the query, key, and value tensors from input sequence 'x'
    Inputs:
        - x: (B, S, d_model) -> S: Sequence (number of tokens), d_model: Embedding Space
        - wq, wk, wv: weights of query, key, value (d_model, d_model)
        - bq, bk, bv: bias of query, key, value  (d_model,)
    """
    # TODO: project x into separate query, key, and value tensors using three linear layers.
    q = linear_projection(x, wq, bq)
    k = linear_projection(x, wk, bk)
    v = linear_projection(x, wv, bv)
    return (q, k, v)

# Step 16 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    """
    Inputs:
        - q,k,v: (B, S, d_model)
        - num_heads: int
    Return:
        multi-head form of each tensor as a tuple
        (q_h, k_h, v_h) where each output has a shape (B, num_heads, S, d_head)
    """
    B, S, d_model = q.shape
    q_h = split_into_heads(q, num_heads)
    k_h = split_into_heads(k, num_heads)
    v_h = split_into_heads(v, num_heads)
    return (q_h, k_h, v_h)

# Step 17 - multi_head_attention_scores
import torch

def multi_head_attention_scores(q_h, k_h, v_h, mask=None):
    """Run scaled dot-product attention in parallel across all heads.

    q_h, k_h, v_h: (B, num_heads, S, d_head)
    mask: broadcastable to (B, num_heads, S, S) or None
    returns: (B, num_heads, S, d_head)
    """
    # TODO: run scaled dot-product attention across the head axis
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

# Step 18 - merge_and_output_project
import torch

def merge_and_output_project(context_heads, wo, bo):
    """
    Merge heads back to d_model and apply the output projection."""
    # TODO: merge multi-head context to (B, S, d_model) then apply linear projection with wo, bo
    # merge multi head context (B, S, d_model)
    # (B, H, S, d_head) -> (B, S, H * d_head) = (B, S, d_model)
    model_dim = merge_heads(context_heads)
    # linear projection; this is what lets differnt heads communicate; without it each head would write into a fixed slice of the residual stream
    return linear_projection(model_dim, wo, bo)

# Step 19 - multi_head_self_attention
import torch

def multi_head_self_attention(x, params, num_heads, mask=None):
    """Run full multi-head self-attention: QKV proj, head split, attention, merge, output proj."""
    # TODO: compose project_qkv, split_qkv_into_heads, multi_head_attention_scores, merge_and_output_project.
    qkv_tuple = project_qkv(x, params["wq"], params["bq"], params["wk"], params["bk"], params["wv"], params["bv"]) # (q,k,v)
    multi_head_qkv = split_qkv_into_heads(qkv_tuple[0], qkv_tuple[1], qkv_tuple[2], num_heads) # (q_h, k_h, v_h)
    context_heads = multi_head_attention_scores(multi_head_qkv[0], multi_head_qkv[1], multi_head_qkv[2], mask) # shape: (B, num_heads, S, d_head)
    return merge_and_output_project(context_heads, params["wo"], params["bo"])

# Step 20 - gelu_activation
import torch

def gelu_activation(x):
    """Apply the exact (erf-based) GELU activation elementwise to x."""
    # TODO: implement GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))

# Step 21 - mlp_first_layer
import torch

def mlp_first_layer(x, w1, b1):
    """
    Apply the first linear layer of the MLP block followed by GELU.
    Inputs:
        - x: input sequence (B, S, d_model); B: Batch, S: Sequence (number of tokens), d_model: Embedding Space
        - w1, b1: MLP weight, bias; (d_ff, d_model), (d_ff,); d_ff: wider feed-forward dimension
    """
    # TODO: project x to the feed-forward dimension and apply GELU
    # apply linear projection first:
    expand_embed = linear_projection(x, w1, b1)
    # apply gelu:
    return gelu_activation(expand_embed)

# Step 22 - mlp_second_layer (not yet solved)
# TODO: implement

# Step 23 - mlp_block (not yet solved)
# TODO: implement

# Step 24 - compute_layernorm_stats (not yet solved)
# TODO: implement

# Step 25 - layer_norm (not yet solved)
# TODO: implement

# Step 26 - residual_add (not yet solved)
# TODO: implement

# Step 27 - pre_norm_sublayer (not yet solved)
# TODO: implement

# Step 28 - vision_encoder_block (not yet solved)
# TODO: implement

# Step 29 - vision_encoder (not yet solved)
# TODO: implement

# Step 30 - extract_patch_features (not yet solved)
# TODO: implement

# Step 31 - projector_first_layer (not yet solved)
# TODO: implement

# Step 32 - projector_second_layer (not yet solved)
# TODO: implement

# Step 33 - vision_language_projector (not yet solved)
# TODO: implement

# Step 34 - build_token_vocabulary (not yet solved)
# TODO: implement

# Step 35 - encode_text_to_ids (not yet solved)
# TODO: implement

# Step 36 - embed_token_ids (not yet solved)
# TODO: implement

# Step 37 - add_text_position_embeddings (not yet solved)
# TODO: implement

# Step 38 - find_image_placeholder_positions (not yet solved)
# TODO: implement

# Step 39 - insert_image_tokens (not yet solved)
# TODO: implement

# Step 40 - build_multimodal_embeddings (not yet solved)
# TODO: implement

# Step 41 - build_label_tensor (not yet solved)
# TODO: implement

# Step 42 - build_causal_mask (not yet solved)
# TODO: implement

# Step 43 - decoder_block (not yet solved)
# TODO: implement

# Step 44 - language_model_decoder (not yet solved)
# TODO: implement

# Step 45 - final_layer_norm (not yet solved)
# TODO: implement

# Step 46 - language_model_head (not yet solved)
# TODO: implement

# Step 47 - encode_image_to_tokens (not yet solved)
# TODO: implement

# Step 48 - vision_language_forward (not yet solved)
# TODO: implement

# Step 49 - shift_logits_and_labels (not yet solved)
# TODO: implement

# Step 50 - per_position_cross_entropy (not yet solved)
# TODO: implement

# Step 51 - masked_mean_loss (not yet solved)
# TODO: implement

# Step 52 - greedy_next_token (not yet solved)
# TODO: implement

# Step 53 - apply_temperature (not yet solved)
# TODO: implement

# Step 54 - top_k_filter (not yet solved)
# TODO: implement

# Step 55 - sample_from_logits (not yet solved)
# TODO: implement

# Step 56 - generate_caption (not yet solved)
# TODO: implement

# Step 57 - initialize_vlm_parameters (not yet solved)
# TODO: implement

# Step 58 - collect_parameters (not yet solved)
# TODO: implement

# Step 59 - zero_gradients (not yet solved)
# TODO: implement

# Step 60 - training_step (not yet solved)
# TODO: implement

# Step 61 - apply_gradient_update (not yet solved)
# TODO: implement

# Step 62 - run_training_loop (not yet solved)
# TODO: implement

