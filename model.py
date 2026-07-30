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

# Step 22 - mlp_second_layer
import torch

def mlp_second_layer(h, w2, b2):
    """
    Inputs:
        - h: (B, S, d_ff); post-acitvation hidden tensor
    """
    # TODO: project the MLP hidden activations back down to d_model using w2 and b2
    return linear_projection(h, w2, b2)

# Step 23 - mlp_block
import torch

def mlp_block(x, params):
    """Two-layer position-wise MLP with GELU between the layers."""
    # TODO: Assemble the position-wise two-layer MLP block with GELU between layers.
    # expansion
    expanded_embeddings = mlp_first_layer(x, params["w1"], params["b1"])
    # compresssion
    compressed_emebeddings = mlp_second_layer(expanded_embeddings, params["w2"], params["b2"])
    return compressed_emebeddings

# Step 24 - compute_layernorm_stats
import torch

def compute_layernorm_stats(x, eps=1e-5):
    """
    Inputs:
    - x (B, S, d_model)
    """
    # TODO: return (mean, var) along the last dim, each with shape (..., 1).
    mean = torch.mean(x, dim=-1, keepdims=True) # (B, S, 1)
    var = torch.var(x, dim=-1, keepdims=True, unbiased=False) # (B, S, 1)
    return (mean, var)

# Step 25 - layer_norm
import torch

def layer_norm(x, gamma, beta, eps=1e-5):
    # TODO: normalize the last dim of x and apply learnable scale gamma and shift beta
    mean, var = compute_layernorm_stats(x, eps)
    layer_norm = (x - mean) / torch.sqrt(var + eps)
    y = layer_norm * gamma + beta
    return y

# Step 26 - residual_add
import torch

def residual_add(residual, sublayer_output):
    """Add residual skip connection to a sublayer's output."""
    # TODO: return the element-wise sum of residual and sublayer_output
    return residual + sublayer_output

# Step 27 - pre_norm_sublayer
import torch

def pre_norm_sublayer(x, gamma, beta, sublayer_fn):
    """Apply pre-norm: LN(x) -> sublayer -> add residual x."""
    # TODO: layer-normalize x, run sublayer_fn on it, then add the residual
    # layer normalize x
    norm_x = layer_norm(x, gamma, beta, eps=1e-5) # (B, S, d_model)
    return residual_add(x, sublayer_fn(norm_x))

# Step 28 - vision_encoder_block
import torch

def vision_encoder_block(x, block_params, num_heads):
    """
    Inputs:
        - x: input (B, S, d_model)
        - block_params
        - num_heads: number of heads
    conceptual flow:  x → LN_attn → attn → add to x → y → LN_mlp → mlp → add to y
    """
    # TODO: pre-norm MHSA sublayer, then pre-norm MLP sublayer, both with residuals.
    # create a lambda func
    attn_lambda = lambda norm_x: multi_head_self_attention(norm_x, block_params["attn"], num_heads, mask=None)
    MHSA = pre_norm_sublayer(x, block_params["ln1_gamma"], block_params["ln1_beta"], attn_lambda)
    mlp_lambda = lambda norm_x: mlp_block(norm_x, block_params["mlp"])
    MLP = pre_norm_sublayer(MHSA, block_params["ln2_gamma"], block_params["ln2_beta"], mlp_lambda)
    return MLP

# Step 29 - vision_encoder
import torch

def vision_encoder(patch_sequence, encoder_params, num_heads):
    """
    Stack ViT encoder blocks then apply a final layer norm to the patch sequence.
    Inputs: 
        - patch_sequence: teh input tensor that enters the encoder (B, S, d_model)
    We feed this patch_sequence through the encoder_blocks in order each block receiving the output of the previous one, so the var (patch_sequence) is updated every step.
    After the loop finishes we apply one more final 'layer_norm'
    """
    # TODO: run patch_sequence through every block in encoder_params['blocks'], then final layer norm.
    for block in encoder_params["blocks"]:
        patch_sequence = vision_encoder_block(patch_sequence, block, num_heads)
    return layer_norm(patch_sequence, encoder_params["final_ln_gamma"], encoder_params["final_ln_beta"], eps=1e-5)

# Step 30 - extract_patch_features
import torch

def extract_patch_features(encoder_output):
    """Drop the [CLS] token from a ViT encoder output of shape (B, num_patches+1, d_model)."""
    # TODO: drop the class token and return only patch feature tokens
    return encoder_output[:,1:,:]

# Step 31 - projector_first_layer
import torch

def projector_first_layer(patch_features, w1, b1):
    """
    Inputs:
        - patch_features: (N, D_vision) or (B, N, D_vision)
        - w1, b1: (D_vision, D_hidden), (D_hidden,)
    """
    # TODO: apply the first projector linear layer followed by GELU
    return gelu_activation(patch_features @ w1 + b1)

# Step 32 - projector_second_layer
import torch

def projector_second_layer(hidden, w2, b2):
    """Map hidden activations (N, D_hidden) into the language space (N, D_lang)."""
    # TODO: apply the second linear layer of the projector (no activation).
    return hidden @ w2 + b2

# Step 33 - vision_language_projector
import torch

def vision_language_projector(patch_features, params):
    """Map (N, D_vision) patch features to (N, D_lang) image tokens."""
    # TODO: chain the two projector layers using params 'w1','b1','w2','b2'.
    gelu_result = projector_first_layer(patch_features, params["w1"], params["b1"])
    return projector_second_layer(gelu_result, params["w2"], params["b2"])

# Step 34 - build_token_vocabulary
def build_token_vocabulary(texts, image_token='<image>', pad_token='<pad>'):
    # 1. Initialize the dictionary with special tokens
    vocab = {
        pad_token: 0,
        image_token: 1
    }
    
    # 2. Extract all distinct tokens from the input texts
    unique_tokens = set()
    for text in texts:
        # text.split() splits the string by whitespace
        for token in text.split():
            unique_tokens.add(token)
            
    # Remove special tokens from our set if they were in the text
    # so we don't accidentally assign them new IDs later
    unique_tokens.discard(pad_token)
    unique_tokens.discard(image_token)
    
    # 3. Sort the tokens to make the assignment deterministic
    sorted_tokens = sorted(list(unique_tokens))
    
    # 4. Assign sequential IDs starting from 2
    current_id = 2
    for token in sorted_tokens:
        vocab[token] = current_id
        current_id += 1
        
    return vocab

# Step 35 - encode_text_to_ids
def encode_text_to_ids(text, vocab):
    # TODO: split text on whitespace and map each token to its vocab id
    split_text = text.split()
    for i in range(len(split_text)):
        split_text[i] = vocab[split_text[i]]
    return split_text

# Step 36 - embed_token_ids
import torch

def embed_token_ids(token_ids, embedding_matrix):
    """Look up embedding vectors for each token id.

    Args:
        token_ids: Long tensor of shape (T,) with values in [0, V).
        embedding_matrix: Tensor of shape (V, D_lang).

    Returns:
        Tensor of shape (T, D_lang).
    """
    # TODO: select the row of embedding_matrix for each token id
    return embedding_matrix[token_ids,:]

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

