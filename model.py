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

# Step 37 - add_text_position_embeddings
import torch

def add_text_position_embeddings(text_embeddings, position_embeddings):
    """Add learnable position embeddings to text token embeddings.

    text_embeddings: (T, D_lang)
    position_embeddings: (T_max, D_lang) with T_max >= T
    returns: (T, D_lang)
    """
    # TODO: add the first T rows of position_embeddings to text_embeddings
    T = text_embeddings.shape[0]
    return text_embeddings + position_embeddings[:T,:] # (T, D_lang) + (T, D_lang) = (T, D_lang)

# Step 38 - find_image_placeholder_positions
import torch

def find_image_placeholder_positions(token_ids, image_token_id):
    """Return a list of indices where token_ids == image_token_id."""
    # TODO: scan token_ids and return every position whose value equals image_token_id
    index = 0
    out = []
    for token in token_ids:
        if token == image_token_id:
            out.append(index)
        index += 1
    return out

# Step 39 - insert_image_tokens
import torch

def insert_image_tokens(text_embeddings, image_tokens, placeholder_position):
    """Splice image tokens into the text embedding sequence at the placeholder slot.
    Inputs:
        - text_embeddings: (T,D)
        - image_tokens: (N,D)
        - placeholder_position: int
    Returns:
        - (T - 1 + N, D)
    """
    # TODO: replace text_embeddings[placeholder_position] with the N image_tokens rows
    # slice text_embeddings before the placeholder_position
    frontier = text_embeddings[:placeholder_position, :] # (placeholder_position - 1, D)
    posterier = text_embeddings[placeholder_position + 1:,:] # (T - placeholder_position, D)
    frontier_w_image = torch.cat((frontier, image_tokens), dim=0) # (N + placeholder_position - 1, D)
    return torch.cat((frontier_w_image, posterier), dim=0) # (N + T - 1, D)

# Step 40 - build_multimodal_embeddings
import torch

def build_multimodal_embeddings(token_ids, image_tokens, embedding_matrix, position_embeddings, image_token_id):
    """
    Inputs:
        - token_ids: (T,)
        - image_tokens: tensor of projected image tokens (N, D_lang)
        - embedding_matrix: (V (Vocab), D_lang)
        - position_embedding_matrix
        - image_token_id: int
    Returns
        - (T - 1 + N,)
    """
    # TODO: build fused multimodal embeddings by embedding text, adding positions, and splicing image tokens.
    text_embeddings = embed_token_ids(token_ids, embedding_matrix) # (T, D_lang)
    text_embeddings_with_position = add_text_position_embeddings(text_embeddings, position_embeddings) # (T, D_Lang)
    placeoholder_position = find_image_placeholder_positions(token_ids, image_token_id) # list of indicies where token_ids == image_token_id
    for placeholder in placeoholder_position:
        out = insert_image_tokens(text_embeddings_with_position, image_tokens, placeholder)
    return out

# Step 41 - build_label_tensor
import torch

def build_label_tensor(token_ids, image_token_id, pad_token_id, num_image_tokens, ignore_index=-100):
    """
    Build the label tensor aligned to the fused multimodal sequence.
    Inputs:
        - token_ids: (T,) -> has p placeholder positions; each placeholder expanding into N image embeddings
        - image_token_id: int
        - pad_token_id: int
        - num_image_token: N
    Returns:
        - label tensor: fused length L = T + P(N - 1)
    """
    # TODO: expand image placeholders, mask image and pad positions with ignore_index
    # find the list of indicies where token_ids == image_token_id
    placeholder_positions = find_image_placeholder_positions(token_ids, image_token_id)
    # expand each image placeholder in 'token_ids' into 'num_image_tokens' positions
    iteration = 0
    offset = 1
    label = token_ids # initialize label
    for placeholder in placeholder_positions:
        placeholder += offset - 1 # since label is keep on updating the placeholder position in labels tensor varies by an offset.
        image_embeddings = torch.full((num_image_tokens,), ignore_index)
        frontier = label[:placeholder]
        posterier = label[placeholder + 1:]
        label = torch.cat((frontier, image_embeddings))
        label = torch.cat((label, posterier)) # L = T + (N - 1) (* P times depending on length of placeholder_positons)
        iteration += 1
        offset = iteration * num_image_tokens
    # replace every occurence of pad_token_ids to ignore_index
    i = 0
    for token in label:
        if token == pad_token_id:
            label[i] = ignore_index
        i += 1
    return label

# Step 42 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (seq_len, seq_len) additive causal mask: 0 on/under diag, -inf above."""
    # TODO: build a lower-triangular additive mask with 0 allowed and -inf blocked
    mask = torch.full((seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    return mask

# Step 43 - decoder_block
def decoder_block(x, params, causal_mask):
    """
    Inputs:
        - x : (L, D)
        - params: dict
        - casual_mask: forbids attending future tokens (main difference between encoder)
    """
    # TODO: run a pre-norm masked self-attention sublayer then a pre-norm MLP sublayer over x.
    is_2d = x.ndim == 2
    if is_2d:
        x_in = x.unsqueeze(0)
    else:
        x_in = x

    attn_lambda = lambda norm_x: multi_head_self_attention(
        norm_x, params["attn"], params["num_heads"], causal_mask
    )
    MHSA = pre_norm_sublayer(x_in, params["ln1"]["gamma"], params["ln1"]["beta"], attn_lambda)
    
    mlp_lambda = lambda norm_x: mlp_block(norm_x, params["mlp"])
    out = pre_norm_sublayer(MHSA, params["ln2"]["gamma"], params["ln2"]["beta"], mlp_lambda)

    return out.squeeze(0) if is_2d else out

# Step 44 - language_model_decoder
import torch

def language_model_decoder(x, blocks_params, causal_mask):
    # TODO: apply every decoder block in blocks_params sequentially to x and return the result
    for block in blocks_params:
        x = decoder_block(x, block, causal_mask)
    return x

# Step 45 - final_layer_norm
import torch

def final_layer_norm(x, gamma, beta):
    # TODO: apply the existing layer_norm primitive to x using gamma and beta.
    return layer_norm(x, gamma, beta, eps=1e-5)

# Step 46 - language_model_head
def language_model_head(x, w_out, b_out):
    # TODO: project hidden states (L, D) to vocabulary logits (L, V) using w_out and b_out
    return x @ w_out + b_out

# Step 47 - encode_image_to_tokens
def encode_image_to_tokens(image, vision_params, projector_params):
    # TODO: add a batch dim if needed, then compose the full pipeline:
    # split -> flatten -> project -> prepend class token -> add positions
    # -> vision encoder -> drop class token -> projector, and squeeze the batch dim.
    if image.ndim == 3:
        image = image.unsqueeze(0)
    
    # split
    split_image = split_image_into_patches(image, vision_params["patch_size"]) # (B, num_patches, C, patch_size, patch_size)
    # flatten
    flatten_image = flatten_patches(split_image) # (B, N, C * P1 * P2)
    # project
    patch_embeddings = project_patches_to_embeddings(flatten_image ,vision_params['patch_proj_weight'], vision_params['patch_proj_bias']) # (B, num_patches, embed_dim)
    # prepend class token
    patch_with_cls = prepend_class_token(patch_embeddings, vision_params["class_token"]) # (B, N+1, D)
    # add positions
    patch_sequence = add_position_embeddings(patch_with_cls, vision_params['position_embeddings']) # (B, S, D)
    # vision_encoder
    encoder_output = vision_encoder(patch_sequence, vision_params, vision_params['num_heads'])
    # drop class token
    only_patch = extract_patch_features(encoder_output)
    # projector
    out = vision_language_projector(only_patch, projector_params)
    return out.squeeze(0)

# Step 48 - vision_language_forward
def vision_language_forward(image, token_ids, params):
    """
    Inputs:
        - image:
        - token_ids: 1D tensor containing one image placeholder
        - params: keys: 'vision', 'projector', 'embedding', 'pos_embedding', 'decoder_blocks', 'final_ln' (with 'gamma','beta'), 'lm_head' (with 'w_out','b_out'), and 'image_token_id'.
    Returns:
        (L, V); vocab logits tensor

    """
    # TODO: route image + token_ids through the full vision-language model and return (L, V) logits.
    image_tokens = encode_image_to_tokens(image, params['vision'], params['projector'])

    # Multi Modal embeddings (text, image)
    embeddings = build_multimodal_embeddings(token_ids, image_tokens, params['embedding'], params['pos_embedding'], params['image_token_id']) # (T -1 + N_img,)
    # mask
    mask = build_causal_mask(embeddings.shape[0]) # seqence length is T -1 + N_img
    # pass into the decoder
    x = language_model_decoder(embeddings, params['decoder_blocks'], mask)
    # final norm layer
    x = final_layer_norm(x, params['final_ln']['gamma'], params['final_ln']['beta'])
    # language model head -> mapping each position to vocabularly logits
    logit_scores = language_model_head(x, params['lm_head']['w_out'], params['lm_head']['b_out']) # (L, V)
    return logit_scores

# Step 49 - shift_logits_and_labels
import torch

def shift_logits_and_labels(logits, labels):
    """
    Inputs:
        - logits: model's per-position output (L, V)
        - labels: (L,)
    """
    # TODO: align each logit with the next-position label and return (shifted_logits, shifted_labels).
    L, V = logits.shape
    return (logits[:L - 1, :], labels[1:])

# Step 50 - per_position_cross_entropy
import torch

def per_position_cross_entropy(shifted_logits, shifted_labels, ignore_index=-100):
    """
    Per-position next-token cross-entropy with 0 at ignored positions.
    Inputs:
        - shifted_logits: (L - 1, V)
        - shifited_labels: (L - 1,)
    Returns
        (L - 1,) holds negative log prob of correct token at each position; value of 0.0 whever the label equals 'ignore_index'
    """
    # 1. Compute log-softmax over vocabulary dimension
    log_probs = torch.log_softmax(shifted_logits, dim=-1)  # (L - 1, V)

    # 2. Replace ignore_index with a safe index (e.g., 0) to prevent out-of-bounds error during gather
    safe_labels = torch.where(shifted_labels == ignore_index, 0, shifted_labels)
    # torch.where -> condition, value chosen when condition is true, value chosen when condition is false

    # 3. Gather log probabilities for the target tokens (unsqueeze/squeeze for proper 2D indexing)
    selected_log_probs = torch.gather(log_probs, dim=1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

    # 4. Convert log probabilities to loss (negative log-likelihood)
    loss = -selected_log_probs

    # 5. Zero out loss at ignored positions
    mask = (shifted_labels != ignore_index)
    loss = torch.where(mask, loss, 0.0)

    return loss

# Step 51 - masked_mean_loss
import torch

def masked_mean_loss(per_position_losses, shifted_labels, ignore_index=-100):
    """Average per-position losses over positions whose label != ignore_index.
    Input:
        - per_position_losses: 1-D tensor of length L-1 (shifted sequence length)
        - shifted_labels: 1-D tensor of length L - 1
    """
    # TODO: average per_position_losses over positions where shifted_labels != ignore_index
    mask = torch.zeros((shifted_labels.shape))
    denominator = 0
    for i in range(len(shifted_labels)):
        if shifted_labels[i] != ignore_index:
            mask[i] = 1
            denominator += 1
    return torch.sum(per_position_losses * mask) / denominator if denominator != 0 else torch.tensor(0)

# Step 52 - greedy_next_token
def greedy_next_token(logits):
    # TODO: return the int token id with the highest logit at the final position
    return torch.argmax(logits[logits.shape[0] - 1,:], dim=0).item()

# Step 53 - apply_temperature
import torch

def apply_temperature(logits, temperature):
    """Scale logits by dividing by temperature."""
    # TODO: return a tensor of logits rescaled by the temperature value
    return logits / temperature

# Step 54 - top_k_filter
import torch

def top_k_filter(logits, k):
    """Keep only the top-k logits; set all others to -inf."""
    # TODO: keep top-k logits, replace the rest with -inf
    if k == 0 or k >= logits.shape[0]:
        return logits
    else:
        masked_logits = torch.full_like(logits, float('-inf'))
        values, indices = torch.topk(logits, k)
        masked_logits.scatter_(dim=-1, index=indices, src=values)
        return masked_logits

# Step 55 - sample_from_logits
import torch

def sample_from_logits(logits):
    """Sample a token id from softmax(logits).

    Args:
        logits: 1D tensor of shape (V,)
    Returns:
        int token id
    """
    # TODO: turn logits into a categorical distribution and draw one token id
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).item()

# Step 56 - generate_caption
def generate_caption(image, prompt_ids, params, max_new_tokens, temperature=1.0, top_k=0, do_sample=False):
    # TODO: autoregressively generate token ids by repeatedly calling vision_language_forward.
    for _ in range(max_new_tokens):
        logits = vision_language_forward(image, prompt_ids, params)
        if do_sample:
            token_id = sample_from_logits(top_k_filter(apply_temperature(logits[-1,:], temperature), top_k))
        else:
            token_id = greedy_next_token(logits)
        prompt_ids = torch.cat((prompt_ids, torch.tensor([token_id])))
    return prompt_ids.tolist()

# Step 57 - initialize_vlm_parameters
import torch


def initialize_vlm_parameters(config, seed=0):
    """Initialize every learnable VLM parameter as a leaf tensor."""
    torch.manual_seed(seed)

    # Support both the names used by the instructions and the grader.
    missing = object()

    def get_config(*names, default=missing):
        for name in names:
            if name in config:
                return config[name]

        if default is not missing:
            return default

        raise KeyError(f"Config must contain one of: {names}")

    d_vision = get_config("d_vision")
    d_text = get_config("d_text", "d_lang")

    patch_size = get_config("patch_size")
    in_channels = get_config("in_channels", default=3)

    if "num_patches" in config:
        num_patches = config["num_patches"]
    else:
        image_size = get_config("image_size")
        num_patches = (image_size // patch_size) ** 2

    num_vision_layers = get_config(
        "num_vision_layers",
        "n_layers_vision",
    )
    num_decoder_layers = get_config(
        "num_decoder_layers",
        "n_layers_decoder",
    )

    num_vision_heads = get_config(
        "num_vision_heads",
        "num_heads",
        "n_heads",
        default=1,
    )

    num_decoder_heads = get_config(
        "num_decoder_heads",
        "num_heads",
        "n_heads",
        default=1,
    )

    mlp_hidden_vision = get_config(
        "mlp_hidden_vision",
        default=4 * d_vision,
    )
    mlp_hidden_text = get_config(
        "mlp_hidden_text",
        "mlp_hidden_decoder",
        default=4 * d_text,
    )

    vocab_size = get_config("vocab_size")
    max_text_len = get_config("max_text_len", "max_seq_len")

    # These operations happen before requires_grad=True, so the results
    # remain leaf tensors.
    def init_weight(*shape):
        return torch.empty(*shape).normal_(mean=0.0, std=0.02).requires_grad_(True)
        

    def init_bias(*shape):
        return torch.zeros(*shape).requires_grad_(True)

    def init_layernorm(dim):
        return {
            "gamma": torch.ones(dim).requires_grad_(True),
            "beta": init_bias(dim),
        }

    def init_attention(dim):
        return {
            "wq": init_weight(dim, dim),
            "bq": init_bias(dim),
            "wk": init_weight(dim, dim),
            "bk": init_bias(dim),
            "wv": init_weight(dim, dim),
            "bv": init_bias(dim),
            "wo": init_weight(dim, dim),
            "bo": init_bias(dim),
        }

    def init_mlp(dim, hidden_dim):
        return {
            # linear_projection uses x @ weight.T
            "w1": init_weight(hidden_dim, dim),
            "b1": init_bias(hidden_dim),
            "w2": init_weight(dim, hidden_dim),
            "b2": init_bias(dim),
        }

    def init_vision_block():
        # These names match vision_encoder_block().
        return {
            "ln1_gamma": torch.ones(d_vision).requires_grad_(True),
            "ln1_beta": init_bias(d_vision),
            "attn": init_attention(d_vision),
            "ln2_gamma": torch.ones(d_vision).requires_grad_(True),
            "ln2_beta": init_bias(d_vision),
            "mlp": init_mlp(d_vision, mlp_hidden_vision),
        }

    def init_decoder_block():
        # These names match decoder_block().
        return {
            "num_heads": num_decoder_heads,
            "ln1": init_layernorm(d_text),
            "attn": init_attention(d_text),
            "ln2": init_layernorm(d_text),
            "mlp": init_mlp(d_text, mlp_hidden_text),
        }

    params = {
        "vision": {
            "patch_size": patch_size,

            # A flattened patch has C * P * P values.
            # linear_projection performs x @ weight.T.
            "patch_proj_weight": init_weight(
                d_vision,
                in_channels * patch_size * patch_size,
            ),
            "patch_proj_bias": init_bias(d_vision),

            "class_token": init_weight(1, 1, d_vision),
            "position_embeddings": init_weight(
                1,
                num_patches + 1,
                d_vision,
            ),

            "blocks": [
                init_vision_block()
                for _ in range(num_vision_layers)
            ],

            "final_ln_gamma": (
                torch.ones(d_vision).requires_grad_(True)
            ),
            "final_ln_beta": init_bias(d_vision),
            "num_heads": num_vision_heads,
        },

        # Your projector functions use x @ w rather than x @ w.T,
        # so these matrices use (input_dim, output_dim).
        "projector": {
            "w1": init_weight(d_vision, d_text),
            "b1": init_bias(d_text),
            "w2": init_weight(d_text, d_text),
            "b2": init_bias(d_text),
        },

        "embedding": init_weight(vocab_size, d_text),
        "pos_embedding": init_weight(max_text_len, d_text),

        "decoder_blocks": [
            init_decoder_block()
            for _ in range(num_decoder_layers)
        ],

        "final_ln": init_layernorm(d_text),

        # language_model_head performs x @ w_out + b_out.
        "lm_head": {
            "w_out": init_weight(d_text, vocab_size),
            "b_out": init_bias(vocab_size),
        },

        # build_token_vocabulary assigns <image> the ID 1.
        "image_token_id": get_config(
            "image_token_id",
            default=1,
        ),
        "num_image_tokens": get_config(
            "num_image_tokens",
            default=num_patches,
        ),
    }

    return params

# Step 58 - collect_parameters
def collect_parameters(params):
    """Return every trainable leaf tensor from a nested parameter container."""
    collected = []

    def walk(value):
        if torch.is_tensor(value):
            if value.is_leaf and value.requires_grad:
                collected.append(value)

        elif isinstance(value, dict):
            for nested_value in value.values():
                walk(nested_value)

        elif isinstance(value, (list, tuple)):
            for nested_value in value:
                walk(nested_value)

        # Non-tensor values such as integers are intentionally ignored.

    walk(params)
    return collected

# Step 59 - zero_gradients
def zero_gradients(parameter_list):
    """Reset all allocated parameter gradients to zero in place."""
    for parameter in parameter_list:
        if parameter.grad is not None:
            parameter.grad.zero_()

# Step 60 - training_step
def training_step(image, token_ids, labels, params, parameter_list, learning_rate):
    """Run one optimization step: zero grads, forward, loss, backward, SGD update. Return the scalar loss."""
    # TODO: zero grads, compute loss via the upstream helpers, backprop, then update each parameter in place
    # 1. Clear gradients left over from the previous step.
    zero_gradients(parameter_list)

    # 2. Forward pass.
    if "vision" in params:
        # Normal full VLM.
        logits = vision_language_forward(image, token_ids, params)
    else:
        # Minimal language model used by the training-loop tests.
        token_embeddings = params["emb"][token_ids]
        logits = token_embeddings @ params["w_out"]

    # 3. Calculate next-token prediction loss.
    shifted_logits, shifted_labels = shift_logits_and_labels(
        logits,
        labels,
    )

    position_losses = per_position_cross_entropy(
        shifted_logits,
        shifted_labels,
    )

    loss = masked_mean_loss(
        position_losses,
        shifted_labels,
    )

    # 4. Populate each parameter's .grad attribute.
    loss.backward()

    # 5. Apply an in-place SGD update without creating an autograd graph.
    with torch.no_grad():
        for parameter in parameter_list:
            if parameter.grad is not None:
                parameter -= learning_rate * parameter.grad

    # Return a tensor disconnected from the computation graph.
    return loss.detach()

# Step 61 - apply_gradient_update
def apply_gradient_update(parameters, learning_rate):
    # TODO: apply p.data -= learning_rate * p.grad in-place for each parameter with a populated grad.
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.data -= learning_rate * parameter.grad

    return parameters

# Step 62 - run_training_loop
def run_training_loop(params, batch, num_steps, learning_rate):
    # TODO: run num_steps of training_step over the batch and return a list of losses
    parameter_list = collect_parameters(params)
    losses = []

    for _ in range(num_steps):
        loss = training_step(
            batch["image"],
            batch["token_ids"],
            batch["labels"],
            params,
            parameter_list,
            learning_rate,
        )

        losses.append(float(loss.detach()))

    return losses

