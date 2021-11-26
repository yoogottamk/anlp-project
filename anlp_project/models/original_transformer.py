import torch
from torch import nn

from anlp_project.models import utils


def CopyParams(base_layer, target_layer, decoder=False):
    if target_layer.self_attn.qkv_same_dim:
        target_layer.self_attn.in_proj_weight = base_layer.self_attn.in_proj_weight
    else:
        target_layer.self_attn.k_proj_weight = base_layer.self_attn.k_proj_weight
        target_layer.self_attn.v_proj_weight = base_layer.self_attn.v_proj_weight
        target_layer.self_attn.q_proj_weight = base_layer.self_attn.q_proj_weight
    if base_layer.self_attn.bias_k is not None:
        target_layer.self_attn.bias_k = base_layer.self_attn.bias_k
    if base_layer.self_attn.bias_v is not None:
        target_layer.self_attn.bias_v = base_layer.self_attn.bias_v
    if target_layer.self_attn.in_proj_bias is not None:
        target_layer.self_attn.in_proj_bias = base_layer.self_attn.in_proj_bias
    target_layer.self_attn.out_proj = base_layer.self_attn.out_proj
    target_layer.fc1 = base_layer.fc1
    target_layer.fc2 = base_layer.fc2
    target_layer.self_attn_layer_norm = base_layer.self_attn_layer_norm
    target_layer.final_layer_norm = base_layer.final_layer_norm
    if decoder:
        if target_layer.encoder_attn.qkv_same_dim:
            target_layer.encoder_attn.in_proj_weight = (
                base_layer.encoder_attn.in_proj_weight
            )
        else:
            target_layer.encoder_attn.k_proj_weight = (
                base_layer.encoder_attn.k_proj_weight
            )
            target_layer.encoder_attn.v_proj_weight = (
                base_layer.encoder_attn.v_proj_weight
            )
            target_layer.encoder_attn.q_proj_weight = (
                base_layer.encoder_attn.q_proj_weight
            )
        target_layer.encoder_attn.out_proj = base_layer.encoder_attn.out_proj
        if base_layer.encoder_attn.bias_k is not None:
            target_layer.encoder_attn.bias_k = base_layer.encoder_attn.bias_k
        if base_layer.encoder_attn.bias_v is not None:
            target_layer.encoder_attn.bias_v = base_layer.encoder_attn.bias_v
        if target_layer.encoder_attn.in_proj_bias is not None:
            target_layer.encoder_attn.in_proj_bias = base_layer.self_attn.in_proj_bias
        target_layer.encoder_attn_layer_norm = base_layer.encoder_attn_layer_norm
    return target_layer


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
