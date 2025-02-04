import torch
from transformers.models.llama.modeling_llama import *
from transformers.models.gpt_neox.modeling_gpt_neox import *
import numpy as np
import torch.nn as nn
import math
from typing import Optional, Tuple
import torch.nn.functional as F
from transformers.cache_utils import Cache
from flash_attn import flash_attn_func, flash_attn_varlen_func
from .selfextend_flash_attn import self_extend_flash_forward
from .selfextend_flash_attn_triton import self_extend_flash_forward_triton



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin) if not q is None else None
    k_embed = (k * cos) + (rotate_half(k) * sin) if not k is None else None
    return q_embed, k_embed

# def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, p_start=None, p_end=None, output_kemb=None,output_rokemb=None, seg = False, prefill_type = 1):
#     """Applies Rotary Position Embedding to the query and key tensors.

#     Modification::
#     new:
#     Modification::
#     1. the shape of cos/sin is [bs, current length, dim] or [bs, segment length, current length, dim] according to rope_forward_dpi_with_pi
#     2. segment length = q_end - q_staet, so we need p_start and p_end to get the absolute q position emb

#     old(dropped):
#     when use kvcache:
#         1. the shape of cos/sin is [bs, current length, dim] due to the modification of position ids and rope_forward_dpi_with_pi. Before, it is [bs, length of q, dim]. The length of q will be 1 in the generation stage or pre-fill stage if the prompt lenght is longer than training context length.
#         2. So we need to keep the shape of q when its length is 1
#     otherwise:
#         1. the shape of cos/sin/q/k is [bs, current length, dim] due to the modification of position ids and rope_forward_dpi_with_pi.
        

#     Args:
#         q (`torch.Tensor`): The query tensor.
#         k (`torch.Tensor`): The key tensor.
#         cos (`torch.Tensor`): The cosine part of the rotary embedding.
#         sin (`torch.Tensor`): The sine part of the rotary embedding.
#         position_ids (`torch.Tensor`, *optional*):
#             Deprecated and unused.
#         unsqueeze_dim (`int`, *optional*, defaults to 1):
#             The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
#             sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
#             that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
#             k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
#             cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
#             the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
#     Returns:
#         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#     """

#     # new:
#     cos = cos.unsqueeze(unsqueeze_dim) # to be [bs, 1, current length, dim] or [bs, 1, seg length, current length, dim]
#     sin = sin.unsqueeze(unsqueeze_dim)
#     # cos = cos[:,None,...]
#     # sin = sin[:,None,...]
#     # print("cos.shape, sin.shape ", cos.shape, sin.shape, q.shape, k.shape)
#     bs, head, q_length, dim = q.shape if q is not None else k.shape# bs, head, len, dim
#     if q_length == 1 and seg == False: 
#         # decode stage
#         if len(cos.shape) == 4:
#             # use pi for each token
#             q = (q * cos[:,:, -1: ,:]) + (rotate_half(q) * sin[:,:, -1: ,:]) if not q is None else None
#             k = (k * cos) + (rotate_half(k) * sin) if not k is None else None
#         else:
#             # use the same pi for all the tokens, i.e., the original implementation
#             q = (q * cos) + (rotate_half(q) * sin)
#             k = (k * cos) + (rotate_half(k) * sin) 
#     else:
#         # prefill stage
#         if len(cos.shape) == 4:
#             if prefill_type == 2:
#                 # for prefill type 2
#                 q = (q * cos[:,:,-q_length:,:]) + (rotate_half(q) * sin[:,:,-q_length:,:]) if not q is None else None
#                 k = (k * cos) + (rotate_half(k) * sin) if not k is None else None
#             else:
#                 # for prefill type 1
#                 # it is possible the computation of the part within the training context window during the pi stage
#                 q = (q * cos[:,:,:q_length,:]) + (rotate_half(q) * sin[:,:,:q_length,:]) if not q is None else None
#                 k = (k * cos) + (rotate_half(k) * sin) if not k is None else None
#         elif p_start is not None and p_end is not None:
#             tc = cos[:,:,0:p_end-p_start,p_start:p_end, :]
#             ts = sin[:,:,0:p_end-p_start,p_start:p_end, :]
#             # for q, we use the diagonal cos and sin value (the corresponding q positions)
#             # id1 = torch.arange(p_end-p_start)
#             # print("p_end ",p_end, p_start, id1)
#             # tc = tc[:,:,id1,id1,:].view(bs,1,-1,dim)
#             # ts = ts[:,:,id1,id1,:].view(bs,1,-1,dim)
#             tc = tc.diagonal(dim1=2, dim2=3).permute(0,1,3,2)
#             ts = ts.diagonal(dim1=2, dim2=3).permute(0,1,3,2)
#             # q_embed2 = torch.einsum('abcd,xxbce->abcd', q, tc_q) + torch.einsum('abcd,xxccd->abcd', rotate_half(q), tc_q)
#             # if p_start <65:
#             #     print("tc_q 2 ",tc_q.shape, ts_q.shape)
#             ro_q = rotate_half(q)
#             torch.mul(q, tc, out=q)
#             torch.mul(ro_q, ts, out=ro_q)
#             torch.add(q, ro_q, out=q)
#             # q = (q * tc) + (rotate_half(q) * ts) if not q is None else None
#             # print("q_embed2 - q_embed", (q_embed2 - q_embed).sum())

#             k = k[:,:,None,:,:]
#             tc = cos[:,:,0:p_end-p_start,:, :]
#             ts = sin[:,:,0:p_end-p_start,:, :]
       
#             torch.mul(k, tc, out=output_kemb)
#             k = rotate_half(k)
#             torch.mul(k, ts, out=output_rokemb)
#             torch.add(output_kemb, output_rokemb, out=output_kemb)
#             # k = (k * tc) + (rotate_half(k) * ts) if not k is None else None
#             return q, output_kemb
#         else:
#             # use the same pi for all the tokens, i.e., the original implementation
#             q = (q * cos) + (rotate_half(q) * sin)
#             k = (k * cos) + (rotate_half(k) * sin) 
#     # print("end, cos.shape, sin.shape ", cos.shape, sin.shape, q.shape, k.shape)
#     return q, k



def self_extend_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    group_size_1: Optional[float] = 8,
    group_size_2: Optional[float] = 1024,
    scale_base: Optional[int] = -1,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    bsz, q_len, _ = hidden_states.size()
    position_ids = position_ids[:,-q_len:] # to make it fit other dpi functions
    # print("attention_mask ",attention_mask.shape)
    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    test_max_dist_influence = False
    if test_max_dist_influence == True and q_len > 1:
        # test whether the maximum interval between the current token and the first token will destroy the representation
        original_dist_matrix = position_ids[0].unsqueeze(1) - position_ids[0].unsqueeze(0)
        se_group_dist_matrix = original_dist_matrix // group_size_1  + group_size_2 - group_size_2 // group_size_1
        seudo_max_position_emb = se_group_dist_matrix.max()
        seudo_max_position_emb = 8192
        seudo_position_ids = torch.arange(seudo_max_position_emb, device=position_ids.device).unsqueeze(0)
        seg_value_states = value_states[:, :, :seudo_max_position_emb, :]
        cos, sin = self.rotary_emb(seg_value_states, seudo_position_ids)
        seg_query_states, seg_key_states = apply_rotary_pos_emb(query_states[:, :, :seudo_max_position_emb, :], key_states[:, :, :seudo_max_position_emb, :], cos, sin)

        seg_key_states = repeat_kv(seg_key_states, self.num_key_value_groups)
        seg_attn_weights = torch.matmul(seg_query_states, seg_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        pad_len = len(position_ids[0]) - len(seudo_position_ids[0])
        seg_attn_weights = F.pad(seg_attn_weights,(0,pad_len),"constant",0)

        kv_seq_len = key_states.shape[-2]
        num_grouped_tokens = kv_seq_len - seudo_max_position_emb
        dpi_position_ids = []
        for i in range(num_grouped_tokens):
            dpi_position_ids.append(i)
            dpi_position_ids.append(i)
        dpi_position_ids += [i for i in range(num_grouped_tokens, seudo_max_position_emb)]
        dpi_position_ids = torch.tensor(dpi_position_ids).to(position_ids.device).unsqueeze(0)
        cos, sin = self.rotary_emb(value_states, dpi_position_ids)
        seg_query_states, seg_key_states = apply_rotary_pos_emb(query_states[:, :, :, :], key_states[:, :, :, :], cos, sin)
        seg_key_states = repeat_kv(seg_key_states, self.num_key_value_groups)
        seg_attn_weights2 = torch.matmul(seg_query_states, seg_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        print("seg_attn_weights ", seg_attn_weights.shape, seg_attn_weights2.shape)
        

    
    if scale_base > 0:
        scaled_query = query_states * ((position_ids + 1)[:, None, :, None].log() / np.log(scale_base)).clip(1).to(query_states.dtype) # log scale 
        #scaled_query = query_states * (((0.1*(((position_ids+1)[:, None, :, None]/scale_base).log())+1)**2).clip(1)).to(query_states.dtype) # Yarn scale 
    else:
        scaled_query = query_states
    
    past_key_value = getattr(self, "past_key_value", past_key_value)
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        cache_kwargs = {"cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    kv_seq_len = key_states.shape[-2]

    query_position = position_ids
    key_position = position_ids if q_len != 1 else torch.arange(kv_seq_len, dtype=position_ids.dtype).to(query_position.device).view(1, kv_seq_len) # only consider bsz=1 for now.

    if test_max_dist_influence == True and q_len == 1:
        seudo_max_position_emb = 8192 # 6035 # 3025
        kv_seq_len = key_states.shape[-2]
        num_grouped_tokens = kv_seq_len - seudo_max_position_emb
        dpi_position_ids = []
        for i in range(num_grouped_tokens):
            dpi_position_ids.append(i)
            dpi_position_ids.append(i)
        dpi_position_ids += [i for i in range(num_grouped_tokens, seudo_max_position_emb)]
        dpi_position_ids = torch.tensor(dpi_position_ids).to(position_ids.device).unsqueeze(0)
        cos, sin = self.rotary_emb(value_states, dpi_position_ids)
        seg_query_states, seg_key_states = apply_rotary_pos_emb(query_states[:, :, :, :], key_states[:, :, :, :], cos, sin)
        seg_key_states = repeat_kv(seg_key_states, self.num_key_value_groups)
        seg_attn_weights2 = torch.matmul(seg_query_states, seg_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    neighbor_q_cos, neighbor_q_sin = self.rotary_emb(value_states, query_position)
    neighbor_k_cos, neighbor_k_sin = self.rotary_emb(value_states, key_position)


    _re_group_size_2 = 0 if query_position.max() < group_size_2 else group_size_2 # in case that, the smallest q position, g2-g2//g1 exceed the max position
    group_query_position = query_position // group_size_1 + _re_group_size_2 - _re_group_size_2 / group_size_1
    group_key_position = key_position // group_size_1

    group_q_cos, group_q_sin = self.rotary_emb(value_states, group_query_position)
    group_k_cos, group_k_sin = self.rotary_emb(value_states, group_key_position)



    neighbor_query_states, _ = apply_rotary_pos_emb(scaled_query, None, neighbor_q_cos, neighbor_q_sin, None)
    _, neighbor_key_states = apply_rotary_pos_emb(None, key_states, neighbor_k_cos, neighbor_k_sin, None)
    group_query_states, _ = apply_rotary_pos_emb(scaled_query, None, group_q_cos, group_q_sin, None)
    _, group_key_states = apply_rotary_pos_emb(None, key_states, group_k_cos, group_k_sin, None)



    neighbor_key_states = repeat_kv(neighbor_key_states, self.num_key_value_groups)
    group_key_states = repeat_kv(group_key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)



    neighbor_attn_weights = torch.matmul(neighbor_query_states, neighbor_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    group_attn_weights = torch.matmul(group_query_states, group_key_states.transpose(2, 3)) / math.sqrt(self.head_dim) 


    # print("attention_mask ",attention_mask.shape, cache_position, key_states.shape)
    # # attention_mask  torch.Size([1, 1, 1, 11048]) tensor([11047], device='cuda:0') torch.Size([1, 8, 11048, 128])
    # if q_len == 1:
    #     import os
    #     os.exit(0)
            
    if attention_mask is not None:  # no matter the length, we just slice it
        # if cache_position is not None:
        #     causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
        # else:
        #     causal_mask = attention_mask
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]] # modified to fit the current LLama implementation by dpi
        group_attn_weights = group_attn_weights + causal_mask
        
        if test_max_dist_influence == True and q_len > 1:
            print("attention_mask ", attention_mask.shape, group_attn_weights.shape)
            seg_attn_weights2 = seg_attn_weights2 + attention_mask
            seg_attn_weights = seg_attn_weights + attention_mask[:, :, :seg_attn_weights.shape[2], : key_states.shape[-2]]

        if test_max_dist_influence == True and q_len == 1:
            seg_attn_weights2 = seg_attn_weights2 + attention_mask

    if q_len == 1:
        neighbor_attention_mask = torch.zeros((q_len, kv_seq_len), device=neighbor_attn_weights.device)
        neighbor_attention_mask[:, -group_size_2:] = 1
    elif q_len == kv_seq_len:
        neighbor_attention_mask = torch.ones((q_len, kv_seq_len), device=neighbor_attn_weights.device)
        neighbor_attention_mask = torch.tril(neighbor_attention_mask)
        if q_len-group_size_2 > 0:
            group_attention_mask =  torch.tril(torch.ones((q_len-group_size_2, kv_seq_len-group_size_2), device=group_attn_weights.device))
            neighbor_attention_mask[group_size_2:, :-group_size_2] -= group_attention_mask
    else:
        raise ValueError("q_len should be 1 or seq_len.")
    
    neighbor_attention_mask = neighbor_attention_mask.bool()
    attn_weights = torch.where(neighbor_attention_mask, neighbor_attn_weights, group_attn_weights)
    if test_max_dist_influence == True and q_len > 1:
        attn_weights = seg_attn_weights2
        attn_weights[:, :, :seg_attn_weights.shape[2], :] = seg_attn_weights

    if test_max_dist_influence == True and q_len == 1:
        attn_weights = seg_attn_weights2 
        
        
        
    
    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )
    

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def flash_self_extend_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    group_size_1: Optional[float] = 8,
    group_size_2: Optional[float] = 1024,
    scale_base: Optional[int] = -1,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
        Require updating tansformers to >= 4.38.2, flash_attn >= 2.5.6
        a. Only support causal mask.
        b. Don't support atttention_mask.
        c. Never test it with batch size > 1.
        d. Only support q_len = 1 or q_len = seq_len.
    """
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
        attention_mask = kwargs.pop("padding_mask")

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if scale_base > 0:
        scaled_query = query_states * ((position_ids + 1)[:, None, :, None].log() / np.log(scale_base)).clip(1).to(query_states.dtype) # log scale 
        #scaled_query = query_states * (((0.1*(((position_ids+1)[:, None, :, None]/scale_base).log())+1)**2).clip(1)).to(query_states.dtype) # Yarn scale 
    else:
        scaled_query = query_states
    
    past_key_value = getattr(self, "past_key_value", past_key_value)
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        cache_kwargs = {"cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    kv_seq_len = key_states.shape[-2]

    query_position = position_ids
    # only consider bsz=1 for now. 
    key_position = position_ids if q_len != 1 else torch.arange(kv_seq_len, dtype=position_ids.dtype).to(query_position.device).view(1, kv_seq_len) 
    attn_dropout = self.config.attention_dropout if self.training else 0.0
    if q_len == 1:
        # We implement the case q_len == 1 separately, by manipulating positions.
        # for our flash implementation doesnot work for  decoding stage at the releasing time.

        neighbor_key_position = position_ids[:, -1] - key_position
        _re_group_size_2 = 0 if position_ids.max() < group_size_2 else group_size_2
        group_key_position = position_ids[:, -1]//group_size_1 - key_position//group_size_1 + (_re_group_size_2 - _re_group_size_2//group_size_1)
        decode_key_position = torch.cat([group_key_position[:, :-group_size_2], neighbor_key_position[:,-group_size_2:]], dim=1)
        
        decode_k_cos, decode_k_sin = self.rotary_emb(value_states, decode_key_position)
        #import pdb; pdb.set_trace()
        #neighbor_query_states, _ = apply_rotary_pos_emb(scaled_query, None, cos, sin, query_position_ids) 
        decode_query_states = scaled_query.transpose(1,2).contiguous() # position 0: cos 0 = 1, sin 0 = 0
        _, decode_key_states = apply_rotary_pos_emb(None, key_states, decode_k_cos, -decode_k_sin, decode_key_position) 

        decode_key_states = repeat_kv(decode_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        decode_value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        
        attn_output = flash_attn_func(decode_query_states,
                                      decode_key_states,
                                      decode_value_states,
                                      attn_dropout, 
                                      softmax_scale=None, 
                                      causal=True)
    elif q_len == kv_seq_len:
        # set correct position_ids & apply RoPE.
        neighbor_q_cos, neighbor_q_sin = self.rotary_emb(value_states, query_position)
        neighbor_k_cos, neighbor_k_sin = self.rotary_emb(value_states, key_position)

        _re_group_size_2 = 0 if query_position.max() < group_size_2 else group_size_2 # in case that, the smallest q position, g2-g2//g1 exceed the max position
        group_query_position = query_position // group_size_1 + _re_group_size_2 - _re_group_size_2 / group_size_1
        group_key_position = key_position // group_size_1

        group_q_cos, group_q_sin = self.rotary_emb(value_states, group_query_position)
        group_k_cos, group_k_sin = self.rotary_emb(value_states, group_key_position)

        neighbor_query_states, _ = apply_rotary_pos_emb(scaled_query, None, neighbor_q_cos, neighbor_q_sin, None)
        _, neighbor_key_states = apply_rotary_pos_emb(None, key_states, neighbor_k_cos, neighbor_k_sin, None)
        group_query_states, _ = apply_rotary_pos_emb(scaled_query, None, group_q_cos, group_q_sin, None)
        _, group_key_states = apply_rotary_pos_emb(None, key_states, group_k_cos, group_k_sin, None)
        

        neighbor_query_states = neighbor_query_states.transpose(1, 2).contiguous()
        neighbor_key_states = repeat_kv(neighbor_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        group_query_states = group_query_states.transpose(1, 2).contiguous()
        group_key_states = repeat_kv(group_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()

        attn_output = self_extend_flash_forward(self,
                                                query_position,
                                                group_size_2,
                                                neighbor_query_states,
                                                neighbor_key_states,
                                                group_query_states,
                                                group_key_states,
                                                value_states,
                                                attention_mask,
                                                bsz,
                                                q_len,
                                                kv_seq_len,
                                                attn_dropout,
                                            )
    else:
        raise ValueError("q_len should be 1 or seq_len.")
    
    attn_output = attn_output.contiguous()
    attn_output = attn_output.view(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value

 



def flash_self_extend_forward_triton(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    group_size_1: Optional[float] = 8,
    group_size_2: Optional[float] = 1024,
    scale_base: Optional[int] = -1,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
        Require updating tansformers to >= 4.38.2, flash_attn >= 2.5.6
        a. Only support causal mask.
        b. Don't support atttention_mask.
        c. Never test it with batch size > 1.
        d. Only support q_len = 1 or q_len = seq_len.
    """
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
        attention_mask = kwargs.pop("padding_mask")

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if scale_base > 0:
        scaled_query = query_states * ((position_ids + 1)[:, None, :, None].log() / np.log(scale_base)).clip(1).to(query_states.dtype) # log scale 
        #scaled_query = query_states * (((0.1*(((position_ids+1)[:, None, :, None]/scale_base).log())+1)**2).clip(1)).to(query_states.dtype) # Yarn scale 
    else:
        scaled_query = query_states
    
    past_key_value = getattr(self, "past_key_value", past_key_value)
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        cache_kwargs = {"cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    kv_seq_len = key_states.shape[-2]

    query_position = position_ids
    # only consider bsz=1 for now. 
    key_position = position_ids if q_len != 1 else torch.arange(kv_seq_len, dtype=position_ids.dtype).to(query_position.device).view(1, kv_seq_len) 
    attn_dropout = self.config.attention_dropout if self.training else 0.0


    # set correct position_ids & apply RoPE.
    neighbor_q_cos, neighbor_q_sin = self.rotary_emb(value_states, query_position, seq_len=None)
    neighbor_k_cos, neighbor_k_sin = self.rotary_emb(value_states, key_position, seq_len=None)

    _re_group_size_2 = 0 if query_position.max() < group_size_2 else group_size_2 # in case that, the smallest q position, g2-g2//g1 exceed the max position
    group_query_position = query_position // group_size_1 + _re_group_size_2 - _re_group_size_2 / group_size_1
    group_key_position = key_position // group_size_1

    group_q_cos, group_q_sin = self.rotary_emb(value_states, group_query_position, seq_len=None)
    group_k_cos, group_k_sin = self.rotary_emb(value_states, group_key_position, seq_len=None)

    neighbor_query_states, _ = apply_rotary_pos_emb(scaled_query, None, neighbor_q_cos, neighbor_q_sin, None)
    _, neighbor_key_states = apply_rotary_pos_emb(None, key_states, neighbor_k_cos, neighbor_k_sin, None)
    group_query_states, _ = apply_rotary_pos_emb(scaled_query, None, group_q_cos, group_q_sin, None)
    _, group_key_states = apply_rotary_pos_emb(None, key_states, group_k_cos, group_k_sin, None)
    
    attn_output = self_extend_flash_forward_triton(self,
                                            query_position,
                                            group_size_2,
                                            neighbor_query_states,
                                            neighbor_key_states,
                                            group_query_states,
                                            group_key_states,
                                            value_states,
                                            attention_mask,
                                            bsz,
                                            q_len,
                                            kv_seq_len,
                                            attn_dropout,
                                        )

    
    attn_output = attn_output.contiguous()
    attn_output = attn_output.view(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value