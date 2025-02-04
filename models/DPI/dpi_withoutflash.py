from types import MethodType
from functools import partial
import models.DPI.patches as P
import models.SE.SelfExtend as SE
import models.DCA.flash_decoding_chunkllama_dpi as CLflash_dec
import torch
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoConfig

from transformers.models.llama.modeling_llama import *
from transformers.models.gpt_neox.modeling_gpt_neox import *
import numpy as np
import torch.nn as nn
import math
from typing import Optional, Tuple
import torch.nn.functional as F
from transformers.cache_utils import Cache
# from flash_attn import flash_attn_func, flash_attn_varlen_func
llama_token =""
def attn_logits_analysis_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

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

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
        attn_logits = attn_weights

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

    attn_output = attn_output.reshape(bsz, q_len, -1)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    else: # added by dpi for saving memory
        attn_weights = attn_weights

    return attn_output, attn_weights, past_key_value

def attn_analysis_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

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

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

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

    attn_output = attn_output.reshape(bsz, q_len, -1)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    else: # added by dpi for saving memory
        attn_weights = attn_weights

    return attn_output, attn_weights, past_key_value


def modify_method_of_instance(instance, target_class_name, target_method_name, new_method, visited_instances=None):
    """
        This function modifies the method of an instance of a model class. 
        It's part from chat-GPT.
        It will replace the method  with the new method.
        Currently, we only use this function to modify the attention method of a model. Do not test it further. 

        instance: 
            instance of a model to modify.
        target_class_name: 
            name of the attention class to modify. E.g. 'LlamaAttention', 'GPTNeoXAttention', etc.
        new_method: new method to replace the original method. E.g. 'self_extend_forward'. 
            It should include a parameter 'self' to be binded to the instance.
    """
    target_found = False
    if visited_instances is None:
        visited_instances = set()
    # Unique identifier for the instance (using id() since object's id is unique)
    instance_id = id(instance)
    if instance_id in visited_instances:
        target_found = False
        return target_found
    # Add the instance to the already_visited set
    visited_instances.add(instance_id)

    # Check if this instance is of the target class
    if instance.__class__.__name__ == target_class_name:
        bond_method = MethodType(new_method, instance) 
        setattr(instance, target_method_name, bond_method)
        target_found = True
        return target_found
    elif hasattr(instance, '__dict__'):
        for attr_name, attr_value in instance.__dict__.items():
            if isinstance(attr_value, object) and not isinstance(attr_value, (list, tuple, dict, set)):
                _found = modify_method_of_instance(attr_value, target_class_name, target_method_name, new_method, visited_instances)
                if _found:
                    target_found = True
            elif isinstance(attr_value, (list, tuple)):
                for item in attr_value:
                    if isinstance(item, object):
                        _found = modify_method_of_instance(item, target_class_name, target_method_name, new_method, visited_instances)
                        if _found:
                            target_found = True
            # If attribute value is a dictionary, iterate over its values and recurse
            # E.g, for a ModuleList, its moudels are stored in a dictionary: ._modules
            elif isinstance(attr_value, dict):
                for key, value in attr_value.items():
                    if isinstance(value, object):
                        _found = modify_method_of_instance(value, target_class_name, target_method_name, new_method, visited_instances)
                        if _found:
                            target_found = True
            # If attribute value is a set, iterate and recurse
            elif isinstance(attr_value, set):
                for item in attr_value:
                    if isinstance(item, object):
                        _found = modify_method_of_instance(item, target_class_name, target_method_name, new_method, visited_instances)
                        if _found:
                            target_found = True

    return target_found

def get_model_and_tokenizer(model_name, cfg, method = "dpi", params={}, max_position_embeddings=None):
    '''
        loaded_model: 
            model to apply the self-attention extension. 

            Two recommended scale factor:
                yarn: https://arxiv.org/abs/2309.00071
                log: https://arxiv.org/abs/2202.12172 ; https://kexue.fm/archives/8823
            This is helpful while retrieving a long sequence (e.g a long passkey).
            But on real-world data, the impact is minor. (e.g. on LongBench, LEval).

            The reported results in our paper does not use this scale except for long passkey retrieval.
    '''

    if 'llama' in model_name.lower():
        config = AutoConfig.from_pretrained(model_name, token=llama_token)
        config.return_dict_in_generate = True
        config.num_hidden_layers = 2
        if max_position_embeddings is not None:
            config.max_position_embeddings = max_position_embeddings
        if method == "dpi":
            config.dpi_config = params
            config.output_attentions = True
 
            tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left',token=llama_token)
            model = LlamaForCausalLM.from_pretrained(model_name, config = config, torch_dtype=torch.bfloat16, token=llama_token)
            modifed_1 = modify_method_of_instance(model, "LlamaSdpaAttention", "forward", P.Llama.dpi_llamaattn_forward)
            modifed_2 = modify_method_of_instance(model, "LlamaRotaryEmbedding", "forward", P.Llama.rope_forward_dpi_with_pi)
            modifed_3 = modify_method_of_instance(model, "LlamaForCausalLM", "_sample", P.Llama.dpi_sample)
            modifed_4 = modify_method_of_instance(model, "LlamaForCausalLM", "prepare_inputs_for_generation", P.Llama.dpi_prepare_inputs_for_generation)
            modifed_5 = modify_method_of_instance(model, "LlamaForCausalLM", "_update_model_kwargs_for_generation", P.Llama.dpi_update_model_kwargs_for_generation)
            modifed_6 = modify_method_of_instance(model, "LlamaModel", "forward", P.Llama.dpi_llamamodel_forward)

            
            if (not modifed_1) or (not modifed_2) or (not modifed_3) or (not modifed_4) or (not modifed_5) or (not modifed_6):
                raise Exception(f"Failed to modify the attention method of {model_name}")
        elif method == "dpi_yarn":
            config.output_attentions = True
            config.rope_scaling = {"rope_type":"yarn","factor":1}
            tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left',token=llama_token)
            model = LlamaForCausalLM.from_pretrained(model_name, config = config, torch_dtype=torch.bfloat16, token=llama_token)
            modifed_1 = modify_method_of_instance(model, "LlamaSdpaAttention", "forward", P.Llama.dpi_llamaattn_forward)
            modifed_2 = modify_method_of_instance(model, "LlamaRotaryEmbedding", "forward", P.Llama.rope_forward_dpi_with_yarn)
            modifed_3 = modify_method_of_instance(model, "LlamaForCausalLM", "_sample", P.Llama.dpi_sample)
            modifed_4 = modify_method_of_instance(model, "LlamaForCausalLM", "prepare_inputs_for_generation", P.Llama.dpi_prepare_inputs_for_generation)
            modifed_5 = modify_method_of_instance(model, "LlamaForCausalLM", "_update_model_kwargs_for_generation", P.Llama.dpi_update_model_kwargs_for_generation)
            modifed_6 = modify_method_of_instance(model, "LlamaModel", "forward", P.Llama.dpi_llamamodel_forward)
            
            if (not modifed_1) or (not modifed_2) or (not modifed_3) or (not modifed_4) or (not modifed_5) or (not modifed_6):
                raise Exception(f"Failed to modify the attention method of {model_name}")
        elif method == "repro_se":
            config.output_attentions = True
            tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left',token=llama_token)
            model = LlamaForCausalLM.from_pretrained(model_name, config = config, attn_implementation="eager",torch_dtype=torch.bfloat16, token=llama_token)
  
            if "Llama-3" in model_name:
                SE.apply(model, params['group_size'], params['window_size'], enable_flash_attention=False, scale_base=-1, use_4_40 = True)
            else:
                SE.apply(model, params['group_size'], params['window_size'], enable_flash_attention=False, scale_base=-1, use_4_40 = True)
        elif method == "repro_yarn":

            factor = cfg['max_pe']/config.max_position_embeddings
            config.output_attentions = True
            config.rope_scaling = {"rope_type":"yarn","factor":factor}# For llama3-16k

            tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left',token=llama_token)
            model = LlamaForCausalLM.from_pretrained(model_name, config = config, attn_implementation="eager",torch_dtype=torch.bfloat16, token=llama_token)

            modifed_1 = modify_method_of_instance(model, "LlamaAttention", "forward", attn_analysis_attn_forward)
        elif method == "repro_ntk":

            factor = cfg['max_pe']/config.max_position_embeddings
            config.output_attentions = True
            config.rope_scaling = {"rope_type":"dynamic","factor":factor}
            config.static_ntk = True
            
            tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left',token=llama_token)
            model = LlamaForCausalLM.from_pretrained(model_name, config = config, attn_implementation="eager", torch_dtype=torch.bfloat16, token=llama_token)
            modifed_1 = modify_method_of_instance(model, "LlamaRotaryEmbedding", "_dynamic_frequency_update", P.Llama._dynamic_frequency_update)
            modifed_1 = modify_method_of_instance(model, "LlamaAttention", "forward", attn_analysis_attn_forward)
        elif method == "repro_dynamic_ntk":
            config.output_attentions = True
            config.rope_scaling = {"rope_type":"dynamic", "factor":params['factor']}
            config.static_ntk = False

            tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left',token=llama_token)
            model = LlamaForCausalLM.from_pretrained(model_name, config = config, attn_implementation="eager", torch_dtype=torch.bfloat16, token=llama_token)
            modifed_1 = modify_method_of_instance(model, "LlamaRotaryEmbedding", "_dynamic_frequency_update", P.Llama._dynamic_frequency_update)
            modifed_2 = modify_method_of_instance(model, "LlamaRotaryEmbedding", "forward", P.Llama.ori_rope_forward)
            modifed_1 = modify_method_of_instance(model, "LlamaAttention", "forward", attn_analysis_attn_forward)
        elif method == "repro_chunkllama":
            config.output_attentions = True
            CLflash_dec.replace_with_chunkllama(pretraining_length=config.max_position_embeddings, max_pe=cfg['max_pe'])
            tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left',token=llama_token)
            model = LlamaForCausalLM.from_pretrained(model_name, config = config, attn_implementation="eager", torch_dtype=torch.bfloat16, token=llama_token)
        elif method == "repro_original":
            config.output_attentions = True
            tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left',token=llama_token)
            model = LlamaForCausalLM.from_pretrained(model_name, config = config, attn_implementation="eager", torch_dtype=torch.bfloat16, token=llama_token)
            modifed_1 = modify_method_of_instance(model, "LlamaAttention", "forward", attn_analysis_attn_forward)
        else:
            raise Exception(f"Wrong interpolation method: {method}")
    else:
        raise NotImplementedError
    
    return model, tokenizer