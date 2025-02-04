from types import MethodType
from functools import partial
import models.DPI.patches as P
import models.SE.SelfExtend as SE
import models.DCA.flash_decoding_chunkllama_dpi as CLflash_dec
import torch
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoConfig
llama_token = ""
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
        config.num_hidden_layers = 1
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
            config.output_attentions = False
            tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left',token=llama_token)
            model = LlamaForCausalLM.from_pretrained(model_name, config = config, attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16, token=llama_token)
  
            if "Llama-3" in model_name:
                SE.apply(model, params['group_size'], params['window_size'], enable_flash_attention=True, scale_base=-1, flash_attention_impl="flash_attn", use_4_40 = True)
            else:
                SE.apply(model, params['group_size'], params['window_size'], enable_flash_attention=True, scale_base=-1, flash_attention_impl="flash_attn", use_4_40 = True)
        elif method == "repro_yarn":

            factor = cfg['factor']
            config.output_attentions = False
            config.rope_scaling = {"rope_type":"yarn","factor":factor}# For llama3-16k

            tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left',token=llama_token)
            model = LlamaForCausalLM.from_pretrained(model_name, config = config, attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16, token=llama_token)
        elif method == "repro_ntk":

            factor = cfg['factor']
            config.output_attentions = False
            config.rope_scaling = {"rope_type":"dynamic","factor":factor}
            config.static_ntk = True
            
            tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left',token=llama_token)
            model = LlamaForCausalLM.from_pretrained(model_name, config = config, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, token=llama_token)
            modifed_1 = modify_method_of_instance(model, "LlamaRotaryEmbedding", "_dynamic_frequency_update", P.Llama._dynamic_frequency_update)
        elif method == "repro_dynamic_ntk":
            config.output_attentions = False
            config.rope_scaling = {"rope_type":"dynamic", "factor":params['factor']}
            config.static_ntk = False

            tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left',token=llama_token)
            model = LlamaForCausalLM.from_pretrained(model_name, config = config, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, token=llama_token)
            modifed_1 = modify_method_of_instance(model, "LlamaRotaryEmbedding", "_dynamic_frequency_update", P.Llama._dynamic_frequency_update)
            modifed_2 = modify_method_of_instance(model, "LlamaRotaryEmbedding", "forward", P.Llama.ori_rope_forward)
        elif method == "repro_chunkllama":
            config.output_attentions = False
            CLflash_dec.replace_with_chunkllama(pretraining_length=config.max_position_embeddings, max_pe=cfg['max_pe'])
            tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left',token=llama_token)
            model = LlamaForCausalLM.from_pretrained(model_name, config = config, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, token=llama_token)
        elif method == "repro_original":
            config.output_attentions = False
            tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left',token=llama_token)
            model = LlamaForCausalLM.from_pretrained(model_name, config = config, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, token=llama_token)
        else:
            raise Exception(f"Wrong interpolation method: {method}")
        

    else:
        raise NotImplementedError
    return model, tokenizer