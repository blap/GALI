import torch
from transformers.models.llama.modeling_llama import *
from transformers.models.gpt_neox.modeling_gpt_neox import *
import numpy as np
import torch.nn as nn
import math
from typing import Optional, Tuple
import torch.nn.functional as F
from transformers.cache_utils import Cache
# from flash_attn import flash_attn_func, flash_attn_varlen_func
import inspect
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.generation.utils import GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput, GenerateNonBeamOutput, LogitsProcessorList, StoppingCriteriaList, GenerationConfig
from models.DPI.help_funcs import *
import os
from torch.profiler import profile, record_function, ProfilerActivity

from collections import Counter
import copy

def array_statistics(arr):
    arr = np.array(arr)
    
    max_value = np.max(arr)
    min_value = np.min(arr)
    
    median_value = np.median(arr)
    
    mean_value = np.mean(arr)
    
    variance_value = np.var(arr)
    
    counter = Counter(arr)
    most_common_top10 = counter.most_common(10)  
    
    top10_counts = {value: count for value, count in most_common_top10}
    
    return {
        "max": max_value,
        "min": min_value,
        "median": median_value,
        "mean": mean_value,
        "variance": variance_value,
        "most_common_top10": top10_counts,
    }

def dpi_sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    max_position_embeddings = self.config.max_position_embeddings

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape

    if output_logits:
        # return all of the logits for ppl testing
        model_kwargs['num_logits_to_keep'] = 0

    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
    # print("model_kwargs 1",model_kwargs)
    # print("max_length ",max_length, cur_len)
    dpi_config = self.config.dpi_config
    local_window = dpi_config.get("local_window",0)
    chunk_coe = dpi_config.get('chunk_coe', None)
    # print("dpi_config ",dpi_config)

    # reset the position_ids cache
    setattr(self.model, "tmp_position_ids", {'prefill-long':[], 'decode':[]})

    # truncate the cache_position in the training context window
    # we split the input_ids into several parts, one is within the training context window, others are computed with PI one by one
    # and the chunk size is also the number of positions to be interpolated
    # Note, here the training context window must be longer than left padded tokens (Normally it will be)
    # max_training_context = 3
    # print("input_ids ",input_ids)
    c_input_ids = None
    chunk_size_list = []
    if cur_len > max_position_embeddings:
        chunk_size_list = get_chunk_size_list(cur_len, max_position_embeddings, chunk_coe)
        # print("chunk_size_list ",chunk_size_list)
        input_ids_list = torch.split(input_ids, chunk_size_list, dim = 1)
        input_ids, c_input_ids = input_ids_list[0], input_ids_list[1:]
        # truncate the attention mask
        # because we don't use attention mask to control the generation, but use the stopping critria (for llama default, it's max_new_token + eos token), we don't need to worry whether the later appended attention mask is 1 or 0 (in the funciton _update_model_kwargs_for_generation)
        # As for in the batch inference, even though it uses left padding, it does't matter
        # the position ids are generated according to the attention mask, we don't need to process it here
        model_kwargs['attention_mask'] = torch.split(model_kwargs['attention_mask'] , [max_position_embeddings, cur_len - max_position_embeddings], dim = 1)[0]
        model_kwargs['cache_position'] = model_kwargs['cache_position'][:input_ids.shape[-1]]


    num_hidden_layers = self.config.num_hidden_layers
    in_prefill = True
    process_c_input_ids = False
    first_in = True
    while self._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
    ):
        # prepare model inputs
        if process_c_input_ids == False:
            model_inputs = self.prepare_inputs_for_generation(input_ids, process_c_input_ids=process_c_input_ids, **model_kwargs)
        else:
            model_inputs = self.prepare_inputs_for_generation(input_ids, process_c_input_ids=process_c_input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        device = input_ids.device
        # Start position interpolation:
        if input_ids.shape[1] > max_position_embeddings and first_in == False:
            new_position_ids = []

            if in_prefill == False:
                last_position_ids = self.model.tmp_position_ids['decode'][-1]
                new_position_ids = decode_interpolation(last_position_ids, max_position_embeddings, unfinished_sequences, device,local_window)
            elif in_prefill == True:
                last_position_ids = self.model.tmp_position_ids['decode'][-1]
                new_position_ids = prefill_interpolation(input_ids, last_position_ids, max_position_embeddings, unfinished_sequences, device, local_window)

            model_inputs["position_ids"] = new_position_ids # bs, current length
        elif first_in == False:
            # must in the decode stage, regardless of whether pi is used 
            # Expend the position_ids in a normal range
            # only support len(input_ids[0]) - len(model_inputs["position_ids"][0]) = 1
            if input_ids.shape[1] - model_inputs["position_ids"].shape[1] == 1:
                model_inputs["position_ids"] = torch.cat([model_inputs["position_ids"], model_inputs["position_ids"][:, -1:]+1],dim=1) # bs, current length
            # print("model_inputs position_ids ", model_inputs["position_ids"])

        # update cache: 
        model_kwargs['position_ids'] = model_inputs["position_ids"]

        # forward pass to get next token
        outputs = self(**model_inputs, return_dict=True)
        # print("outputs 0000",outputs.logits.shape)
        
        first_in = False

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        if in_prefill == True:
            if len(chunk_size_list) > 1:
                chunk_size_list = chunk_size_list[1:]
                num_new_tokens = chunk_size_list[0]
            else:
                num_new_tokens = 1
        else:
            num_new_tokens = 1
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
            num_new_tokens = num_new_tokens
        )
        if synced_gpus and this_peer_finished:
            continue
        # if the length of pre-filled context is longer than training context window, split it and feed the second part in a way one-by-one
        
        if c_input_ids is not None:
            process_c_input_ids = num_new_tokens
            if type(c_input_ids) == torch.Tensor:
                if c_input_ids.shape[1] > 1:
                    next_tokens, c_input_ids = torch.split(c_input_ids, [1, c_input_ids.shape[1] - 1], dim=1)
                else:
                    next_tokens, c_input_ids = c_input_ids, None
            elif type(c_input_ids) == tuple:
                if len(c_input_ids) > 1:
                    next_tokens, c_input_ids = c_input_ids[0], c_input_ids[1:]
                else:
                    next_tokens, c_input_ids = c_input_ids[0], None
        
            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        else:
            in_prefill = False
            process_c_input_ids = False
            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits.clone()[:, -1, :].float()
            next_token_logits = next_token_logits.to(input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            # print(next_tokens.shape, input_ids.shape)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                # return all of the logits for ppl testing
                # print("outputs ",outputs.logits.shape)
                # num_logits_to_keep
                raw_logits += (outputs.logits.clone().float().to('cpu'),)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                ) 
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        # print("unfinished_sequences ",unfinished_sequences)
        # print("unfinished_sequences  ",unfinished_sequences)
        # print("\n")
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len = input_ids.shape[1]

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids

def dpi_update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ):
    # in prefill stage, we pass num_new_tokens as the chunk_size

    # update past_key_values keeping its naming used in model code
    cache_name, cache = self._extract_past_from_model_output(outputs)
    model_kwargs[cache_name] = cache
    if getattr(outputs, "state", None) is not None:
        model_kwargs["state"] = outputs.state

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1).repeat([1, num_new_tokens])], dim=-1) # dpi modification

    if not is_encoder_decoder:
        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], num_new_tokens))], dim=-1 # dpi modification
            )
    else:
        # update decoder attention mask
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            model_kwargs["decoder_attention_mask"] = torch.cat(
                [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], num_new_tokens))], # dpi modification
                dim=-1,
            )

    if model_kwargs.get("use_cache", True):
        model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
    else:
        past_positions = model_kwargs.pop("cache_position")
        new_positions = torch.arange(
            past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
        ).to(past_positions.device)
        model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
    return model_kwargs

def dpi_prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        process_c_input_ids = False,
        **kwargs,
    ):
    """
    Modifications:
    1. In the pre-fill stage, split the input to meet the requirement of training context window. 
    2. We don't compute the interpolated positions here
    3. We only pass the position ids in this function and modify them outside


    Prepare the model inputs for generation. In includes operations like computing the 4D attention mask or
    slicing inputs given the existing cache.

    See the forward pass in the model documentation for expected arguments (different models might have different
    requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
    """
    # 1. Handle BC:
    model_inputs = {}
    # - some models don't have `Cache` support (which implies they don't expect `cache_position` in `forward`)
    if self._supports_cache_class:
        model_inputs["cache_position"] = cache_position
    # - `cache_position` was not a mandatory input in `prepare_inputs_for_generation` for those models, and this
    #   function may be called outside of `generate`. Handle most use cases by creating `cache_position` on the fly
    #   (this alternative is not as robust as calling `generate` and letting it create `cache_position`)
    elif cache_position is None:
        past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        cache_position = torch.arange(past_length, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

    # 2. Generic cache-dependent input preparation
    # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
    # Exception 1: when passing input_embeds, input_ids may be missing entries
    # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
    # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case

    if past_key_values is not None:
        model_inputs["past_key_values"] = past_key_values
        if process_c_input_ids == False:
            if inputs_embeds is not None or cache_position[-1] >= input_ids.shape[1]:  # Exception 1 or Exception 3
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
        else:
            input_ids = input_ids[:, -process_c_input_ids:]

    # 3. Prepare base model inputs
    input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if not self.config.is_encoder_decoder:
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs[input_ids_key] = None
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            # `clone` calls in this function ensure a consistent stride. See #32227
            model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
            model_inputs["inputs_embeds"] = None
    else:
        model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)

    # 4. Create missing `position_ids` on the fly
    # print("kwargs.get(position_ids) ",kwargs.get("position_ids"))
    if (
        attention_mask is not None
        and kwargs.get("position_ids") is None
        and "position_ids" in set(inspect.signature(self.forward).parameters.keys())
    ):
        # when go into this function in the first time, we can use the attention mask to compute the position ids because it won't exceed the training context window
        # print("4. Create missing `position_ids` on the fly")
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        kwargs["position_ids"] = position_ids  # placed in kwargs for further processing (see below)
    else:
        position_ids = kwargs["position_ids"]



    # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
    # We can't slice the position_ids here
    for model_input_name in ["token_type_ids"]:
        model_input = kwargs.get(model_input_name)
        if model_input is not None:
            if past_key_values:
                model_input = model_input[:, -input_ids.shape[1] :]
                model_input = model_input.clone(memory_format=torch.contiguous_format)
            model_inputs[model_input_name] = model_input
    model_inputs["position_ids"] = position_ids

    # 6. Create 4D attention mask is we are using a `StaticCache` (important for performant compiled forward pass)
    if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
        if model_inputs["inputs_embeds"] is not None:
            batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
            device = model_inputs["inputs_embeds"].device
        else:
            batch_size, sequence_length = model_inputs[input_ids_key].shape
            device = model_inputs[input_ids_key].device

        # Create the causal mask with fixed shape in advance, to reduce recompilations. If the function to create
        # the 4D causal mask exists, it should be present in the base model (XXXModel class).
        base_model = getattr(self, self.base_model_prefix, None)
        if base_model is None:
            causal_mask_creation_function = getattr(
                self, "_prepare_4d_causal_attention_mask_with_cache_position", None
            )
        else:
            causal_mask_creation_function = getattr(
                base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
            )
        if causal_mask_creation_function is None:
            logger.warning_once(
                f"{self.__class__.__name__} has no `_prepare_4d_causal_attention_mask_with_cache_position` method "
                "defined in its base modeling class. Compiled forward passes will be sub-optimal. If you're "
                "writing code, see Llama for an example implementation. If you're a user, please report this "
                "issue on GitHub."
            )
        else:
            attention_mask = causal_mask_creation_function(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )
    if attention_mask is not None:
        model_inputs["attention_mask"] = attention_mask

    # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
    for key, value in kwargs.items():
        if key not in model_inputs:
            model_inputs[key] = value

    # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
    model_inputs.pop("labels", None)
    return model_inputs


def dpi_llamamodel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    # set position_ids cache for returning
    if not hasattr(self, "tmp_position_ids"):
        setattr(self, "tmp_position_ids", {'prefill-long':[], 'decode':[]})

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False
    # sync device
    input_ids = input_ids.to(self.device)
    if cache_position is not None:
        cache_position = cache_position.to(self.device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(self.device)
    if position_ids is not None:
        position_ids = position_ids.to(self.device)


    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # kept for BC (non `Cache` `past_key_values` inputs)
    return_legacy_cache = False
    if use_cache and not isinstance(past_key_values, Cache):
        return_legacy_cache = True
        if past_key_values is None:
            past_key_values = DynamicCache()
        else:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
            )

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    ) # bs, 1, q, k
    if causal_mask.shape[2] > 1 and causal_mask.shape[2] < causal_mask.shape[3]:
        # print("causal_mask shape",causal_mask.shape, inputs_embeds.shape)
        # in this case (c_input_ids in prefill stage) the casual mask was wrong, we need to recompute the complete casual mask and truncate it
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=causal_mask.shape[3],
            target_length=causal_mask.shape[3],
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
            cache_position=torch.arange(causal_mask.shape[3],device = inputs_embeds.device),
            batch_size=inputs_embeds.shape[0],
        )
        # causal_mask = causal_mask[:,:,-inputs_embeds.shape[1]:,:]
        causal_mask = causal_mask[:,:,-inputs_embeds.shape[1]:,:].clone() # release memory


    hidden_states = inputs_embeds

    max_position_embeddings = self.config.max_position_embeddings

    # create position embeddings to be shared across the decoder layers

    # Dropped plan, due to the error estimation of the pi after the first computation of each layer
    # we use position_embeddings in the 3 cases: 
    # 1, in the prefill stage: no input is longer than training context length
    # 2, in the prefill stage: the first computation of each layer
    # 3, in the decode stage
    # in the above cases, the shape of position_ids is bs, current length

    dpi_config = self.config.dpi_config
    appro_attn = dpi_config.get("appro_attn", False)
    
    if appro_attn == False or position_ids.shape[1] <= max_position_embeddings:
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
    else:
        # for attention logit interpolation
        cos_1, sin_1 = self.rotary_emb(hidden_states, position_ids.ceil())
        cos_2, sin_2 = self.rotary_emb(hidden_states, position_ids.floor())
        position_embeddings = (cos_1, sin_1, cos_2, sin_2)


    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)
        del layer_outputs # save memory

    
    tmp_position_ids = self.tmp_position_ids
    tmp_position_ids['decode'].append(position_ids)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def ori_rope_forward(self, x, position_ids):
    if "dynamic" in self.rope_type:
        self._dynamic_frequency_update(position_ids, device=x.device)

    # Core RoPE block
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        # print("inv_freq_expanded ",inv_freq_expanded.device, position_ids_expanded.device, x.device)
        inv_freq_expanded = inv_freq_expanded.to(x.device) # add by dpi
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

    # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
    cos = cos * self.attention_scaling
    sin = sin * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def _dynamic_frequency_update(self, position_ids, device):
    """
    Modify this function to make it suitable for NTK
    """
    if self.config.static_ntk == True:
        if getattr(self, "reset_static_ntk", False) == False:
            config = copy.deepcopy(self.config)
            seq_len = self.original_max_seq_len * config.rope_scaling['factor']
            config.rope_scaling['factor'] = 1
            inv_freq, self.attention_scaling = self.rope_init_fn(
                config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False) 
            setattr(self, "reset_static_ntk", True)
        return
    seq_len = torch.max(position_ids) + 1
    if seq_len > self.max_seq_len_cached:  # growth
        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device, seq_len=seq_len, **self.rope_kwargs
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
        self.max_seq_len_cached = seq_len

    if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
        self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
        self.max_seq_len_cached = self.original_max_seq_len



# dynamic position interpolation using position ids only
# In this case, the position_ids with interpolated indecies should be computed in advance
def rope_forward_dpi_with_pi(self, x, position_ids, p_start=None, p_end=None, cos_out=None, sin_out=None):
    '''
    In this case, self.rope_type should be 'default'
    Return two cos and sin embs, one for the context within the original_max_seq_len, the other one for the the rest context

    position_ids has two possible shapes:
    1. in the prefill stage:
        a. first computation of the first layer: bs, current length
        b. otherwise: bs, current length, current length (including the interpolated position ids)
    2. in the decode stage: bs, current length (including the interpolated position ids)

    When the shape is bs, current length, just follow the original logic and return bs, current length, dim
    But if the shape is bs, current length, current length, the returned shape will be bs, current length, current length, dim. It's extreamly large. To balance the time and space usage, we use a forloop outside this function and pass the p_start/p_end to compute the attentions segment-by-segment. It also means we can't pass or use the position embedding in the prefill stage. In this case, the memory usage will be (p_end-p_start) times.

    '''
    with torch.no_grad():
        # todo: cache implementation
        # if "dynamic" in self.rope_type:
        #     self._dynamic_frequency_update(position_ids, device=x.device)
        shape = position_ids.shape
        # if True:
        #     position_ids = torch.floor(position_ids)
        # Core RoPE block
        if len(shape) == 2:
            # return bs, len,  dim
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()
        else:
            # return bs, p_end - p_start, len, dim
            segment_position_ids = position_ids[:, p_start:p_end, :]
            inv_freq_expanded = self.inv_freq[None, None, None, :].float()
            position_ids_expanded = segment_position_ids[:,:,:,None].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            if len(shape) == 2:
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            else:
                freqs = position_ids_expanded * inv_freq_expanded
            if cos_out is None or sin_out is None:
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()
                cos = cos * self.attention_scaling
                sin = sin * self.attention_scaling
                return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
                # return cos, sin
            else:
                emb = torch.cat((freqs, freqs), dim=-1).to(dtype=x.dtype)
                torch.cos(emb, out=cos_out)
                torch.sin(emb, out=sin_out)
                return cos_out, sin_out
  
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    shape = hidden_states.shape
    if len(shape) == 4:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    else:
        batch, num_key_value_heads, seglen, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :, :].expand(batch, num_key_value_heads, n_rep, seglen, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, seglen, slen, head_dim)

def rotate_half(x, inplace=False):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def dpi_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1, p_start=None, p_end=None, seg = False):
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

    # new:
    cos = cos.unsqueeze(unsqueeze_dim) # to be [bs, 1, current length, dim] or [bs, 1, seg length, current length, dim]
    sin = sin.unsqueeze(unsqueeze_dim)
    # cos = cos[:,None,...]
    # sin = sin[:,None,...]
    # print("cos.shape, sin.shape ", cos.shape, sin.shape, q.shape, k.shape)
    bs, head, q_length, dim = q.shape # bs, head, len, dim
    if q_length == 1 and seg == False: 
        # decode stage
        if len(cos.shape) == 4:
            # use pi for each token
            q = (q * cos[:,:, -1: ,:]) + (rotate_half(q) * sin[:,:, -1: ,:]) if not q is None else None
            k = (k * cos) + (rotate_half(k) * sin) if not k is None else None
        else:
            # use the same pi for all the tokens, i.e., the original implementation
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin) 
    else:
        # prefill stage
        if len(cos.shape) == 4:
            q = (q * cos[:,:,-q_length:,:]) + (rotate_half(q) * sin[:,:,-q_length:,:]) if not q is None else None
            k = (k * cos) + (rotate_half(k) * sin) if not k is None else None
        elif p_start is not None and p_end is not None:
            pass #
        else:
            # use the same pi for all the tokens, i.e., the original implementation
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin) 
    return q, k


def attention_score_approximate(cos_1, cos_2, sin_1, sin_2, query_states, key_states, position_ids, num_key_value_groups, head_dim, noise_params):
    # cos_1, sin_1: new_pi.ceil().
    # cos_2, sin_2: new_pi.floor().
    # print("prefill query_states ",query_states.shape, key_states.shape, cos_1.shape, sin_1.shape, prefill_type)
    q_len = query_states.shape[2]
     
    query_states_ceil, key_states_ceil = dpi_apply_rotary_pos_emb(query_states, key_states, cos_1, sin_1)
    key_states_ceil = repeat_kv(key_states_ceil, num_key_value_groups)
    attend_floor = torch.matmul(query_states_ceil, key_states_ceil.transpose(2, 3)) / math.sqrt(head_dim)
    
    query_states_floor, key_states_floor = dpi_apply_rotary_pos_emb(query_states, key_states, cos_2, sin_2)
    key_states_floor = repeat_kv(key_states_floor, num_key_value_groups)
    attend_ceil = torch.matmul(query_states_ceil, key_states_floor.transpose(2, 3)) / math.sqrt(head_dim)
    
    rel_coef = position_ids[:,-q_len:].ceil().unsqueeze(2).to(attend_floor.dtype) - position_ids.unsqueeze(1).to(attend_floor.dtype)
    rel_coef.remainder_(1)

    attend_ceil.sub_(attend_floor)
    attend_ceil.mul_(-1).mul_(rel_coef)
    attend_floor.sub_(attend_ceil)
    
    del attend_ceil
    del key_states_floor
    del query_states_floor
    del key_states_ceil
    del query_states_ceil
    # attn = attend_floor + (attend_ceil - attend_floor ) * rel_coef[:,-q_len:,:]
    # print("attn dif ", (attn - attend_ceil).sum())

    if noise_params['noise_type'] == "uniform":
        attend_floor = add_uniform_noise_to_rel_coef(attend_floor, rel_coef, noise_params['addon_relcoef'], noise_params['scale_max'], noise_params['scale_std'], noise_params['std_base'])
    elif noise_params['noise_type'] == "gaussian":
        attend_floor = add_gaussian_noise_to_rel_coef(attend_floor, rel_coef, noise_params['addon_relcoef'], noise_params['scale_mean'], noise_params['scale_std'], noise_params['std_base'])
    return attend_floor

def dpi_llamaattn_forward(
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
    '''
    In the decode stage, we only support the way using k/v cache now.
    We only support batch_size == 1
    '''
    bsz, q_len, _ = hidden_states.size()
    dpi_config = self.config.dpi_config
    appro_attn = dpi_config.get("appro_attn", True)
    noise_params = {
        "noise_type": dpi_config.get("noise_type", "no"),
        "addon_relcoef": dpi_config.get("addon_relcoef", 1),
        "scale_mean": dpi_config.get("scale_mean", 1),
        "scale_std": dpi_config.get("scale_std", 1),
        "std_base": dpi_config.get("std_base", 1.0),
        "scale_max": dpi_config.get("scale_max", 1),
    }
    use_chunk_softmax = dpi_config.get("use_chunk_softmax", False)

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

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache

        # we don't cache the states with position information
        # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        cache_kwargs = {}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    else:
        # todo: add k/v cache checking in the decode stage
        pass


    k_len = key_states.shape[-2]
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    if q_len > 1:
        # in the prefill stage 
        if k_len <= self.max_position_embeddings:
            # no input is longer than training context window
            # position_embeddings shape: bs, current length, dim
            cos, sin = position_embeddings
            # print("query_states ",query_states.shape, key_states.shape, cos.shape, sin.shape)
            query_states, key_states = dpi_apply_rotary_pos_emb(query_states, key_states, cos, sin)
            
            key_states = repeat_kv(key_states, self.num_key_value_groups)

            # print("key_states ",query_states.shape, key_states.shape)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        elif len(position_ids.shape) == 3:
            pass
        else:
            if appro_attn == False:
                cos, sin = position_embeddings
                # print("prefill query_states ",query_states.shape, key_states.shape, cos.shape, sin.shape)
                query_states, key_states = dpi_apply_rotary_pos_emb(query_states, key_states, cos, sin, prefill_type=2)
                
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            else:
                cos_1, sin_1, cos_2, sin_2 = position_embeddings
                attn_weights = attention_score_approximate(cos_1, cos_2, sin_1, sin_2, query_states, key_states, position_ids, num_key_value_groups = self.num_key_value_groups, head_dim = self.head_dim, noise_params=noise_params)
    else:
        # in the decode stage 
        # position_embeddings shape: bs, current length, dim
        if appro_attn == False or k_len <= self.max_position_embeddings:
            cos, sin = position_embeddings
            query_states, key_states = dpi_apply_rotary_pos_emb(query_states, key_states, cos, sin)
            key_states = repeat_kv(key_states, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        else:
            cos_1, sin_1, cos_2, sin_2 = position_embeddings
            attn_weights = attention_score_approximate(cos_1, cos_2, sin_1, sin_2, query_states, key_states, position_ids, num_key_value_groups = self.num_key_value_groups, head_dim = self.head_dim, noise_params=noise_params)

        

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    
    if use_chunk_softmax == True:
        # upcast attention to fp32
        attn_weights = attn_weights.to(torch.float32)
        attn_weights = chunk_softmax(attn_weights)
    else:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype = torch.float32)
    # attn_weights = in_place_softmax(attn_weights)
    attn_weights = attn_weights.to(query_states.dtype)
#     attn_weights = chunked_softmax(attn_weights, query_states, 500)
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
    else:
        attn_weights = attn_weights

    return attn_output, attn_weights, past_key_value

def chunked_softmax(attn_weights, query_states, chunk_size):
    """
    Compute softmax in chunks to reduce memory usage.

    Args:
        tensor: Input tensor of shape (..., seq_len, ...).
        dim: Dimension along which to compute the softmax.
        chunk_size: Size of each chunk.

    Returns:
        Tensor: Softmax output with the same shape as input tensor.
    """
    chunks = torch.split(attn_weights, chunk_size, dim=2)
    softmax_chunks = [torch.softmax(chunk, dim=-1, dtype=torch.float32).to(query_states.dtype) for chunk in chunks]
    softmax_chunks = torch.cat(softmax_chunks, dim=2)
    return softmax_chunks