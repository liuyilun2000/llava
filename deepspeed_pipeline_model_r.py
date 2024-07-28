
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import warnings
from typing import List, Optional, Tuple, Union


import transformers
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa

from deepspeed.pipe import PipelineModule, LayerSpec

def load_balancing_loss_func(
    gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2, attention_mask: Optional[torch.Tensor] = None
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts



class LlavaMultiModalModuleWrapper(nn.Module):
    def __init__(self, config, vision_tower, embed_tokens, multi_modal_projector):
        super().__init__()
        self.config = config
        self.vision_tower = vision_tower
        self.embed_tokens = embed_tokens
        self.multi_modal_projector = multi_modal_projector
        self.device = self.vision_tower.device
    #
    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.config.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)
        #
        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]
        #
        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)
        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]
        #
        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)
        #
        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )
        #
        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        #
        if labels is None:
            final_labels = None
        #
        return final_embedding, final_attention_mask, final_labels
    #
    def forward(self, inputs):
        if len(inputs) == 4:
            input_ids, pixel_values, attention_mask, labels = inputs
        elif len(inputs) == 3:
            input_ids, attention_mask, labels = inputs
            pixel_values = None
        else:
            raise ValueError(
                f"Unexpected multimodal input format: {inputs}"
            )        
        inputs_embeds = self.embed_tokens(input_ids)        
        if pixel_values is not None and input_ids.shape[1] != 1:
            image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
            selected_image_feature = image_outputs.hidden_states[self.config.vision_feature_layer]
            if self.config.vision_feature_select_strategy ==  "default":
                selected_image_feature = selected_image_feature[:, 1:]
            elif self.config.vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature
            else:
                raise ValueError(
                    f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                )        
            image_features = self.multi_modal_projector(selected_image_feature)
            inputs_embeds, attention_mask, labels = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask, labels
            )  
        hidden_states = inputs_embeds
        batch_size, seq_length, _ = hidden_states.shape
        all_router_logits = torch.empty((self.config.text_config.num_hidden_layers,seq_length,self.config.text_config.num_local_experts)).to(hidden_states.device)
        all_shared_routing_adapter_router_logits = torch.empty(((self.config.text_config.num_hidden_layers,seq_length,self.config.text_config.shared_routing_adapter_num_experts))).to(hidden_states.device)
        current_layer=torch.tensor((0)).to(hidden_states.device)
        return current_layer, hidden_states, attention_mask, labels, all_router_logits, all_shared_routing_adapter_router_logits

class LanguageModelLayerWrapper(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    def forward(self, inputs):
        current_layer, hidden_states, attention_mask, labels, all_router_logits, all_shared_routing_adapter_router_logits = inputs
        current_layer = int(current_layer)
        batch_size, seq_length, _ = hidden_states.shape
        position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_((attention_mask == 0), 1)
        position_ids = position_ids.view(-1, seq_length).long()
        batch_size, seq_length, _ = hidden_states.shape
        attention_mask_4d = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            hidden_states,
            0, #sliding_window=self.config.text_config.sliding_window,
        )
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states=hidden_states, 
                attention_mask=attention_mask_4d, 
                position_ids=position_ids,
                output_router_logits=True)
            hidden_states = layer_outputs[0]
            current_router_logits = layer_outputs[-2]
            current_shared_routing_adapter_router_logits = layer_outputs[-1]
            
            all_router_logits[current_layer] = current_router_logits.to(all_router_logits.device)
            all_shared_routing_adapter_router_logits[current_layer] = current_shared_routing_adapter_router_logits.to(all_shared_routing_adapter_router_logits.device)
            current_layer += 1
        
        current_layer = torch.tensor(current_layer).to(hidden_states.device)
        return current_layer, hidden_states, attention_mask, labels, all_router_logits, all_shared_routing_adapter_router_logits

class LanguageModelFinalWrapper(nn.Module):
    def __init__(self, config, norm, lm_head):
        super().__init__()
        self.config = config
        self.norm = norm
        self.lm_head = lm_head
    def forward(self, inputs):
        current_layer, hidden_states, attention_mask, labels, all_router_logits, all_shared_routing_adapter_router_logits = inputs
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        # Shift so that tokens < n predict n
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:]
            shift_logits = logits[..., :-1, :][shift_attention_mask != 0].contiguous()            
            shift_labels = labels[..., 1:][shift_attention_mask != 0].contiguous() 
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        router_config = torch.tensor([    
            self.config.text_config.router_aux_loss_coef,
            self.config.text_config.shared_routing_adapter_router_aux_loss_coef,
            self.config.text_config.num_local_experts,
            self.config.text_config.num_experts_per_tok,
            self.config.text_config.shared_routing_adapter_num_experts,
            self.config.text_config.shared_routing_adapter_num_experts_per_tok
        ]).to(shift_logits.device)
        return shift_logits, shift_labels, attention_mask, all_router_logits, all_shared_routing_adapter_router_logits, router_config

def loss_fn(outputs, labels_original):
    shift_logits, shift_labels, attention_mask, all_router_logits, all_shared_routing_adapter_router_logits, router_config = outputs    # we use shift_labels here instead of labels_original because of the image tokens. 
    router_aux_loss_coef, shared_routing_adapter_router_aux_loss_coef, num_local_experts, num_experts_per_tok, shared_routing_adapter_num_experts, shared_routing_adapter_num_experts_per_tok = list(router_config)
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
    )
    aux_loss = load_balancing_loss_func(
        tuple(torch.unbind(all_router_logits)),
        int(num_local_experts),
        int(num_experts_per_tok),
        attention_mask,
    )
    loss += float(router_aux_loss_coef) * aux_loss.to(loss.device)  # make sure to reside in the same device
    shared_routing_adapter_aux_loss = load_balancing_loss_func(
        tuple(torch.unbind(all_shared_routing_adapter_router_logits)),
        int(shared_routing_adapter_num_experts),
        int(shared_routing_adapter_num_experts_per_tok),
        attention_mask,
    )
    #print(loss, aux_loss, shared_routing_adapter_aux_loss)
    loss += float(shared_routing_adapter_router_aux_loss_coef) * shared_routing_adapter_aux_loss.to(loss.device)  # make sure to reside in the same device

    return loss
    
def get_layer_specs(model):
    layer_specs = [
        LayerSpec(
            LlavaMultiModalModuleWrapper,
            model.config, 
            model.vision_tower,
            model.language_model.model.embed_tokens,
            model.multi_modal_projector
        ),
        LayerSpec(LanguageModelLayerWrapper, model.language_model.model.layers[:4]),
        LayerSpec(LanguageModelLayerWrapper, model.language_model.model.layers[4:8]),
        LayerSpec(LanguageModelLayerWrapper, model.language_model.model.layers[8:12]),
        LayerSpec(LanguageModelLayerWrapper, model.language_model.model.layers[12:16]),
        LayerSpec(LanguageModelLayerWrapper, model.language_model.model.layers[16:20]),
        LayerSpec(LanguageModelLayerWrapper, model.language_model.model.layers[20:24]),
        LayerSpec(LanguageModelLayerWrapper, model.language_model.model.layers[24:28]),
        LayerSpec(LanguageModelLayerWrapper, model.language_model.model.layers[28:]),        
        LayerSpec(
            LanguageModelFinalWrapper,
            model.config,
            model.language_model.model.norm,
            model.language_model.lm_head
        ),
        #LayerSpec(LossLayer)
    ]
    return layer_specs