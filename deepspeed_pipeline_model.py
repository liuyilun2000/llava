
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa

from deepspeed.pipe import PipelineModule, LayerSpec

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
        return hidden_states, attention_mask, labels

class LanguageModelLayerWrapper(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    def forward(self, inputs):
        hidden_states, attention_mask, labels = inputs
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
                position_ids=position_ids)
            hidden_states = layer_outputs[0]
        return hidden_states, attention_mask, labels

class LanguageModelFinalWrapper(nn.Module):
    def __init__(self, norm, lm_head):
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head
    def forward(self, inputs):
        hidden_states, attention_mask, labels = inputs
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
        return shift_logits, shift_labels

def loss_fn(outputs, labels_original):
    shift_logits, shift_labels = outputs    # we use shift_labels here instead of labels_original because of the image tokens. 
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
    )
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
            model.language_model.model.norm,
            model.language_model.lm_head
        ),
        #LayerSpec(LossLayer)
    ]
    return layer_specs