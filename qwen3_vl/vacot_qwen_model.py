import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from torch.nn import CrossEntropyLoss
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast
from transformers.cache_utils import StaticCache
from transformers.utils import ModelOutput
import copy
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
class Qwen3VLForVaCoT(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.num_line_break = 0
        self.num_sub_imgs = 0
        self.captured_attentions = []

        self.vision_aware_heads = {}
        
        self.last_t_mas_score = 0.0 
        self._register_intervention_hooks()
        self.cooldown_steps = 0

        

        self.t_mas_threshold = 25 
        self.ema_alpha = 0.07
        self.ema_k = 1.2
        self.ema_mean = 0.0
        self.ema_var = 0.0
        self.is_ema_init = False
        
        self.is_hunting_peak = False
        self.peak_score = 0.0 
        self.peak_attentions = None 
        self.pending_mas_trigger = False

        self.query_to_image_mask = None

        
    def _compute_initial_mas(self, inputs_embeds, image_mask, attention_mask):
        with torch.no_grad():
            base_outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=True
            )
            
            inputs_embeds_no_img = inputs_embeds.clone()
            inputs_embeds_no_img[image_mask] = 0.0 
            
            no_img_outputs = self.model(
                inputs_embeds=inputs_embeds_no_img,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=True
            )

            for layer_idx in range(len(self.model.language_model.layers)):
                attn_base = base_outputs.attentions[layer_idx].float()
                attn_no_img = no_img_outputs.attentions[layer_idx].float()

                eps = 1e-7
                p = attn_base.clamp(min=eps)
                q = attn_no_img.clamp(min=eps)
                m = 0.5 * (p + q)
                
                kl_p = p * (torch.log(p) - torch.log(m))
                kl_q = q * (torch.log(q) - torch.log(m))
                
                jsd = 0.5 * (kl_p + kl_q).sum(dim=-1)
                
                diff = jsd.mean(dim=-1)
                
                ta = torch.norm(attn_no_img, dim=-1).pow(2).mean(dim=-1)
                
                if diff.dim() > 1:
                    diff = diff[0]
                    ta = ta[0]
                
                mu_diff, std_diff = diff.mean(), diff.std()
                mu_ta, std_ta = ta.mean(), ta.std()
                
                valid_mask = (diff <= (mu_diff + std_diff)) & (ta <= (mu_ta + std_ta))
                diff = diff * valid_mask.float()

                mean_mas = diff.mean()
                top_heads = torch.nonzero(diff > mean_mas).squeeze(-1)
                
                num_vision_heads = top_heads.numel()
                self.vision_aware_heads[layer_idx] = top_heads.tolist()
           
            text_query_start = int(self.query_image_end)
            text_query_end = base_outputs.attentions[-1].shape[-1]
            
            if text_query_end > text_query_start:
                mid_layer = len(base_outputs.attentions) // 2
                query_to_img_attn = base_outputs.attentions[mid_layer][:, :, text_query_start:text_query_end, self.query_image_start:self.query_image_end]
                
                spatial_prior = query_to_img_attn.mean(dim=(1, 2))
                spatial_prior = spatial_prior / (spatial_prior.max() + 1e-6)
                self.query_to_image_mask = spatial_prior.unsqueeze(1)
            else:
                self.query_to_image_mask = None
                
    def _get_dynamic_semantic_bbox(self, attentions_1d, grid_h, grid_w, threshold=0.5):
        attentions_2d = attentions_1d.view(grid_h, grid_w)
        max_val = attentions_2d.max()
        
        core_mask = attentions_2d > (max_val * threshold)
        non_zero_coords = torch.nonzero(core_mask)
        
        if non_zero_coords.numel() == 0:
            max_idx = attentions_1d.argmax().item()
            return [max_idx], 1, 1
            
        r_min, r_max = non_zero_coords[:, 0].min().item(), non_zero_coords[:, 0].max().item()
        c_min, c_max = non_zero_coords[:, 1].min().item(), non_zero_coords[:, 1].max().item()
        
        MAX_TOKENS = 64
        while (r_max - r_min + 1) * (c_max - c_min + 1) > MAX_TOKENS:
            if (r_max - r_min) > (c_max - c_min):
                if attentions_2d[r_min, c_min:c_max+1].mean() < attentions_2d[r_max, c_min:c_max+1].mean():
                    r_min += 1
                else:
                    r_max -= 1
            else:
                if attentions_2d[r_min:r_max+1, c_min].mean() < attentions_2d[r_min:r_max+1, c_max].mean():
                    c_min += 1
                else:
                    c_max -= 1

        indices = []
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                indices.append(r * grid_w + c)
                
        return indices, r_min, r_max, c_min, c_max

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen3VLCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        judge = input_ids[:, -1] in [198]
        if judge:
            self.num_line_break += 1

        if self.num_sub_imgs > 0 and past_key_values is not None:

            if hasattr(past_key_values, "get_seq_length"):
                true_kv_len = past_key_values.get_seq_length()
            else:
                true_kv_len = past_key_values[0][0].shape[-2]
                
            if cache_position is not None and cache_position[0] < true_kv_len:
                offset = true_kv_len - cache_position[0].item()
                cache_position = cache_position + offset

            if attention_mask is not None:
                current_input_len = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
                expected_mask_len = true_kv_len + current_input_len
                if attention_mask.shape[-1] < expected_mask_len:
                    pad_len = expected_mask_len - attention_mask.shape[-1]
                    padding = torch.ones((attention_mask.shape[0], pad_len), dtype=attention_mask.dtype, device=attention_mask.device)
                    attention_mask = torch.cat([attention_mask, padding], dim=-1)
        vision_start_token_id = getattr(self.config, "vision_start_token_id", 151652)
        vision_end_token_id = getattr(self.config, "vision_end_token_id", 151653)
        image_token_id = getattr(self.config, "image_token_id", 151655)
        spatial_merge_size = getattr(self.config.vision_config, "spatial_merge_size", 2)
        merge_factor = spatial_merge_size ** 2
        
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.to(dtype=next(self.model.visual.parameters()).dtype)
                vision_outputs = self.model.visual(pixel_values, grid_thw=image_grid_thw)
                if not isinstance(vision_outputs, torch.Tensor):
                    image_embeds = vision_outputs[0]
                else:
                    image_embeds = vision_outputs
                image_embeds = self.model.visual.merger(image_embeds)
                self.reasoning_img_embeds = image_embeds[-(image_grid_thw[:, 1] * image_grid_thw[:, 2] // merge_factor)[-1]:, ...]
                self.query_image_start = (input_ids == vision_start_token_id).nonzero(as_tuple=True)[1][-1] + 1
                self.query_image_end = (input_ids == vision_end_token_id).nonzero(as_tuple=True)[1][-1]
                self.query_image_mask = torch.zeros_like(input_ids, device=input_ids.device, dtype=torch.bool)
                self.query_image_mask[:, self.query_image_start: self.query_image_end] = True
                
                self.num_line_break = 0
                self.num_sub_imgs = 0
                self.is_hunting_peak = False
                self.peak_score = 0.0
                self.peak_attentions = None
                self.pending_mas_trigger = False
                
                self.cooldown_steps = 0
                self.ema_mean = 0.0
                self.ema_var = 0.0
                self.is_ema_init = False
                self.t_mas_threshold = 25.0

                self.vision_aware_heads = {}
                self.last_t_mas_score = 0.0

                self.query_to_image_mask = None
                
                n_image_tokens = (input_ids == image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

                self._compute_initial_mas(inputs_embeds, self.query_image_mask, attention_mask)
                self.last_t_mas_score = 0.0
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (cache_position is not None and cache_position[0] == 0) or rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None: 
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(4, -1, -1)
        if hasattr(self, 'query_image_end') and position_ids.shape[-1] > getattr(self, 'query_image_end', 0):
            self.cached_full_pos_ids = position_ids.detach().clone()
        if hasattr(self, 'cooldown_steps') and self.cooldown_steps > 0:
            self.cooldown_steps -= 1
        trigger_condition = (
            getattr(self, 'pending_mas_trigger', False) and
            self.num_sub_imgs < 6 and
            getattr(self, 'cooldown_steps', 0) == 0
        )
        if trigger_condition:
            self.pending_mas_trigger = False 
            va_attentions = self.peak_attentions 
            self.peak_attentions = None 
            
            if self.query_image_mask.shape[-1] != va_attentions.shape[-1]:
                self.query_image_mask = torch.cat([self.query_image_mask, torch.zeros(self.query_image_mask.shape[0],
                                                                                      va_attentions.shape[-1] - self.query_image_mask.shape[-1],
                                                                                      device=self.query_image_mask.device).bool()],
                                                                                      dim=1)
            valid_image_attentions = va_attentions[self.query_image_mask]
            
            spatial_merge_size = getattr(self.config.vision_config, "spatial_merge_size", 2)
            grid_h = image_grid_thw[-1, 1].item() // spatial_merge_size
            grid_w = image_grid_thw[-1, 2].item() // spatial_merge_size
            
            attentions_1d = valid_image_attentions.squeeze()
            
            indices_list, r_min, r_max, c_min, c_max = self._get_dynamic_semantic_bbox(attentions_1d, grid_h, grid_w)
            
            indices = torch.tensor(indices_list, device=valid_image_attentions.device)
            sampled_reasoning_embeds = self.reasoning_img_embeds[indices]
            
            current_text_embed = inputs_embeds[:, -1:, :].clone() # [batch, 1, hidden_dim]
            visual_kv_embeds = sampled_reasoning_embeds.unsqueeze(0) # [1, num_img_tokens, hidden_dim]
            
            head_dim = current_text_embed.shape[-1]
            attn_weights = torch.matmul(current_text_embed, visual_kv_embeds.transpose(-1, -2)) / (head_dim ** 0.5)
            
            attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
            
            attended_visual_feature = torch.matmul(attn_probs, visual_kv_embeds)

            norm_text = torch.norm(current_text_embed, dim=-1, keepdim=True)
            norm_vis = torch.norm(attended_visual_feature, dim=-1, keepdim=True)
            aligned_visual_feature = attended_visual_feature * (norm_text / (norm_vis + 1e-6))
            
            alpha_gate = 0.2
            
            inputs_embeds[:, -1:, :] = current_text_embed + alpha_gate * aligned_visual_feature
            
            self.num_sub_imgs += 2
            self.cooldown_steps = 4
            
            
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        if outputs.attentions is not None and len(self.vision_aware_heads) > 0:
            t_mas_score = 0.0
            total_dangerous_heads = 0 
            total_deep_va_heads = 0
            k = self.model.language_model.config.num_attention_heads // 2

            start_idx = self.query_image_start
            end_idx = self.query_image_end

            for layer_idx in range(len(self.model.language_model.layers)):
                va_heads_idx = self.vision_aware_heads.get(layer_idx, [])
                if not va_heads_idx:
                    continue
                layer_attn_heads = outputs.attentions[layer_idx][:, va_heads_idx, -1, :]
                seq_len = outputs.attentions[layer_idx].shape[-1]
                
                img_attn = layer_attn_heads[..., start_idx:end_idx]

                head_mas_scores = img_attn.sum(dim=-1)

                t_mas_score += torch.topk(head_mas_scores, min(k, head_mas_scores.shape[-1]), dim=-1)[0].sum().item()

                    
            self.last_t_mas_score = t_mas_score
            
            if not self.is_ema_init:
                self.ema_mean = t_mas_score
                self.ema_var = 0.0
                self.is_ema_init = True
                self.t_mas_threshold = max(t_mas_score + 5.0, 15.0) 
            else:
                delta = t_mas_score - self.ema_mean
                self.ema_mean += self.ema_alpha * delta
                self.ema_var = (1 - self.ema_alpha) * (self.ema_var + self.ema_alpha * delta ** 2)
                std_dev = self.ema_var ** 0.5
                
                self.t_mas_threshold = max(self.ema_mean + self.ema_k * std_dev, 15.0)
            
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            logits_probs = torch.softmax(logits[:, -1, :], dim=-1)
            max_prob = torch.max(logits_probs, dim=-1)[0].item() 
            if self.num_sub_imgs < 6 and not self.pending_mas_trigger:
                if self.last_t_mas_score > self.t_mas_threshold:
                    self.is_hunting_peak = True
                    if self.last_t_mas_score > self.peak_score:
                        self.peak_score = self.last_t_mas_score
                        last_layer_idx = len(self.model.language_model.layers) - 1
                        va_heads_idx = self.vision_aware_heads.get(last_layer_idx, list(range(16)))
                        self.peak_attentions = outputs.attentions[-1][:, va_heads_idx, -1, :].mean(dim=1).detach().clone()
                
                if self.is_hunting_peak and self.last_t_mas_score < self.peak_score:
                    self.pending_mas_trigger = True
                    self.is_hunting_peak = False
                    self.peak_score = 0.0
        else:
            self.last_t_mas_score = 0
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )
