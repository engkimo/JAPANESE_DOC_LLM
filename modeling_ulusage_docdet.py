#    Copyright 2024 Ohori Ryosuke Ye (Modified from LLaVA)
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM
from .modeling_llama2_mam import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_ulusage_docdet import (UlusageDocDetConfig, UlusageDetVisionConfig, UlusageDocDetHReducerConfig, UlusageDocDetHRDocCompressorConfig)
from .visual_encoder import UlusageDetVisionModel, UlusageDocDetHReducerModel
from .visual_compressor import UlusageDocDetHRDocCompressor
from .processor import DocProcessor

from .constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from icecream import ic

from transformers import StoppingCriteria, TextStreamer

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

class UlusageDocDetMetaModel:
    _no_split_modules = ["UlusageDetVisionModel", "UlusageDocDetHReducerModel", "UlusageDocDetHRDocCompressor"]
    def __init__(self, config):
        super(UlusageDocDetMetaModel, self).__init__(config)
        self.vision_model = UlusageDetVisionModel(
            UlusageDetVisionConfig(**config.visual_config["visual_model"])
        )
        v_img_row_tokens = int((config.visual_config["visual_model"]['image_size']/config.visual_config["visual_model"]['patch_size']))
        v_img_col_tokens = v_img_row_tokens

        self.vision2text = UlusageDocDetHReducerModel(
            UlusageDocDetHReducerConfig(**config.visual_config["visual_hreducer"]), config.hidden_size
        )

        horizontal_reduce = int(config.visual_config["visual_hreducer"]['conv_shape'].split('x')[1])
        v2t_img_col_tokens = int(v_img_row_tokens / horizontal_reduce)

        self.hr_compressor = UlusageDocDetHRDocCompressor(
            UlusageDocDetHRDocCompressorConfig(**config.visual_config["visual_hrcompressor"]), 
            config.hidden_size, 
            v2t_img_col_tokens
        )

    def get_vision_tower(self):
        vision_model = getattr(self, 'vision_model', None)
        if type(vision_model) is list:
            vision_model = vision_model[0]
        return vision_model

    def get_vision2text(self):
        vision2text = getattr(self, 'vision2text', None)
        if type(vision2text) is list:
            vision2text = vision2text[0]
        return vision2text
    
    def get_hrcompressor(self):
        hrcompressor = getattr(self, 'hr_compressor', None)
        if type(hrcompressor) is list:
            hrcompressor = hrcompressor[0]
        return hrcompressor

class UlusageDocDetMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def encode_images(self, images, patch_positions):
        image_features = self.get_model().vision_model(images).last_hidden_state
        image_features = self.get_model().vision2text(encoder_hidden_states=image_features)
        image_features = self.get_model().hr_compressor(hidden_states=image_features, patch_positions=patch_positions)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, patch_positions
    ):  
        # ic(images.shape, patch_positions.shape)
        if images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            multiway_indices = torch.zeros_like(input_ids).long().to(self.device)
            return input_ids, multiway_indices, attention_mask, past_key_values, None, labels
        
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images, patch_positions)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images, patch_positions) # Sum(Crop Image Number) x L x d

        new_input_embeds = []
        new_modality_indicators = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                
                cur_modality_indicators = torch.zeros(len(cur_input_embeds)).long().to(self.device)
                new_modality_indicators.append(cur_modality_indicators)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            cur_modality_indicators = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                cur_new_input_embeds.append(cur_image_features)
                
                # Add modality indicator
                assert image_token_start == len(cur_input_ids[:image_token_start])
                cur_modality_indicators.append(torch.zeros(len(cur_input_ids[:image_token_start])).long())
                cur_modality_indicators.append(torch.ones(len(cur_image_features)).long())
                
                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                cur_modality_indicators.append(torch.zeros(len(cur_input_ids)).long())
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            
            # Modality
            cur_modality_indicators = [x.to(device=self.device) for x in cur_modality_indicators]
            cur_modality_indicators = torch.cat(cur_modality_indicators, dim=0)
            new_modality_indicators.append(cur_modality_indicators)
            
            
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)
            
            # Embedding
            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)
            
            # Modality
            new_modality_indicators_align = []
            for cur_modality_indicator in new_modality_indicators:
                cur_new_embed = torch.cat((cur_modality_indicator, torch.zeros(max_len - cur_modality_indicator.shape[0], dtype=cur_modality_indicator.dtype, device=cur_modality_indicator.device)), dim=0)
                new_modality_indicators_align.append(cur_new_embed)
            new_modality_indicators = torch.stack(new_modality_indicators_align, dim=0)
            
            # Label
            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)
            
            # Attention Mask
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            new_modality_indicators = torch.stack(new_modality_indicators, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]
        return None, new_modality_indicators, attention_mask, past_key_values, new_input_embeds, new_labels



class UlusageDocDetLlamaModel(UlusageDocDetMetaModel, LlamaModel):
    config_class = UlusageDocDetConfig

    def __init__(self, config: UlusageDocDetConfig):
        super(UlusageDocDetLlamaModel, self).__init__(config)


class UlusageDocDet2(LlamaForCausalLM, UlusageDocDetMetaForCausalLM):
    config_class = UlusageDocDetConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = UlusageDocDetLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def init_processor(self, tokenizer, basic_image_size, crop_anchors):
        self.processor = DocProcessor(tokenizer=tokenizer, image_size=basic_image_size, anchors=crop_anchors)
        return self.processor

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        # modality_indicators: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        patch_positions: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # print('modeling_mplug_docow2.py patch_positions:', patch_positions)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_ids, modality_indicators, attention_mask, past_key_values, inputs_embeds, labels = \
            self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, patch_positions)
        # ic(inputs_embeds.shape, labels.shape)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            modality_indicators=modality_indicators,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        # ic(outputs[0].shape)

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # ic(loss.shape)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "patch_positions": kwargs.get("patch_positions", None),
            }
        )
        return model_inputs

    def chat(self, messages, images, tokenizer):
        streamer = JapaneseTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # 日本語文書特有の前処理を追加
        image_tensor, patch_positions, input_ids = self.japanese_processor(
            images=images,
            messages=messages,
            vertical_text=True,  # 縦書きテキストの処理
            ruby_text=True      # ルビの処理
        )
        
        # 日本語用のストップワード
        stopping_criteria = KeywordsStoppingCriteria(
            ["</s>", "。", "！", "？"], 
            tokenizer,
            input_ids
        )
        
        with torch.inference_mode():
            output_ids = self.generate(
                input_ids,
                images=image_tensor,
                patch_positions=patch_positions,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=1024,  # 日本語の文章長に合わせて調整
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
        )

class JapaneseDocOwlInfer():
    def __init__(self, ckpt_path):
        # 日本語学習済みモデルの読み込み
        self.base_model = AutoModel.from_pretrained(
            "rinna/japanese-gpt-neox-3.6b-instruction-ppo",
            trust_remote_code=True,
            device_map='auto'
        )
        
        # DocOwl2の視覚エンコーダーの読み込み
        self.vision_model = UlusageDocDet2.from_pretrained(
            ckpt_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map='auto'
        ).get_vision_tower()
        
        # 日本語トークナイザーの初期化
        self.tokenizer = AutoTokenizer.from_pretrained(
            "rinna/japanese-gpt-neox-3.6b-instruction-ppo",
            use_fast=False,
            trust_remote_code=True
        )
        
        # プロセッサの初期化
        self.vision_model.init_processor(
            tokenizer=self.tokenizer,
            basic_image_size=504,
            crop_anchors='grid_12'
        )
    
    def inference(self, images, query):
        # 日本語用のストップワードを追加
        stopping_criteria = ["</s>", "。", "！", "？"]
        
        # メッセージの作成
        messages = [{
            'role': 'USER',
            'content': '<|image|>' * len(images) + query
        }]
        
        # 推論の実行
        answer = self.model.chat(
            messages=messages,
            images=images,
            tokenizer=self.tokenizer,
            stopping_criteria=stopping_criteria
        )
        
        return answer

class JapaneseUlusageDocDet2(UlusageDocDet2):
    def __init__(self, config):
        super().__init__(config)
        
        # 日本語トークナイザーの設定を反映
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
        
        # 日本語特有の処理のための追加レイヤー
        self.japanese_specific = nn.ModuleDict({
            'vertical_text': nn.Linear(config.hidden_size, config.hidden_size),
            'ruby_text': nn.Linear(config.hidden_size, config.hidden_size)
        })
        
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        patch_positions: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 親クラスのforward処理を呼び出し
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            patch_positions=patch_positions,
            return_dict=return_dict,
        )
        
        # 日本語特有の処理を追加
        if output_hidden_states:
            hidden_states = outputs.hidden_states[-1]
            hidden_states = self.japanese_specific['vertical_text'](hidden_states)
            hidden_states = self.japanese_specific['ruby_text'](hidden_states)
            outputs.hidden_states = outputs.hidden_states[:-1] + (hidden_states,)
            
        return outputs

    def chat(self, messages, images, tokenizer):
        # 日本語用のストップワードを設定
        stopping_criteria = KeywordsStoppingCriteria(
            ["</s>", "。", "！", "？"], 
            tokenizer, 
            input_ids
        )
        
        # 親クラスのchat処理を呼び出し
        outputs = super().chat(
            messages=messages,
            images=images,
            tokenizer=tokenizer
        )
        
        return outputs

class JapaneseSpecificLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vertical_text_embedding = nn.Embedding(2, config.hidden_size)
        self.ruby_processor = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, x, is_vertical=False, has_ruby=False):
        if is_vertical:
            x = x + self.vertical_text_embedding(torch.ones_like(x[:, :, 0]))
        if has_ruby:
            x = self.ruby_processor(x)
        return x

class JapaneseLayoutProcessor(nn.Module):
    def __init__(self, hidden_size, language_hidden_size):
        super().__init__()
        self.layout_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, language_hidden_size)
        )
        
    def forward(self, x):
        return self.layout_encoder(x)

AutoConfig.register("ulusage_docdet", UlusageDocDetConfig)
AutoModelForCausalLM.register(UlusageDocDetConfig, UlusageDocDet2)
    