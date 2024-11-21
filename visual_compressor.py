import math
from typing import Any, Optional, Tuple, Union

from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from icecream import ic

from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func
from einops import rearrange


class UlusageDocDetVisualMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        in_features = config.high_reso_cross_hid_size
        self.act = nn.SiLU()

        ffn_hidden_size = int(2 * 4 * in_features / 3)
        multiple_of = 256
        ffn_hidden_size = multiple_of * ((ffn_hidden_size + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(in_features, ffn_hidden_size)
        self.w2 = nn.Linear(ffn_hidden_size, in_features)
        self.w3 = nn.Linear(in_features, ffn_hidden_size)
        self.ffn_ln = nn.LayerNorm(ffn_hidden_size, eps=config.layer_norm_eps)

        torch.nn.init.zeros_(self.w1.bias.data)
        torch.nn.init.zeros_(self.w2.bias.data)
        torch.nn.init.zeros_(self.w3.bias.data)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act(self.w1(hidden_states)) * self.w3(hidden_states)
        hidden_states = self.ffn_ln(hidden_states)
        hidden_states = self.w2(hidden_states)
        return hidden_states


class FlashCrossAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()
        
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v, **kwargs):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)

            or

            q: (Sum_q, H, D), k,v : (Sum_k, H, D), 
            must with batch_size, max_seqlen_q, max_seqlen_k, cu_seqlens_q, cu_seqlens_k in kwargs
        """

        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q,k,v)))
        assert all((i.is_cuda for i in (q,k,v)))


        if q.dim() == 4:
            batch_size, seqlen_q = q.shape[0], q.shape[1]
            q = rearrange(q, 'b s ... -> (b s) ...')
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                        device=q.device)
        else:
            batch_size, seqlen_q = kwargs['batch_size'], kwargs['max_seqlen_q']
            cu_seqlens_q = kwargs['cu_seqlens_q']

        if k.dim() == 4:
            seqlen_k = k.shape[1]
            k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [k, v]]
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                        device=q.device)
        else:
            seqlen_k = kwargs['max_seqlen_k']
            cu_seqlens_k = kwargs['cu_seqlens_k']

        # q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        # self.dropout_p = 0
        
        """print('FlashCrossAttention: q.shape:', q.shape)
        print('FlashCrossAttention: k.shape:', k.shape)
        print('FlashCrossAttention: v.shape:', v.shape)
        print('FlashCrossAttention: cu_seqlens_q:', cu_seqlens_q)
        print('FlashCrossAttention: cu_seqlens_k:', cu_seqlens_k)"""

        # print('visual_compressor.py q.shape:', q.shape, ' k.shape:', k.shape, ' v.shape:', v.shape)
        output = flash_attn_unpadded_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
            self.dropout_p if self.training else 0.0,
            softmax_scale=self.softmax_scale, causal=False
        )

        if q.dim() == 4: # keep the shape of output shape same as the input query
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        return output


class UlusageDocDetVisualMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.high_reso_cross_hid_size % config.high_reso_cross_num_att_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.high_reso_cross_hid_size, config.high_reso_cross_num_att_heads)
            )
        if config.high_reso_cross_hid_size // config.high_reso_cross_num_att_heads > 256:
            raise ValueError(
                "The hidden size of each head (%d) > 256 and is illegal for flash attention"
                % (config.high_reso_cross_hid_size // config.high_reso_cross_num_att_heads)
            )
        

        self.num_attention_heads = config.high_reso_cross_num_att_heads
        self.attention_head_size = int(config.high_reso_cross_hid_size / config.high_reso_cross_num_att_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.high_reso_cross_hid_size, self.all_head_size)
        self.key = nn.Linear(config.high_reso_cross_hid_size, self.all_head_size)
        self.value = nn.Linear(config.high_reso_cross_hid_size, self.all_head_size)
        self.core_attention_flash = FlashCrossAttention(attention_dropout=config.high_reso_cross_dropout)

        # bias init
        torch.nn.init.zeros_(self.query.bias.data)
        torch.nn.init.zeros_(self.key.bias.data)
        torch.nn.init.zeros_(self.value.bias.data)
    
    def transpose_for_scores(self, x):
        # [B, S, D] -> [B, S, H, D] or [Sum_S, D] -> [Sum_S, H, D]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        **kwargs
    ):
        # assert not torch.isnan(hidden_states).any()
        # assert not torch.isnan(encoder_hidden_states).any()

        key = self.transpose_for_scores(self.key(encoder_hidden_states))
        value = self.transpose_for_scores(self.value(encoder_hidden_states))
        query = self.transpose_for_scores(self.query(hidden_states))
        # print('visual_compressor.py key(after projection): ', key.shape, key)
        # print('visual_compressor.py value(after projection): ', value.shape, value)
        # print('visual_compressor.py query(after projection): ', query.shape, query)
        # assert not torch.isnan(key).any()
        # assert not torch.isnan(value).any()
        # assert not torch.isnan(query).any()
        outputs = self.core_attention_flash(q=query, k=key, v=value, **kwargs)
        outputs = rearrange(outputs, 's h d -> s (h d)').contiguous()
        # print('visual_compressor.py outputs(after cross_att): ', outputs.shape, outputs)
        return outputs


class UlusageDocDetVisualCrossOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.high_reso_cross_hid_size
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = UlusageDocDetVisualMLP(config)

        # bias init
        torch.nn.init.zeros_(self.out_proj.bias.data)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = input_tensor + self.out_proj(hidden_states)
        input_tensor = input_tensor + self.mlp(self.norm2(input_tensor))
        return input_tensor


class UlusageDocDetVisualCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = UlusageDocDetVisualMultiHeadAttention(config)
        self.output = UlusageDocDetVisualCrossOutput(config)
        self.norm1 = nn.LayerNorm(config.high_reso_cross_hid_size)
        self.normk = nn.LayerNorm(config.high_reso_cross_hid_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        # print('visual_compressor.py hidden_states: ', hidden_states.shape, hidden_states)
        # print('visual_compressor.py encoder_hidden_states: ', encoder_hidden_states.shape, encoder_hidden_states)
        # assert not torch.isnan(hidden_states).any()
        # assert not torch.isnan(encoder_hidden_states).any()
        hidden_states = self.norm1(hidden_states)
        encoder_hidden_states = self.normk(encoder_hidden_states)
        # print('visual_compressor.py hidden_states(after norm): ', hidden_states.shape, hidden_states)
        # print('visual_compressor.py encoder_hidden_states(after norm): ', encoder_hidden_states.shape, encoder_hidden_states)
        attention_output = self.attention(
            hidden_states,
            encoder_hidden_states,
            **kwargs
        )

        outputs = self.output(attention_output, hidden_states)
        
        return outputs


class UlusageDocDetVisualCrossAttentionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_num = config.layer
        self.layers = nn.ModuleList(
            [UlusageDocDetVisualCrossAttentionLayer(config) for layer_idx in range(self.layer_num)]
        )
        self.gradient_checkpointing = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        for i in range(self.layer_num):
            layer_module = self.layers[i]
            layer_outputs = layer_module(
                hidden_states,
                encoder_hidden_states,
                **kwargs
            )
            hidden_states = layer_outputs

        return hidden_states


def ensemble_crop_feats(crop_feats, patch_positions, col_feat_num):
    """
    ensemble vision feats from different crops to a feature map according the position of the raw image
    crop_feats: [N_crop, Len_feat, D]
    patch_positions: [N_crop, 2], 2 == (rowl_index, col_index)
    col_feat_num: the feature num of a row in a crop image
    """
    assert crop_feats.size(0) == patch_positions.size(0)
    row_feats = []
    crop_row = torch.max(patch_positions[:,0])+1 # 
    crop_feats = rearrange(crop_feats, '(R C) L D -> R C L D', R=crop_row) # [N_crop_row, N_crop_col, Len_feat, D]
    crop_feats = rearrange(crop_feats, 'R C (X Y) D-> R C X Y D', Y=col_feat_num) # [N_crop_row, N_crop_col, Len_row_feat, Len_col_feat, D]
    # 1. concatenate same row feats across crops; 2. ensemble row feats to get 1 feature map
    hw_feats = rearrange(crop_feats, 'R C X Y D-> (R X) (C Y) D') # [N_crop_row x Len_row_feat, N_crop_col x Len_col_feat, D]

    return hw_feats

def group_window_feats(feats, window):
    """
    collect vision feats from a window (win_row, win_col) to 1 group
    feats: [H, W, D]
    window: (win_row, win_col)
    
    return: [H/win_row, H/win_col, win_row x win_col, D]
    """

    group_feats = rearrange(feats, '(X R) (Y C) D -> (X Y) (R C) D', R=window[0], C=window[1]) # [H/win_row x H/win_col, win_row x win_col, D]
    return group_feats
    
    
def distinguish_global_crop_features(hidden_states, patch_positions, reorganize_crop_feats=True, col_feat_num=None, group_feats_by_crop_shape=False, keep_row_col=False):
    """
    distinguish global and crop features with the help of patcg_positions
    # hidden_states: [B, s+1, h] 
    # (B is the sum of cropped num across samples in a micro_batch, s is the visual tokens, +1 means the vit end token)
    # patch_positions: [B, 2], 
    # 2 == (rowl_index, col_index), the first crop is (0,0), global img is (anchor_max, anchor_max)

    col_feat_num is used when reorganize_crop_feats == True

    outputs:
    img_global_features: list of [Len_global_feat, D]
    img_crop_features: list of [Len_global_feat, D]
    """
    hidden_states = hidden_states[:, :-1, :] # remove the last vit end token emb
    # the first crop is (0,0)
    first_crop_indices = (patch_positions.sum(dim=-1) == 0).nonzero().squeeze(1) # Num_img
    # the global image is before the first crop
    global_indices = first_crop_indices - 1 # Num_img
    # print('vision2text_model.py patch_positions:', patch_positions)
    # print('vision2text_model.py global_indices:', global_indices)
    # collect cropped vision features of an identical image
    batch_size = hidden_states.size(0)
    img_global_features = []
    img_crop_features = [] # store list of Num_crop (variable) x Len_feat (fixed)
    img_crop_positions = [] # store list of Num_crop (variable) x 2 
    for i in range(len(global_indices)):
        index = global_indices[i]
        img_global_features.append(hidden_states[index])
        if i == (len(global_indices)-1):
            img_crop_features.append(hidden_states[index+1:])
            img_crop_positions.append(patch_positions[index+1:])
        else:
            next_index = global_indices[i+1]
            img_crop_features.append(hidden_states[index+1:next_index])
            img_crop_positions.append(patch_positions[index+1:next_index])
    
    if reorganize_crop_feats:
        for i in range(len(img_crop_features)):
            img_crop_features[i] = ensemble_crop_feats(img_crop_features[i], img_crop_positions[i], col_feat_num) # [H W D]
            if group_feats_by_crop_shape: # collect vision feats from a window (crop_row, crop_col) to 1 group
                crop_row = torch.max(img_crop_positions[i][:,0])+1 # 
                crop_col = torch.max(img_crop_positions[i][:,1])+1 # 
                img_crop_features[i] =  group_window_feats(img_crop_features[i], window=(crop_row, crop_col)) # [H/crop_row x W/crop_col, crop_row x crop_row, D]
            else:
                # img_crop_features = [rearrange(x, 'H W D -> (H W) D') for x in img_crop_features]
                if not keep_row_col:
                    img_crop_featuress[i] = rearrange(img_crop_featuress[i], 'H W D -> (H W) D')
    else:
        img_crop_features = [rearrange(x, 'N L D -> (N L) D') for x in img_crop_features]

    return img_global_features, img_crop_features

            
class UlusageDocDetHRDocCompressor(PreTrainedModel):
    """
    After vision-to-text module, use low-resolution global features to select high-resolution crop features with cross-attention
    the key/value from high-resolution crop features are contrained in a window size
    positions of the features within the window in raw images are the same as the global query features
    """
    def __init__(self, config, output_hidden_size, v2t_img_col_tokens):
        super().__init__(config)
        self.use_flash_attn = True
        assert self.use_flash_attn

        self.v2t_img_col_tokens = v2t_img_col_tokens

        self.compressor_crossatt = UlusageDocDetVisualCrossAttentionEncoder(config)

        self.compressor_fc = torch.nn.Linear(output_hidden_size, output_hidden_size)

        self.compressor_eos = torch.nn.Parameter(torch.randn(1, 1, output_hidden_size))

    
    def forward(self, hidden_states,  patch_positions=None):
        # hidden_states: outputs of vision2textmodel: [Sum(crop), s+1, h] 
        # (Sum(crop) is the sum of cropped num across samples in a micro_batch, s is the visual tokens, +1 is the special vit_eos token added in H-Reducer)
        # patch_positions: [Sum(crop), 2]

        # print('visual_compressor.py HRDocCompressor hidden_states.shape:', hidden_states.shape)
        # print('visual_compressor.py HRDocCompressor patch_positions.shape:', patch_positions.shape)
        
        # N_img x [L_global (fixed), D], N_img x [L_global (fixed), Crop_row x Crop_Col (Variable), D]
        img_global_features, img_crop_features = distinguish_global_crop_features(hidden_states, 
                                                patch_positions, 
                                                reorganize_crop_feats=True, 
                                                col_feat_num=self.v2t_img_col_tokens,
                                                group_feats_by_crop_shape=True)

        # cross-attention to accumulate high-resolution features
        # if self.use_flash_attn: # flash_attn_varlen_func don't need to pad crop_features
        img_global_features = torch.stack(img_global_features, dim=0).to(hidden_states.device) # Num_img x Len_global_feat x D
        batch_size, global_feat_num, seqlen_q = img_global_features.shape[0], img_global_features.shape[1], 1
        img_global_features = rearrange(img_global_features, 'b s ... -> (b s) ...')
        cu_seqlens_q = torch.arange(0, batch_size*global_feat_num+1, step=1, dtype=torch.int32, device=img_global_features.device) # # (Num_img x Len_global_feat +1, )
        cu_seqlens_k = [0]
        max_seqlens_k = 0
        for crop_feat in img_crop_features:
            for i in range(crop_feat.shape[0]): 
                cu_seqlens_k.append(cu_seqlens_k[-1]+crop_feat.shape[1]) # same k within a image shares the seq len
            max_seqlens_k = max(max_seqlens_k, crop_feat.size(1))

        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32).to(hidden_states.device) # (Num_img x Len_global_feat+1, )
        # cu_seqlens_k = torch.arange(0, (batch_size + 1) * max_seqlens_k, step=max_seqlens_k, dtype=torch.int32, device=img_global_features.device) # # (Num_img+1, )

        img_crop_features = torch.cat([rearrange(x, 'N L D -> (N L) D') for x in img_crop_features], dim=0).to(hidden_states.device) # Sum(L_hr) x D
        flash_kwargs = {
            'batch_size': batch_size*global_feat_num, # each feat in global feats use different keys
            'max_seqlen_q': seqlen_q, # key are unique for each query
            'max_seqlen_k': max_seqlens_k,
            'cu_seqlens_q': cu_seqlens_q, # the seq len of each q
            'cu_seqlens_k': cu_seqlens_k # the seq len of each k
        }
        # print('visual_compressor.py HRDocCompressor img_global_features.shape:', img_global_features.shape, img_global_features)
        # print('visual_compressor.py HRDocCompressor img_crop_features.shape:', img_crop_features.shape, img_crop_features)
        """print('visual_compressor.py HRDocCompressor cu_seqlens_q, cu_seqlens_q.shape:', cu_seqlens_q, cu_seqlens_q.shape)
        print('visual_compressor.py HRDocCompressor cu_seqlens_k, cu_seqlens_k.shape:', cu_seqlens_k, cu_seqlens_k.shape)"""
        # assert not torch.isnan(img_global_features).any()
        # assert not torch.isnan(img_crop_features).any()
        for x_name, x in self.compressor_crossatt.named_parameters():
            try:
                assert not torch.isnan(x).any()
                # print('visual_compressor.py ', x_name, x.shape, x)
            except Exception as e:
                print(e)
                print('visual_compressor.py nan', x_name, x.shape, x)
        hidden_states = self.compressor_crossatt(
                img_global_features.contiguous(), # Sum(L_global) x D
                img_crop_features.contiguous(),  # Sum(L_hr) x D
                **flash_kwargs
            ) # Sum(L_global) x D
        hidden_states = rearrange(hidden_states, '(B S) D -> S B D', B=batch_size) # L_global x N_img x D

        hidden_states = self.compressor_fc(hidden_states) # L_global x N_img x D

        hidden_states = hidden_states.transpose(0, 1).contiguous() # N_img x L_global x D
        # print('visual_compressor.py hidden_states:', hidden_states.shape)

        hidden_states = torch.cat([hidden_states, self.compressor_eos.repeat(hidden_states.shape[0], 1, 1)], dim=1) # N_img x (L_global+1) x D
        # print('visual_compressor.py HRDocCompressor hidden_states.shape:', hidden_states.shape)

        return hidden_states