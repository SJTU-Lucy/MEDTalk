from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.file_utils import ModelOutput
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput


# the implementation of Wav2Vec2Model is borrowed from https://huggingface.co/transformers/_modules/transformers/models/wav2vec2/modeling_wav2vec2.html#Wav2Vec2Model
# initialize our encoder with the pre- trained wav2vec 2.0 weights.
def _compute_mask_indices(
        shape: Tuple[int, int],
        mask_prob: float,
        mask_length: int,
        attention_mask: Optional[torch.Tensor] = None,
        min_masks: int = 0,
) -> np.ndarray:
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )
    all_num_mask = max(min_masks, all_num_mask)
    mask_idcs = []
    padding_mask = attention_mask.ne(1) if attention_mask is not None else None
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        lengths = np.full(num_mask, mask_length)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        min_len = min(lengths)
        if sz - min_len <= num_mask:
            min_len = sz - num_mask - 1

        mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)
        mask_idc = np.asarray([mask_idc[j] + offset for j in range(len(mask_idc)) for offset in range(lengths[j])])
        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True
    return mask


# linear interpolation layer
def linear_interpolation(features, seq_len):
    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=seq_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


class Wav2Vec2Model(Wav2Vec2Model):
    def __init__(self, config):
        super().__init__(config)

    def forward(
            self,
            input_values,
            seq_len,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        self.config.output_attentions = True
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # input_values: <batch_size, samples(sampling_rate: 16000)>
        # hidden_states: <batch_size, feature_dim, seq_len(fps:49.9hz)>
        hidden_states = self.feature_extractor(input_values)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = linear_interpolation(hidden_states, seq_len=seq_len)

        hidden_states = self.feature_projection(hidden_states)[0] # feature size from 512 to 768

        if self.config.apply_spec_augment and self.training:
            batch_size, sequence_length, hidden_size = hidden_states.size()
            if self.config.mask_time_prob > 0:
                mask_time_indices = _compute_mask_indices(
                    (batch_size, sequence_length),
                    self.config.mask_time_prob,
                    self.config.mask_time_length,
                    attention_mask=attention_mask,
                    min_masks=2,
                )
                hidden_states[torch.from_numpy(mask_time_indices)] = self.masked_spec_embed.to(hidden_states.dtype)
            if self.config.mask_feature_prob > 0:
                mask_feature_indices = _compute_mask_indices(
                    (batch_size, hidden_size),
                    self.config.mask_feature_prob,
                    self.config.mask_feature_length,
                )
                mask_feature_indices = torch.from_numpy(mask_feature_indices).to(hidden_states.device)
                hidden_states[mask_feature_indices[:, None].expand(-1, sequence_length, -1)] = 0
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs.hidden_states[0]

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
            frame_num=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            seq_len=frame_num,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states_1 = hidden_states
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        return SpeechClassifierOutput(
            logits=logits,
            hidden_states=hidden_states_1,
            attentions=outputs.attentions,
        )
