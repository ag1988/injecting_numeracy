# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
import sys
from io import open
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from allennlp.nn import util
from allennlp.models.reading_comprehension.util import get_best_span

from pytorch_pretrained_bert.file_utils import cached_path, WEIGHTS_NAME, CONFIG_NAME

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
}
PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-config.json",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-config.json",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-config.json",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-config.json",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-config.json",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json",
    'bert-base-german-cased': "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-config.json",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-config.json",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-config.json",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-config.json",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-config.json",
}
BERT_CONFIG_NAME = 'bert_config.json'
TF_WEIGHTS_NAME = 'model.ckpt'

def prune_linear_layer(layer, index, dim=0):
    """ Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())

# try:
#     from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
# except ImportError:
#     logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
# BertLayerNorm = nn.LayerNorm

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias        

    
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, random_shift=False):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        assert len(position_ids.size()) == 2
        if random_shift:
            shift = torch.randint(512 - seq_length + 1, size=(input_ids.size(0),1), device=input_ids.device)
            position_ids = position_ids + shift.to(dtype=position_ids.dtype)
          
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class BertSelfAttention(nn.Module):
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = output_attentions
        self.keep_multihead_output = keep_multihead_output
        self.multihead_output = None

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  #[b, n_heads, seq_len, head_size]

    def forward(self, query, key, value, attention_mask, head_mask=None):
        # attention mask: [b, n_heads, query_seq_len, key_seq_len] ([b,1,1,max_seq_len])
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # query_layer: [b, n_heads, query_seq_len, head_size] 
        # key_layer.transpose(-1, -2): [b, n_heads, head_size, key_seq_len]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # attention_scores: [b, n_heads, query_seq_len, key_seq_len]
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities
        # attention_probs: [b, n_heads, query_seq_len, key_seq_len]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        # attention_probs: [b, n_heads, query_seq_len, key_seq_len]
        # value_layer:     [b, n_heads, key_seq_len, head_size]
        context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer:   [b, n_heads, query_seq_len, head_size]
        if self.keep_multihead_output:
            self.multihead_output = context_layer
            self.multihead_output.retain_grad()

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()       # [b, query_seq_len, n_heads, head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # [b, query_seq_len, hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)
        if self.output_attentions:
            return attention_probs, context_layer
        return context_layer  # [b, query_seq_len, hidden_size]


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(BertAttention, self).__init__()
        self.output_attentions = output_attentions
        self.self = BertSelfAttention(config, output_attentions=output_attentions,
                                              keep_multihead_output=keep_multihead_output)
        self.output = BertSelfOutput(config)
        
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # Update hyper params
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

    def forward(self, input_tensor, attention_mask, head_mask=None, memory_tensor=None, src_attention_mask=None):
        # self attention
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask, head_mask)
        if self.output_attentions:
            attentions, self_output = self_output
        attention_output = self.output(self_output, input_tensor)
        if memory_tensor is None:
            if self.output_attentions:
                return attentions, attention_output
            return attention_output
        
        # else decoder layer
        # resuing the attention layer and output layer for target to source attn in Decoder layer as 
        # the same as that in target self attn to avoid errors while loading from pre-trained
        self_output = self.self(attention_output, memory_tensor, memory_tensor, src_attention_mask, head_mask)
        if self.output_attentions:
            _, self_output = self_output
        attention_output = self.output(self_output, input_tensor)
        if self.output_attentions:
            return attentions, attention_output # attentions are that of self attn and not src attn
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(BertLayer, self).__init__()
        self.output_attentions = output_attentions
        self.attention = BertAttention(config, output_attentions=output_attentions,
                                               keep_multihead_output=keep_multihead_output)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None, memory_tensor=None, src_attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask, head_mask, memory_tensor, src_attention_mask)
        if self.output_attentions:
            attentions, attention_output = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if self.output_attentions:
            return attentions, layer_output
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(BertEncoder, self).__init__()
        self.output_attentions = output_attentions
        layer = BertLayer(config, output_attentions=output_attentions,
                                  keep_multihead_output=keep_multihead_output)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, head_mask=None, 
                memory_tensor=None, src_attention_mask=None):
        all_encoder_layers = []
        all_attentions = []
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask, head_mask[i], memory_tensor, src_attention_mask)
            if self.output_attentions:
                attentions, hidden_states = hidden_states
                all_attentions.append(attentions)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        if self.output_attentions:
            return all_attentions, all_encoder_layers
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                    . `bert-base-german-cased`
                    . `bert-large-uncased-whole-word-masking`
                    . `bert-large-cased-whole-word-masking`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        cache_dir = kwargs.get('cache_dir', None)
        kwargs.pop('cache_dir', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)

        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
            config_file = PRETRAINED_CONFIG_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            if from_tf:
                # Directly load from a TensorFlow checkpoint
                archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME)
                config_file = os.path.join(pretrained_model_name_or_path, BERT_CONFIG_NAME)
            else:
                archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
                logger.error(
                    "Couldn't reach server at '{}' to download pretrained weights.".format(
                        archive_file))
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name_or_path,
                        ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                        archive_file))
            return None
        try:
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in PRETRAINED_CONFIG_ARCHIVE_MAP:
                logger.error(
                    "Couldn't reach server at '{}' to download pretrained model configuration file.".format(
                        config_file))
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name_or_path,
                        ', '.join(PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
                        config_file))
            return None
        if resolved_archive_file == archive_file and resolved_config_file == config_file:
            logger.info("loading weights file {}".format(archive_file))
            logger.info("loading configuration file {}".format(config_file))
        else:
            logger.info("loading weights file {} from cache at {}".format(
                archive_file, resolved_archive_file))
            logger.info("loading configuration file {} from cache at {}".format(
                config_file, resolved_config_file))
        # Load config
        config = BertConfig.from_json_file(resolved_config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        return model


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model
        `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
        `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
            This can be used to compute head importance metrics. Default: False

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
        `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
            It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.


    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(BertModel, self).__init__(config)
        self.output_attentions = output_attentions
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config, output_attentions=output_attentions,
                                           keep_multihead_output=keep_multihead_output)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_multihead_outputs(self):
        """ Gather all multi-head outputs.
            Return: list (layers) of multihead module outputs with gradients
        """
        return [layer.attention.self.multihead_output for layer in self.encoder.layer]

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, 
                output_all_encoded_layers=False, head_mask=None,
                random_shift=False, memory_tensor=None, src_attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        dtype = next(self.parameters()).dtype  # for fp16 compatibility
        
        def extend(mask, is_decoder=False):
            extended_mask = mask.unsqueeze(1).unsqueeze(2).to(dtype=dtype) # [bsz, 1, 1, key_seq_len]
            assert len(extended_mask.size()) == 4
            # can broadcast to [bsz, n_heads, query_seq_len, key_seq_len]
            if is_decoder:
                # apply subsequent mask to decoder self-attn to prevent lookahead
                key_seq_len = extended_mask.size(-1)
                extended_mask = extended_mask.expand(-1, -1, key_seq_len, -1)
                # [bsz, 1, key_seq_len, key_seq_len]
                subsequent_mask = torch.triu(torch.ones_like(extended_mask)).transpose(2, 3)
                extended_mask = extended_mask * subsequent_mask
                
            # mask should be 1.0 for positions we want to attend and 0.0 for others.
            extended_mask = (1.0 - extended_mask) * -10000.0
            return extended_mask
        
        # extended_attention_mask : [bsz, n_heads, query_seq_len, key_seq_len]
        extended_attention_mask = extend(attention_mask, memory_tensor is not None)
        extended_src_attention_mask = None if src_attention_mask is None else extend(src_attention_mask)
        
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand_as(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, token_type_ids, random_shift)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      head_mask=head_mask,
                                      memory_tensor=memory_tensor,
                                      src_attention_mask=extended_src_attention_mask)
        if self.output_attentions:
            all_attentions, encoded_layers = encoded_layers
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        if self.output_attentions:
            return all_attentions, encoded_layers, pooled_output
        return encoded_layers, pooled_output


class BertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, 
                masked_lm_labels=None, random_shift=False, ignore_idx=0):
        outputs = self.bert(input_ids, token_type_ids, attention_mask, random_shift=random_shift)
        sequence_output, _ = outputs
        prediction_scores = self.cls(sequence_output)
        preds = prediction_scores.argmax(dim=-1)                       # [bsz, seq_len]
        
        if masked_lm_labels is not None:
            errors = ((preds != masked_lm_labels) * (masked_lm_labels != ignore_idx)).sum(dim=-1)
            loss_fct = CrossEntropyLoss(ignore_index=ignore_idx)
            prediction_scores = prediction_scores.view(-1, self.config.vocab_size)
            masked_lm_loss = loss_fct(prediction_scores, masked_lm_labels.view(-1))
            return masked_lm_loss, errors
        return preds


class SimpleMLMHead(nn.Module):
    '''used for simple decoding based on similarity with output embeddings.
    '''
    def __init__(self, embed):
        super(SimpleMLMHead, self).__init__()
        # embed : bert.embeddings.word_embeddings
        # output wts are the same as input embeds, but there's an output-only bias for each tok.
        self.vocab_size = embed.weight.size(0)
        self.decoder = nn.Linear(embed.weight.size(1), embed.weight.size(0), bias=False)
        self.decoder.weight = embed.weight
        self.bias = nn.Parameter(torch.zeros(embed.weight.size(0)))
        
    def forward(self, sequence_output, labels=None, ignore_idx=0, include=None, sample_wise=False):
        prediction_scores = self.decoder(sequence_output) + self.bias  # [bsz, seq_len, vocab]
        preds = prediction_scores.argmax(dim=-1)                       # [bsz, seq_len]
        if labels is None:
            return preds
        # labels : [bsz, seq_len]
        include = torch.ones_like(labels[:, 0]) if include is None else include
        errors = ((preds != labels) & (labels != ignore_idx)).sum(dim=-1) * include
        loss_fct = CrossEntropyLoss(ignore_index=ignore_idx, reduction='none' if sample_wise else 'mean')
        prediction_scores = prediction_scores.view(-1, self.vocab_size)
        nlls = loss_fct(prediction_scores, labels.contiguous().view(-1))
        if not sample_wise:
            loss = nlls                                       # mean already applied
            return loss, errors
        else:
            element_wise_nll = nlls.view(*labels.size())      # 0.0 for ignored indices
            sample_wise_nll = element_wise_nll.sum(dim=-1) * include.float()
            return sample_wise_nll, errors                    # [bsz], [bsz]


class BertTransformer(BertPreTrainedModel):
    '''Bert encoder + Bert decoder with extra mlm, span extraction, etc heads.
    TODO: for exractive QA, include BIO tagging head.
    '''
    def __init__(self, config):
        super(BertTransformer, self).__init__(config)
        self.bert = BertModel(config)
        # encoder, decoder share bert - so we apply two separate layers to distinguish
        self.extra_enc_layer = BertPredictionHeadTransform(config) # extra encoder specific layer on top of bert
        self.embed = self.bert.embeddings.word_embeddings
        self.qa_head = SpanExtractionHead(config)                  # for span exraction
        self.extra_dec_layer = BertPredictionHeadTransform(config) # extra decoder specific layer on top of bert
        self.head_type = nn.Linear(config.hidden_size, 2)          # to decide which head to use: decoder / extract span
        self.dec_head = SimpleMLMHead(self.embed)                  # for decoding the decoder output
        self.cls = BertOnlyMLMHead(config, self.embed.weight)      # for mlm task: extra linear + transform 
                                                                   # as in orig bert pre-training 
        self.apply(self.init_bert_weights)
        # Module.apply() applies init_bert_weights recursively on all submodules.
        # In from_pretrained() code, first the model constructor is called and then model weights are loaded from the file
        
    def mlm_task(self, input_ids, token_type_ids=None, input_mask=None, 
                 masked_lm_labels=None, random_shift=False, ignore_idx=0):
        # this task uses bert output and not the encoder output
        # implementation is same as orig bert
        input_mask = self.mask(input_ids)
        sequence_output, _ = self.bert(input_ids, token_type_ids, input_mask)
        prediction_scores = self.cls(sequence_output)
        preds = prediction_scores.argmax(dim=-1)                       # [bsz, seq_len]
        
        if masked_lm_labels is not None:
            errors = ((preds != masked_lm_labels) & (masked_lm_labels != ignore_idx)).sum(dim=-1)
            loss_fct = CrossEntropyLoss(ignore_index=ignore_idx)
            prediction_scores = prediction_scores.view(-1, self.config.vocab_size)
            masked_lm_loss = loss_fct(prediction_scores, masked_lm_labels.view(-1))
            return masked_lm_loss, errors
        return preds

    def encode(self, input_ids, token_type_ids=None, input_mask=None, random_shift=False):
        # passes the input through bert and applies the extra encoder layer
        outputs = self.bert(input_ids, token_type_ids, self.mask(input_ids, input_mask), random_shift=random_shift)
        sequence_output, _ = outputs
        encoder_output = self.extra_enc_layer(sequence_output)
        pooled_encoder_output = encoder_output[:, 0]                   # [CLS] output
        return encoder_output, pooled_encoder_output
    
    @staticmethod
    def mask(ids, mask=None):
        # assumes pad id == 0
        return (ids != 0) if mask is None else mask.bool()
    
    @staticmethod
    def get1d(x):
        # multi-gpu might add an extra dim
        if x is not None and len(x.size()) > 1:
            return x.squeeze(-1)
        return x
    
    def forward(self, input_ids, token_type_ids=None, input_mask=None, random_shift=False, 
                target_ids=None, target_mask=None, answer_as_question_spans=None, answer_as_passage_spans=None,
                head_type=None, ignore_idx=0, task=''):
        # answer_as_question_spans: [bsz, # answers, 2], -1: ignore
        # head_type: [bsz]  0 for decoder, 1 for span-extraction, -1: ignore 
        # segment_ids are only used for question, passage mask computation and not for encoding
        
        if task.lower() == 'mlm':
            return self.mlm_task(input_ids, None, input_mask, target_ids, ignore_idx=ignore_idx)
        
        if task.lower() == 'inference':
            return self.inference(input_ids, token_type_ids, input_mask)
        
        # encode
        input_mask = self.mask(input_ids, input_mask)
        encoder_output, pooled_output = self.encode(input_ids, None, input_mask, random_shift)
        
        only_generative_head = False
        if answer_as_question_spans is None:
            head_type, only_generative_head = torch.zeros_like(input_ids[:, 0]), True
        
        # decode
        target_ids_in, target_ids_out = target_ids[:, :-1], target_ids[:, 1:]
        # subsequent mask is applied inside self.bert
        dec_out, _ = self.bert(target_ids_in, attention_mask=self.mask(target_ids_in), 
                               memory_tensor=encoder_output, src_attention_mask=input_mask)
        
        # apply extra dec layer and compute the decoder losses wrt left-shifted target seq
        if only_generative_head:
            dec_loss, dec_errors = self.dec_head(self.extra_dec_layer(dec_out), target_ids_out,
                                                 ignore_idx=0)
            loss = dec_loss     # already mean reduced
            errs = dec_errors > 0
        else:
            # generative head can always generate the ans
            dec_nlls, dec_errors = self.dec_head(self.extra_dec_layer(dec_out), target_ids_out, 
                                                 ignore_idx=0, sample_wise=True)
            # dec_nlls, dec_errors : [bsz], [bsz]
            dec_log_probs = - dec_nlls
            dec_loss = dec_nlls.mean()
        
        if not only_generative_head:
            # extractive answer span: for samples without a valid span span_errors is 1
            span_log_probs, span_errors, span_loss, start_preds, end_preds = self.qa_head(
                encoder_output, input_mask, token_type_ids, answer_as_question_spans, answer_as_passage_spans)
            # span_errors, start_preds, end_preds : [bsz]
            # for samples without a valid gold span, log prob is a large -ve val,
            # this'll drive the head_type to choose the generative head.
            
            # answer head type
            type_logits = self.head_type(pooled_output)           # [bsz, 2]
            type_log_probs = nn.LogSoftmax(dim=-1)(type_logits)   # [bsz, 2]
            type_preds = type_logits.argmax(dim=-1)               # [bsz] 
            # compute loss for samples with head_type supervision
            head_type = self.get1d(head_type)                     # [bsz]
            type_loss = CrossEntropyLoss(ignore_index=-1)(type_logits, head_type) 
            type_errors = ((type_preds != head_type) & (head_type != -1))  # [bsz]
            
            # marginalize over head types
            log_probs_list = [dec_log_probs + type_log_probs[:,0], span_log_probs + type_log_probs[:,1]]
            all_log_probs = torch.stack(log_probs_list, dim=-1)
            marginal_log_probs = torch.logsumexp(all_log_probs, -1)
            loss = - marginal_log_probs.mean()
            errs = torch.gather(torch.stack([dec_errors > 0, span_errors], dim=-1), 
                                1, type_preds.unsqueeze(1)).squeeze(1)   # [bsz]
        else:
            type_preds = torch.zeros_like(dec_errors)     # [bsz]
            span_loss, span_errors, type_loss, type_errors = None, None, None, None
        
        return (loss, errs, dec_loss, dec_errors, span_loss, span_errors, type_loss, 
                type_errors, type_preds)
    
    
    def inference(self, input_ids, token_type_ids=None, input_mask=None, start_tok_id=1030, max_decoding_steps=20):
        # encode
        input_mask = self.mask(input_ids, input_mask)
        encoder_output, pooled_output = self.encode(input_ids, None, input_mask, random_shift=False)
        
        # decode
        # max_decoding_steps: drop: 20,  numeric syn data: 11
        start_ids = torch.zeros_like(input_ids[:,0]) + start_tok_id
        dec_out_ids = start_ids if start_ids.dim() > 1 else start_ids.unsqueeze(1) # [bsz, 1]
        
        for i in range(max_decoding_steps-1): # -1 as we included end_tok in max_decoding_steps
            # subsequent mask is applied inside self.bert
            dec_out, _ = self.bert(dec_out_ids, attention_mask=self.mask(dec_out_ids),
                                   memory_tensor=encoder_output, src_attention_mask=input_mask)
            # apply extra dec layer and extract the last time step pred
            dec_preds_i = self.dec_head(self.extra_dec_layer(dec_out))[:, -1:]
            # update decoded seq
            dec_out_ids = torch.cat((dec_out_ids, dec_preds_i), dim=-1)  # [bsz, i+2]
            
        # extractive answer span: only relevant errs are included in span_errors
        start_preds, end_preds = self.qa_head(encoder_output, input_mask, token_type_ids=token_type_ids)
        
        # answer head type
        type_logits = self.head_type(pooled_output)  # [bsz, 2]
        type_preds = type_logits.argmax(dim=-1)
        
        return dec_out_ids, type_preds, start_preds, end_preds, type_logits 
        # [bsz, max_decoding_steps], [bsz], [bsz],    [bsz],     [bsz, 2]


    
class SpanExtractionHead(nn.Module):
    '''Span extraction head for QA. Mostly borrowed from https://github.com/raylin1000/drop-bert/blob/master/drop_bert/augmented_bert.py .
    '''
    def __init__(self, config):
        super(SpanExtractionHead, self).__init__()
        bert_dim = config.hidden_size
        self.dropout = config.hidden_dropout_prob
        
        self._answer_ability_predictor = self.ff(2 * bert_dim, bert_dim, 2)
        self._passage_weights_predictor = torch.nn.Linear(bert_dim, 1)
        self._question_weights_predictor = torch.nn.Linear(bert_dim, 1)
        
        self._passage_span_start_predictor = torch.nn.Linear(bert_dim, 1)
        self._passage_span_end_predictor = torch.nn.Linear(bert_dim, 1)
        self._question_span_start_predictor = self.ff(2 * bert_dim, bert_dim, 1)
        self._question_span_end_predictor = self.ff(2 * bert_dim, bert_dim, 1)
        
    def ff(self, input_dim, hidden_dim, output_dim):
        return torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                                   torch.nn.ReLU(),
                                   torch.nn.Dropout(self.dropout),
                                   torch.nn.Linear(hidden_dim, output_dim))
    
    def summary_vector(self, encoding, mask, in_type="passage"):
        if in_type == "passage":
            # Shape: (batch_size, seqlen)
            alpha = self._passage_weights_predictor(encoding).squeeze()
        elif in_type == "question":
            # Shape: (batch_size, seqlen)
            alpha = self._question_weights_predictor(encoding).squeeze()
        # Shape: (batch_size, seqlen) 
        alpha = util.masked_softmax(alpha, mask)
        # Shape: (batch_size, hidden_size)
        return util.weighted_sum(encoding, alpha)

    def forward(self, sequence_out, input_mask, token_type_ids, 
                answer_as_question_spans=None, answer_as_passage_spans=None):
        # answer_as_question_spans: (bsz, # answer spans, 2) (ignore index is -1) 
        # Samples without a valid span are excluded from loss.
        
        # Shape: (batch_size, seqlen)
        passage_mask, question_mask = token_type_ids, (1-token_type_ids) * input_mask.long()
        # Shape: (batch_size, seqlen, bert_dim)
        passage_out = sequence_out
        del sequence_out
        # Shape: (batch_size, bert_dim)
        question_vector = self.summary_vector(passage_out, question_mask, "question")
        passage_vector = self.summary_vector(passage_out, passage_mask)
        
        # Shape: (batch_size, 2)
        answer_ability_logits = \
                self._answer_ability_predictor(torch.cat([passage_vector, question_vector], -1))
        answer_ability_log_probs = torch.nn.functional.log_softmax(answer_ability_logits, -1)
        # Shape: (batch_size, 1)
        best_answer_ability = torch.argmax(answer_ability_log_probs, 1).unsqueeze(1)
        
        passage_span_start_log_probs, passage_span_end_log_probs, best_passage_span = \
                self._passage_span_module(passage_out, passage_mask)

        question_span_start_log_probs, question_span_end_log_probs, best_question_span = \
                self._question_span_module(passage_vector, passage_out, question_mask)
            
        span_preds = (best_passage_span.float() * (1 - best_answer_ability.float())
                      + best_question_span.float() * best_answer_ability.float()).long()
        start_preds, end_preds = span_preds[:, 0], span_preds[:, 1]
        
        if answer_as_passage_spans is None or answer_as_question_spans is None:
            return start_preds, end_preds
        
        log_marginal_likelihood_list = []

        log_marginal_likelihood_for_passage_span = \
            self._span_log_likelihood(answer_as_passage_spans,
                                      passage_span_start_log_probs,
                                      passage_span_end_log_probs)
        log_marginal_likelihood_list.append(log_marginal_likelihood_for_passage_span)

        log_marginal_likelihood_for_question_span = \
            self._span_log_likelihood(answer_as_question_spans,
                                      question_span_start_log_probs,
                                      question_span_end_log_probs)
        log_marginal_likelihood_list.append(log_marginal_likelihood_for_question_span)
        
        all_log_marginal_likelihoods = torch.stack(log_marginal_likelihood_list, dim=-1)
        all_log_marginal_likelihoods = all_log_marginal_likelihoods + answer_ability_log_probs
        marginal_log_likelihood = util.logsumexp(all_log_marginal_likelihoods)
        
        # loss from samples without a valid answer span is a large -ve value
        has_valid_span = ((answer_as_question_spans != -1).all(-1).any(-1) 
                          | (answer_as_passage_spans != -1).all(-1).any(-1))
        loss = - (marginal_log_likelihood * has_valid_span.float()).mean()
        
        # log marginal is a large -ve val for samples without a valid span 
        # error is 0 for excluded samples and 1 for samples without a valid span
        start_errs = ((start_preds.unsqueeze(1) != answer_as_passage_spans[:,:,0]).all(dim=-1) 
                      & (start_preds.unsqueeze(1) != answer_as_question_spans[:,:,0]).all(dim=-1))
        end_errs = ((end_preds.unsqueeze(1) != answer_as_passage_spans[:,:,1]).all(dim=-1) 
                      & (end_preds.unsqueeze(1) != answer_as_question_spans[:,:,1]).all(dim=-1))
        # this is just a lower bound
        span_errs = (start_errs | end_errs)                         # [bsz]
        return marginal_log_likelihood, span_errs, loss, start_preds, end_preds

    def _passage_span_module(self, passage_out, passage_mask):
        # Shape: (batch_size, seq_length)
        passage_span_start_logits = self._passage_span_start_predictor(passage_out).squeeze(-1)

        # Shape: (batch_size, seq_length)
        passage_span_end_logits = self._passage_span_end_predictor(passage_out).squeeze(-1)

        # Shape: (batch_size, seq_length)
        passage_span_start_log_probs = util.masked_log_softmax(passage_span_start_logits, passage_mask)
        passage_span_end_log_probs = util.masked_log_softmax(passage_span_end_logits, passage_mask)

        # Info about the best passage span prediction
        passage_span_start_logits = util.replace_masked_values(passage_span_start_logits, passage_mask, -1e7)
        passage_span_end_logits = util.replace_masked_values(passage_span_end_logits, passage_mask, -1e7)

        # Shape: (batch_size, 2)
        best_passage_span = get_best_span(passage_span_start_logits, passage_span_end_logits)
        return passage_span_start_log_probs, passage_span_end_log_probs, best_passage_span
    
    def _question_span_module(self, passage_vector, question_out, question_mask):
        # Shape: (batch_size, seq_length)
        encoded_question_for_span_prediction = \
            torch.cat([question_out,
                       passage_vector.unsqueeze(1).repeat(1, question_out.size(1), 1)], -1)
        question_span_start_logits = \
            self._question_span_start_predictor(encoded_question_for_span_prediction).squeeze(-1)
        # Shape: (batch_size, seq_length)
        question_span_end_logits = \
            self._question_span_end_predictor(encoded_question_for_span_prediction).squeeze(-1)
        question_span_start_log_probs = util.masked_log_softmax(question_span_start_logits, question_mask)
        question_span_end_log_probs = util.masked_log_softmax(question_span_end_logits, question_mask)

        # Info about the best question span prediction
        question_span_start_logits = \
            util.replace_masked_values(question_span_start_logits, question_mask, -1e7)
        question_span_end_logits = \
            util.replace_masked_values(question_span_end_logits, question_mask, -1e7)

        # Shape: (batch_size, 2)
        best_question_span = get_best_span(question_span_start_logits, question_span_end_logits)
        return question_span_start_log_probs, question_span_end_log_probs, best_question_span
    
    def _span_log_likelihood(self, answer_as_spans, span_start_log_probs, span_end_log_probs):
        # Shape: (batch_size, # of answer spans)
        gold_span_starts = answer_as_spans[:, :, 0]
        gold_span_ends = answer_as_spans[:, :, 1]
        # Some spans are padded with index -1,
        # so we clamp those paddings to 0 and then mask after `torch.gather()`.
        gold_span_mask = (gold_span_starts != -1).long()
        clamped_gold_span_starts = \
            util.replace_masked_values(gold_span_starts, gold_span_mask, 0)
        clamped_gold_span_ends = \
            util.replace_masked_values(gold_span_ends, gold_span_mask, 0)
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_span_starts = \
            torch.gather(span_start_log_probs, 1, clamped_gold_span_starts)
        log_likelihood_for_span_ends = \
            torch.gather(span_end_log_probs, 1, clamped_gold_span_ends)
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_spans = \
            log_likelihood_for_span_starts + log_likelihood_for_span_ends
        # For those padded spans, we set their log probabilities to be very small negative value
        log_likelihood_for_spans = \
            util.replace_masked_values(log_likelihood_for_spans, gold_span_mask, -1e7)
        # Shape: (batch_size, )
        log_marginal_likelihood_for_span = util.logsumexp(log_likelihood_for_spans)
        return log_marginal_likelihood_for_span
