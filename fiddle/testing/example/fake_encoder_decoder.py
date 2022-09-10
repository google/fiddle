# coding=utf-8
# Copyright 2022 The Fiddle-Config Authors.
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

"""Test mock of an encoder-decoder module structure.

This fixture doesn't have a lot of sharing, but does test a lot of container
types.
"""

import dataclasses
from typing import Any, Dict, List

from fiddle.experimental import auto_config


@dataclasses.dataclass
class Attention:
  dtype: Any
  kernel_init: Any
  bias_init: Any


@dataclasses.dataclass
class CrossAttention(Attention):
  """Example subclass."""


@dataclasses.dataclass
class Mlp:
  dtype: Any
  use_bias: bool
  sharding_axes: List[str]


@dataclasses.dataclass
class TokenEmbedder:
  dtype: Any


# In real models there are multiple layers, but pretend there is just one for
# this test.
@dataclasses.dataclass
class FakeEncoder:
  embedders: Dict[str, Any]
  attention: Attention
  mlp: Mlp


@dataclasses.dataclass
class FakeDecoder:
  embedders: Dict[str, Any]
  self_attention: Attention
  encoder_decoder_attention: CrossAttention
  mlp: Mlp


@dataclasses.dataclass
class FakeEncoderDecoder:
  encoder: FakeEncoder
  decoder: FakeDecoder


@auto_config.auto_config
def fixture():
  dtype = "float32"
  kernel_init = "uniform()"
  bias_init = "zeros()"
  shared_token_embedder = TokenEmbedder(dtype)
  return FakeEncoderDecoder(
      encoder=FakeEncoder(
          embedders={
              "tokens": shared_token_embedder,
              "position": None
          },
          attention=Attention(dtype, kernel_init, bias_init),
          mlp=Mlp(dtype, False, ["embed", "num_heads", "head_dim"]),
      ),
      decoder=FakeDecoder(
          embedders={"tokens": shared_token_embedder},
          self_attention=Attention(dtype, kernel_init, bias_init),
          encoder_decoder_attention=CrossAttention(dtype, kernel_init,
                                                   bias_init),
          mlp=Mlp(dtype, False, ["num_heads", "head_dim", "embed"]),
      ))
