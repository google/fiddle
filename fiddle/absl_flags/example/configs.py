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

"""A collection of Fiddle-based configurations and fiddlers."""

import fiddle as fdl
from fiddle.absl_flags.example import business_logic
from fiddle.absl_flags.example import tags


def simple(
    filename="data.txt") -> fdl.Config[business_logic.BulkInferenceRunner]:
  """A simple base configuration-generating function."""
  model = fdl.Config(
      business_logic.MyLinearModel,
      w=-3.8,
      b=0.5,
      activation_dtype=tags.ActivationDtype.new(default="float32"))
  loader = fdl.Config(business_logic.MyDataLoader, filename=filename)
  return fdl.Config(business_logic.BulkInferenceRunner, model, loader)


def swap_weight_and_bias(cfg: fdl.Config[business_logic.BulkInferenceRunner]):
  """A sample fiddler."""
  tmp = cfg.model.w
  cfg.model.w = cfg.model.b
  cfg.model.b = tmp
