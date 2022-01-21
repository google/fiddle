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

"""Functions to help serialize configuration."""

import collections
import copy

import fiddle as fdl
import tree


def clear_argument_history(config: fdl.Buildable, deepcopy: bool = True):
  """Creates a copy of a config, clearing its history.

  This can be useful when a Config's history contains a non-serializable value.

  Args:
    config: Fiddle Buildable.
    deepcopy: Whether to deepcopy the configuration before clearing its history.

  Returns:
    Deepcopy of `config` with `__argument_history__` set to empty.
  """

  def _clear_history(node: fdl.Buildable):
    object.__setattr__(node, "__argument_history__",
                       collections.defaultdict(list))

    for leaf in tree.flatten(node.__arguments__):
      if isinstance(leaf, fdl.Buildable):
        _clear_history(leaf)

  if deepcopy:
    config = copy.deepcopy(config)
  _clear_history(config)
  return config
