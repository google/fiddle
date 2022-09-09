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

"""The materialize module makes implicit defaults explicit.

Fiddle Buildable's allow accessing argument defaults, but they are treated
differently by the systems compared to values that have been explicitly set
(e.g. by a user's configuration). (This enables users and tooling to quickly see
what has been configured explicitly.)

For when a configuration is to be persisted for longer-term archival,
materializing the defaulted values can make the configuration archive somewhat
more hermetic.
"""

from typing import Any

from fiddle import config
from fiddle import daglish


def materialize_defaults(value: Any) -> None:
  """Explicitly sets values for defaulted arguments.

  Buildable makes a distinction between a defaulted argument that hasn't been
  set, and an argument that has been explicitly set to a value (that may or may
  not be the same as the argument's default value).

  However, there are some instances where it can be valuable to explicitly
  materialize the defaulted values (e.g. when persisting a configuration in a
  text format, or for certain visualizations).

  Calling `materialize_defaults` on a Buildable is the equivalent of doing
  `cfg.MY_ARG = cfg.MY_ARG` for all arguments with default values that have not
  yet had a value explicitly set.

  Args:
    value: A nested collection which may contain Buildable arguments.
  """

  def traverse(node, state: daglish.State):
    if isinstance(node, config.Buildable):
      for arg in node.__signature__.parameters.values():
        if arg.default is not arg.empty and arg.name not in node.__arguments__:
          setattr(node, arg.name, arg.default)
    if state.is_traversable(node):
      state.flattened_map_children(node)

  daglish.MemoizedTraversal.run(traverse, value)
