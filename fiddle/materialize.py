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

from fiddle import config as buildable
import tree


def materialize_defaults(config: buildable.Buildable, *, recurse: bool = True):
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
    config: The Buildable to materialize defaults on; materialization occurs in
      place, and mutates `cfg`. If you would like `cfg` to be unmodified, make a
      copy first with `copy.deepcopy(cfg)`.
    recurse: Whether to recurse into the arguments of `cfg` and materialize
      their defaults too.
  """
  for arg in config.__signature__.parameters.values():
    if arg.default is not arg.empty and arg.name not in config.__arguments__:
      setattr(config, arg.name, arg.default)
  if recurse:
    tree.traverse(_materialize_defaults_traverse_function, config.__arguments__)


def _materialize_defaults_traverse_function(leaf: Any):
  """Callback used when traversing a tree in `materialize_config`."""
  if isinstance(leaf, buildable.Buildable):
    materialize_defaults(leaf, recurse=True)
