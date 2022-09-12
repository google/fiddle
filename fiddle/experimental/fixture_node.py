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

"""Declares a fixture node, which allows two-stage configuration.

By delegating implementation of the fixture to the user, we sidestep questions
of what sub-configuation DAGs should be shared, how to replicate them, etc.,
which are hard questions to answer well at the Fiddle library level.
"""

from typing import Any

from fiddle import config as fdl_config
from fiddle.experimental import daglish_legacy


class FixtureNode(fdl_config.Buildable):

  def __build__(self, /, *args: Any, **kwargs: Any):
    raise ValueError(
        "You must first materialize a Fiddle configuration that contains "
        "`FixtureNode`s. Please call "
        "`config = fixture_node.materialize(config)`.")


def materialize(config: Any):
  """Invokes any `FixtureNode`s, resulting in low-level configuration.

  Args:
    config: fdl.Buildable object, or collection which may include Buildable
      objects. Any `FixtureNode`s within this DAG will be invoked.

  Returns:
    Lower-level configuration as a result of invoking `FixtureNode`s in
    `config`.
  """

  def traverse_fn(unused_all_paths, unused_value):
    new_value = (yield)
    if isinstance(new_value, FixtureNode):
      # TODO: If this proposal is taken forward, preserve
      # __argument_history__ as well.
      return new_value.__fn_or_cls__(**new_value.__arguments__)
    else:
      return new_value

  return daglish_legacy.memoized_traverse(traverse_fn, config)
