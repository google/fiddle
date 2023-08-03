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

"""Checks that custom objects/instances are not present in a configuration.

In general, we recommend building configurations from

(a) primitives like int/float/str/etc.
(b) basic python collections like lists/tuples/dicts
(c) Fiddle configs like fdl.Config, fdl.Partial

Usually, when a custom user object is present in a config, it can be rewritten
to be a fdl.Config that constructs that object. For example, instead of

config.foo.preprocessor = MyPreprocessor(dtype="int32")

write,

config.foo.preprocessor = fdl.Config(MyPreprocessor, dtype="int32")
"""

import dataclasses
from typing import Any, List, Optional

import fiddle as fdl
from fiddle import daglish
from fiddle import history


def _concise_history(entries: List[history.HistoryEntry]) -> str:
  """Returns a concise string for a history entry."""
  if not entries:
    return "(no history)"
  set_or_deleted = (
      "Set in "
      if entries[-1].new_value is not history.DELETED
      else "Deleted in "
  )
  return set_or_deleted + entries[-1].location.format(max_filename_parts=3)


def _get_history_from_state(
    state: daglish.State,
) -> Optional[List[history.HistoryEntry]]:
  """Returns the history from a Buildable for a state."""
  while state.current_path and state.parent is not None:
    attr = state.current_path[-1]
    state = state.parent
    if isinstance(state.original_value, fdl.Buildable):
      assert isinstance(attr, daglish.Attr)
      entries = state.original_value.__argument_history__[attr.name]
      return entries if entries else None
  return None


def get_config_errors(config: Any) -> List[str]:
  """Returns a list of errors found in the given config.

  Args:
    config: Fiddle config object, or nested structure of configs.
  """
  errors = []

  def history_str(state):
    return ", " + _concise_history(_get_history_from_state(state))

  def traverse(value, state: daglish.State):
    path_str = daglish.path_str(state.current_path)
    if isinstance(value, tuple) and hasattr(type(value), "_fields"):
      errors.append(f"Found namedtuple at {path_str}{history_str(state)}")
    elif dataclasses.is_dataclass(value):
      errors.append(f"Found dataclass at {path_str}{history_str(state)}")
    elif (not state.is_traversable(value)) and not daglish.is_unshareable(
        value
    ):
      errors.append(f"Found {type(value)} at {path_str}{history_str(state)}")
    return state.map_children(value)

  daglish.MemoizedTraversal.run(traverse, config)
  return errors


def check_no_custom_objects(config: Any) -> None:
  """Checks that no custom objects are present in the given config.

  Args:
    config: Fiddle config object, or nested structure of configs.

  Raises:
    ValueError: If the configuration contains custom objects.
  """
  errors = get_config_errors(config)
  if errors:
    raise ValueError(
        "Custom objects were found in the config. In general, you should "
        "be able to convert these to fdl.Config's. Custom objects:\n  "
        + "\n  ".join(errors),
    )
