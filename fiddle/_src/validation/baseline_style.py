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

"""Programmatic checks for the style of a baseline configuration.

"Baseline configurations" are configurations for baseline or flagship models.
They will be read by a large number of people, so keeping them simple and
readable is important.
"""

from typing import Any, Dict, List, Optional

from fiddle._src import config as config_lib
from fiddle._src import daglish
from fiddle._src import history


def _check_max_writes_per_attribute(
    attr_history: List[history.HistoryEntry],
    max_writes_per_attribute: int,
    path_str: str,
) -> None:
  """Style check for the max number of writes to an attribute.

  Checks the max number of writes to an attribute doesn't exceed a certain
  threshold.

  Args:
    attr_history: The list of history entries for a given config attribute.
    max_writes_per_attribute: Threshold for maximum number of writes to a given
      config attribute.
    path_str: A string that would be returned by `daglish.path_str(p)` for some
      path p.
  """
  value_history = [
      entry
      for entry in attr_history
      if entry.kind == history.ChangeKind.NEW_VALUE
  ]

  filenames = set()
  for history_entry in attr_history:
    location: history.Location = history_entry.location
    filenames.add(location.filename)

  debug_str = ", ".join(f"{filename}" for filename in filenames)

  if len(value_history) > max_writes_per_attribute:
    raise ValueError(
        f"{len(value_history)} value(s) written to <config>{path_str} in files:"
        f" {debug_str}. Please limit the number of overrides of this config"
        f" attribute to {max_writes_per_attribute} to maintain baseline"
        " configuration readability."
    )


def _check_max_config_source_files(
    sample_paths_by_file: Dict[str, str],
    max_config_source_files: int,
) -> None:
  """Style check for number of files where an attribute is written.

  Checks the max number of files where an attribute is written, doesn't exceed a
  certain threshold.

  Args:
    sample_paths_by_file: A dictionary of filenames mapped to string that would
      be returned by `daglish.path_str(p)` for some path p for a config's
      attribute.
    max_config_source_files: Threshold for maximum number of files producing
      `config`.
  """
  if len(sample_paths_by_file) > max_config_source_files:
    debug_str = ", ".join(
        f"{filename} (example: wrote to <config>{path})"
        for filename, path in sample_paths_by_file.items()
    )
    raise ValueError(
        f"More than {max_config_source_files} file(s) produced this"
        " config. For baseline configurations, please ensure that you don't"
        " have a big hierarchy of files each overriding values in a config,"
        " because this is harder to read for new users. The files which have"
        f" written to this config are: {debug_str}."
    )


def check_baseline_style(
    config: Any,
    *,
    max_config_source_files: int = 3,
    max_writes_per_attribute: int = 2,
    ignore: Optional[List[Any]] = None,
) -> None:
  """Checks that a configuration is style-conformant.

  Args:
    config: fdl.Buildable object, or collection which may include Buildable
      objects.
    max_config_source_files: Threshold for maximum number of files producing
      `config`.
    max_writes_per_attribute: Threshold for maximum number of writes to
      `config`'s attributes.
    ignore: An optional list of config sub-objects that are ignored for these
      readability validations/style checks.
  """

  # filename --> sample path.
  sample_paths_by_file = {}

  ignore = ignore or []
  ignore_ids = {id(obj) for obj in ignore}

  def traverse(value: Any, state: daglish.State) -> Any:
    if id(value) in ignore_ids:
      return

    if isinstance(value, config_lib.Buildable):
      for attr_name, attr_value in config_lib.ordered_arguments(value).items():
        if id(attr_value) in ignore_ids:
          continue

        attr_history = value.__argument_history__[attr_name]
        path_str = daglish.path_str(state.current_path)

        _check_max_writes_per_attribute(
            attr_history, max_writes_per_attribute, path_str
        )

        for history_entry in attr_history:
          location: history.Location = history_entry.location
          sample_paths_by_file.setdefault(location.filename, path_str)

    for _ in state.yield_map_child_values(value, ignore_leaves=True):
      pass  # Run lazy iterator.

  daglish.MemoizedTraversal.run(traverse, config)

  _check_max_config_source_files(sample_paths_by_file, max_config_source_files)
