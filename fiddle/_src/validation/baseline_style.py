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

from typing import Any

from fiddle._src import config as config_lib
from fiddle._src import daglish
from fiddle._src import history


def check_baseline_style(
    config: Any,
    *,
    max_files_writing_attributes: int = 3,
) -> None:
  """Checks that a configuration is style-conformant."""

  # filename --> sample path.
  sample_paths_by_file = {}

  for value, path in daglish.iterate(config):
    if isinstance(value, config_lib.Buildable):
      for attr_name in config_lib.ordered_arguments(value).keys():
        path_str = daglish.path_str((*path, daglish.Attr(attr_name)))
        attr_history = value.__argument_history__[attr_name]
        for history_entry in attr_history:
          location: history.Location = history_entry.location
          sample_paths_by_file.setdefault(location.filename, path_str)
      pass

  if len(sample_paths_by_file) > max_files_writing_attributes:
    debug_str = ", ".join(
        f"{filename} (example: wrote to <config>{path})"
        for filename, path in sample_paths_by_file.items()
    )
    raise ValueError(
        f"More than {max_files_writing_attributes} file(s) produced this"
        " config. For baseline configurations, please ensure that you don't"
        " have a big hierarchy of files each overriding values in a config,"
        " because this is harder to read for new users. The files which have"
        f" written to this config are: {debug_str}."
    )
