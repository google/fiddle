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

"""Super simple regex search API for configs.

We might re-implement this on top of the printing API at some point.
"""

import re
from typing import Any, Optional, Union

from fiddle import daglish
from fiddle._src import config as config_lib


def grep(
    config: Any,
    # Note: switch these back to Pattern[str] once we can drop 3.8 support.
    pattern: Union[str, re.Pattern],  # pylint: disable=g-bare-generic
    *,
    exclude: Optional[Union[str, re.Pattern]] = None,  # pylint: disable=g-bare-generic
    case_sensitive=True,
    output_fn=print,
) -> None:
  """Grep through a config.

  Example:

    from fiddle.experimental import grep as fdl_grep

    fdl_grep.grep(config, r"loss|accumulator")

  Args:
    config: Base configuration, usually a fdl.Buildable but can be any value.
    pattern: Pattern to search for.
    exclude: Alternate pattern to exclude from results. This can sometimes be
      easier to use than crafting `pattern` to remove certain results.
    case_sensitive: When `pattern` (or `exclude`) is a string, whether to use a
      case-sensitive regex. Ignored when `pattern` (or `exclude`) are passed as
      re.Pattern objects.
    output_fn: Optional override function to output values with. For example,
      you could pass `logging.info`, or if you want to gather results in a list,
      pass `grep(..., output_fn=my_list.append)`.
  """
  flags = 0 if case_sensitive else re.IGNORECASE
  if isinstance(pattern, str):
    pattern = re.compile(pattern, flags=flags)
  if isinstance(exclude, str):
    exclude = re.compile(exclude, flags=flags)

  def traverse(value, state: daglish.State):
    path_str = daglish.path_str(state.current_path)
    if state.is_traversable(value):
      state.flattened_map_children(value)
      if isinstance(value, config_lib.Buildable):
        fn_or_cls = config_lib.get_callable(value)
        value_str = f"<{type(value).__name__}({fn_or_cls.__name__})>"
      else:
        value_str = None
    else:
      value_str = repr(value)

    output_str = f"{path_str}: {value_str}"
    if pattern.search(output_str):
      if exclude is None or not exclude.search(output_str):
        output_fn(output_str)

  daglish.BasicTraversal.run(traverse, config)
