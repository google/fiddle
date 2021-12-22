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

"""A collection of Fiddle-internal tools to reflect over modules."""

import inspect
import logging
from typing import Any, List


def find_fiddler_like_things(source_module: Any) -> List[str]:
  """Returns a list of names that look like fiddlers.

  A fiddler is a 1-ary function that mutates a config (and returns None).

  Args:
    source_module: A module upon which to probe for potential fiddler-shaped
      attributes.

  Returns:
    Names of attributes that look like potential fiddlers.
  """
  found_names = []
  for name in dir(source_module):
    if name.startswith('__'):
      continue
    try:
      sig = inspect.signature(getattr(source_module, name))
    except Exception:  # pylint: disable=broad-except
      continue  # Ignore.
    required_args = [
        param for param in sig.parameters.values()
        if _is_non_defaulted_positional_args(param)
    ]
    if len(required_args) == 1:
      found_names.append(name)
  return sorted(found_names)


def _is_non_defaulted_positional_args(param: inspect.Parameter) -> bool:
  """Returns True if `param` is a positional argument with no default."""
  return ((param.kind == param.POSITIONAL_OR_KEYWORD or
           param.kind == param.POSITIONAL_ONLY) and
          param.default is param.empty)


def find_base_config_like_things(source_module: Any) -> List[str]:
  """Returns names of attributes of 0-arity functions that might return Configs.

  A base config-producting function is a function that takes no (required)
  arguments, and returns a `fdl.Buildable`.

  Args:
    source_module: A module upon which to search for `base_config`-like
      functions.

  Returns:
    A list of attributes on `source_module` that appear to be base
    config-producing functions.
  """
  available_base_names = []
  for name in dir(source_module):
    if name.startswith('__'):
      continue
    try:
      sig = inspect.signature(getattr(source_module, name))

      def is_required_arg(name: str) -> bool:
        param = sig.parameters[name]  # pylint: disable=cell-var-from-loop
        return ((param.kind == param.POSITIONAL_ONLY or
                 param.kind == param.POSITIONAL_OR_KEYWORD) and
                param.default is param.empty)

      if not any(filter(is_required_arg, sig.parameters)):
        available_base_names.append(name)
    except Exception:  # pylint: disable=broad-except
      logging.debug(
          'Encountered exception while inspecting function called: %s', name)
  return available_base_names
