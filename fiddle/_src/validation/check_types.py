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

"""Checks that values in a `Buildable` conform to their type annotations."""
from typing import Any, List, get_origin

from fiddle._src import config as config_lib
from fiddle._src import daglish
from fiddle._src import signatures


def get_type_errors(config: config_lib.Buildable) -> List[str]:
  """Returns a list of type errors found in the given `config`.

  NOTE: Please see the documentation of `check_types()` below for a list of
  limitations of this API.

  Args:
    config: A ``Buildable`` instance to build, or a nested structure of
      ``Buildable`` objects.

  Returns:
    A list of type errors.
  """
  errors = []

  def _check_types_recursive(value: Any, state: daglish.State) -> Any:
    if isinstance(value, config_lib.Buildable):
      sub_traversal = state.flattened_map_children(value)
      arguments = sub_traversal.metadata.arguments(sub_traversal.values)
      type_hints = signatures.get_type_hints(value.__fn_or_cls__)

      for arg_name, arg_value in arguments.items():
        if isinstance(arg_value, config_lib.Buildable):
          # For args configured as `fdl.Config` or `fdl.Partial` don't
          # raise a TypeError.
          continue
        if arg_name in type_hints and type_hints[arg_name] != Any:
          add_error = False
          try:
            if not isinstance(arg_value, type_hints[arg_name]):
              add_error = True
          except TypeError:
            # isinstance() calls for objects whose types are subscripted
            # generics raise a `TypeError`. For these, get their origin type and
            # check that the origin types match.
            type_hint_origin = get_origin(type_hints[arg_name])
            if not isinstance(arg_value, type_hint_origin):
              add_error = True
          if add_error:
            path_str = daglish.path_str(state.current_path)
            errors.append(
                'For attribute'
                f' <config>{path_str}.{arg_name} provided type:'
                f' {type(arg_value)} is not of annotated/declared type:'
                f' {type_hints[arg_name]}'
            )

    return state.map_children(value)

  daglish.MemoizedTraversal.run(_check_types_recursive, config)
  return errors


def check_types(config: config_lib.Buildable):
  """Checks that `config`'s values match their type annotations.

  NOTE: Please keep in mind the following limitations:
  - Leaf values inside standard collections aren't type checked (e.g., no
    attempt is made to ensure that a collection annotated as `list[int]` only
    contains `int` values).

  Args:
    config: A ``Buildable`` instance to build, or a nested structure of
      ``Buildable`` objects.

  Raises:
    TypeError: if the types of the arguments do not match those of the
    `Buildable`s attributes declared type annotations.
  """
  errors = get_type_errors(config)
  if errors:
    raise TypeError(
        'The following type mismatches were found in the config: \n'
        + '\n'.join(errors),
    )
