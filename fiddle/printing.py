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

"""Functions to output representations of `fdl.Buildable`s."""

import dataclasses
import inspect
from typing import Any, Iterator, Optional, Sequence, Tuple, Type, Union

from fiddle import config
from fiddle import placeholders
import tree


@dataclasses.dataclass(frozen=True)
class _ParamName:
  """Wrapper for a string to differentiate Buildable param names from dict keys."""
  __slots__ = ('name',)
  name: str


@dataclasses.dataclass(frozen=True)
class _UnsetValue:
  __slots__ = ('parameter',)
  parameter: inspect.Parameter

  def __repr__(self) -> str:
    if self.parameter.default is self.parameter.empty:
      return '<[unset]>'
    else:
      return f'<[unset; default: {self.parameter.default!r}]>'


_Leaf = Any
_Path = Sequence[Union[int, str, _ParamName]]


def _path_to_str(path: _Path) -> str:
  """Converts a path to a string representation."""
  output = ''
  for item in path:
    if isinstance(item, _ParamName):
      if output:
        output += '.' + item.name
      else:
        # First item.
        output = item.name
    else:
      output += f'[{item!r}]'
  return output


@dataclasses.dataclass(frozen=True)
class _PlaceholderWrapper:
  """Customizes representation for placeholders in flattened output."""
  __slots__ = ('wrapped',)
  wrapped: placeholders.Placeholder

  def __repr__(self):
    if self.wrapped.value is placeholders.NO_VALUE:
      value_repr = '<[unset]>'
    else:
      value_repr = repr(self.wrapped.value)
    return (f'fdl.Placeholder({self.wrapped.key.name!r}, '
            f'value={value_repr})')


def as_str_flattened(cfg: config.Buildable,
                     *,
                     include_types: bool = True) -> str:
  """Returns a string of cfg's paths and values, one pair per line.

  Some automated tools (e.g. grep, sort, diff, ...) handle data line-by-line.
  This output format contains the full path to a value and a string
  representation of said value on a single line.

  Args:
    cfg: A buildable to generate a string representation for.
    include_types: If true, include type annotations (where available) in the
      string representation of each leaf value.
  Returns: a string representation of `cfg`.
  """

  def has_nested_builder(flat_children: Sequence[Tuple[_Path, _Leaf]]) -> bool:

    def is_path_and_leaf_a_buildable(path_and_leaf) -> bool:
      return isinstance(path_and_leaf[1], config.Buildable)

    return any(map(is_path_and_leaf_a_buildable, flat_children))

  def flatten_children(
      value: Any, annotation: Optional[Type[Any]],
      path: _Path) -> Iterator[Tuple[_Path, Optional[Type[Any]], _Leaf]]:
    flattened_children = tree.flatten_with_path(value)
    if has_nested_builder(flattened_children):
      for child_path, leaf in flattened_children:
        if isinstance(leaf, placeholders.Placeholder):
          yield path + child_path, None, _PlaceholderWrapper(leaf)
        elif isinstance(leaf, config.Buildable):
          for subpath, subannotation, actual_leaf in recursive_flatten(leaf):
            yield path + child_path + subpath, subannotation, actual_leaf
        else:
          yield path + child_path, None, leaf
    else:
      yield path, annotation, value

  def recursive_flatten(
      buildable: config.Buildable
  ) -> Iterator[Tuple[_Path, Optional[Type[Any]], _Leaf]]:
    signature = buildable.__signature__
    kwarg_param = None
    for param_name, param in signature.parameters.items():
      if param.kind == param.VAR_KEYWORD:
        kwarg_param = param
        continue
      annotation = None if param.annotation is param.empty else param.annotation
      path = (_ParamName(param_name),)
      if param_name not in buildable.__arguments__:
        yield path, annotation, _UnsetValue(param)
      else:
        yield from flatten_children(buildable.__arguments__[param_name],
                                    annotation, path)
    kwarg_annotation = None
    if kwarg_param and kwarg_param.annotation is not kwarg_param.empty:
      kwarg_annotation = kwarg_param.annotation
    not_yielded_argument_names = (
        buildable.__arguments__.keys() - signature.parameters.keys())
    for name in sorted(not_yielded_argument_names):
      path = (_ParamName(name),)
      yield from flatten_children(buildable.__arguments__[name],
                                  kwarg_annotation, path)

  def format_line(line: Tuple[_Path, Optional[Type[Any]], _Leaf]):
    type_annotation = ''
    if include_types and line[1] is not None:
      type_annotation = f': {line[1].__qualname__}'
    return f'{_path_to_str(line[0])}{type_annotation} = {line[2]!r}'

  return '\n'.join(map(format_line, recursive_flatten(cfg)))
