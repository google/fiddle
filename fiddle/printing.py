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
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Type, Union

from fiddle import config
from fiddle import history
from fiddle import tagging
from fiddle.codegen import formatting_utilities
import tree


@dataclasses.dataclass(frozen=True)
class _ParamName:
  """Wrapper for a string to differentiate Buildable param names from dict keys."""
  __slots__ = ('name',)
  name: str


@dataclasses.dataclass(frozen=True)
class _UnsetValue:
  """A wrapper class indicating an unset value."""
  __slots__ = ('parameter',)
  parameter: inspect.Parameter

  def __repr__(self) -> str:
    if self.parameter.default is self.parameter.empty:
      return '<[unset]>'
    else:
      default_value = _format_value(
          self.parameter.default, raw_value_repr=False)
      return f'<[unset; default: {default_value}]>'


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


def _has_nested_builder(children: Sequence[Tuple[_Path, _Leaf]]) -> bool:
  return any(
      isinstance(child, config.Buildable) for unused_path, child in children)


def _format_value(value: Any, *, raw_value_repr: bool) -> str:
  if raw_value_repr:
    return repr(value)
  else:
    if isinstance(value, str):
      return repr(value)
    else:
      return formatting_utilities.pretty_print(value)


@dataclasses.dataclass(frozen=True)
class _TaggedValueWrapper:
  """Customizes representation for TaggedValues in flattened output."""
  __slots__ = ('wrapped',)
  wrapped: tagging.TaggedValue

  def __repr__(self):
    if self.wrapped.value is tagging.NO_VALUE:
      value_repr = '<[unset]>'
    else:
      value_repr = repr(self.wrapped.value)
    tag_str = ' '.join(sorted(str(t) for t in self.wrapped.tags))
    return f'{value_repr} {tag_str}'


def as_str_flattened(cfg: config.Buildable,
                     *,
                     include_types: bool = True,
                     raw_value_repr: bool = False) -> str:
  """Returns a string of cfg's paths and values, one pair per line.

  Some automated tools (e.g. grep, sort, diff, ...) handle data line-by-line.
  This output format contains the full path to a value and a string
  representation of said value on a single line.

  Args:
    cfg: A buildable to generate a string representation for.
    include_types: If true, include type annotations (where available) in the
      string representation of each leaf value.
    raw_value_repr: If true, use `repr` on values, otherwise, a custom
      pretty-printed format is used.
  Returns: a string representation of `cfg`.
  """

  def flatten_children(
      value: Any, annotation: Optional[Type[Any]],
      path: _Path) -> Iterator[Tuple[_Path, Optional[Type[Any]], _Leaf]]:
    flattened_children = tree.flatten_with_path(value)
    if _has_nested_builder(flattened_children):
      for child_path, leaf in flattened_children:
        if isinstance(leaf, tagging.TaggedValue):
          yield path + child_path, None, _TaggedValueWrapper(leaf)
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
      try:
        type_annotation = f': {line[1].__qualname__}'
      except AttributeError:
        # Certain types, such as Union, do not have a __qualname__ attribute.
        type_annotation = f': {line[1]}'
    value = _format_value(line[2], raw_value_repr=raw_value_repr)
    return f'{_path_to_str(line[0])}{type_annotation} = {value}'

  return '\n'.join(map(format_line, recursive_flatten(cfg)))


def history_per_leaf_parameter(cfg: config.Buildable,
                               *,
                               raw_value_repr: bool = False) -> str:
  """Returns a string representing the history of cfg's leaf params.

  Because Buildable's are designed to be mutated, tracking down when and where
  a particular parameter's value is changed can be difficult. This function
  returns a string representation of each parameter and (in
  reverse-chronological order) its current and past values.

  This representation elides the "DAG"-construction aspects of constructing the
  `cfg`, and instead only prints the history of the "outer-most" parameters
  (ones that don't contain a reference to another Buildable).

  Args:
    cfg: A buildable to generate a history for.
    raw_value_repr: If true, use `repr` when string-ifying values, otherwise use
      a customized pretty-printing routine.

  Returns:
    A string representation of `cfg`'s history, organized by param name.
  """
  return '\n'.join(
      _make_per_leaf_histories_recursive(cfg, raw_value_repr=raw_value_repr))


def _make_per_leaf_histories_recursive(
    cfg: config.Buildable,
    raw_value_repr: bool,
    path: Optional[_Path] = None,
) -> Iterator[str]:
  """Recursively traverses `cfg` and generates per-param history summaries."""
  for name, param_history in cfg.__argument_history__.items():
    if path:
      param_path = tuple(path) + (_ParamName(name),)
    else:
      param_path = (_ParamName(name),)
    children = tree.flatten_with_path(getattr(cfg, name))
    if _has_nested_builder(children):
      for child_path, child_leaf in children:
        full_child_path = param_path + child_path
        if isinstance(child_leaf, config.Buildable):
          yield from _make_per_leaf_histories_recursive(child_leaf,
                                                        raw_value_repr,
                                                        full_child_path)
        else:
          yield f'{_path_to_str(full_child_path)} = {child_leaf}'
    else:
      yield _make_per_leaf_history_text(param_path, param_history,
                                        raw_value_repr)

  unset_fields = sorted(
      set(cfg.__signature__.parameters.keys()) -
      set(cfg.__argument_history__.keys()))
  for name in unset_fields:
    if path:
      param_path = tuple(path) + (_ParamName(name),)
    else:
      param_path = (_ParamName(name),)
    yield f'{_path_to_str(param_path)} = <[unset]>'


def _make_per_leaf_history_text(path: _Path,
                                param_history: List[history.HistoryEntry],
                                raw_value_repr: bool) -> str:
  """Returns a string representing a parameter's history.

  Args:
    path: The path to the parameter.
    param_history: The parameter's history.
    raw_value_repr: If True, use `repr` unmodified, otherwise, use a custom
      pretty-printing routine.

  Returns:
    A string representation of the parameter's history that can be displayed to
    the user.
  """
  assert param_history, 'param_history should never be empty.'

  def make_previous_text(entry: history.HistoryEntry) -> str:
    value = _format_value(entry.value, raw_value_repr=raw_value_repr)
    return f'  - previously: {value} @ {entry.location}'

  if len(param_history) > 1:
    past = '\n'.join(map(make_previous_text, reversed(param_history[:-1])))
    past = '\n' + past  # prefix with a newline.
  else:
    past = ''
  current_value = _format_value(
      param_history[-1].value, raw_value_repr=raw_value_repr)
  current = f'{current_value} @ {param_history[-1].location}'
  return f'{_path_to_str(path)} = {current}{past}'
