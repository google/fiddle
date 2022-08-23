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

import copy
import dataclasses
import inspect
from typing import Any, Iterator, List, Optional, Sequence, Type, Union

from fiddle import config
from fiddle import history
from fiddle import tagging
from fiddle.codegen import formatting_utilities
from fiddle.experimental import daglish
from fiddle.experimental import daglish_traversal
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


def _has_nested_builder(value: Any, state=None) -> bool:
  state = state or daglish_traversal.MemoizedTraversal.begin(
      _has_nested_builder, value)
  return (isinstance(value, config.Buildable) or
          (state.is_traversable(value) and
           any(state.flattened_map_children(value).values)))


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
  wrapped: tagging.TaggedValueCls

  def __repr__(self):
    if self.wrapped.value is tagging.NO_VALUE:
      value_repr = '<[unset]>'
    else:
      value_repr = repr(self.wrapped.value)
    tag_str = ' '.join(sorted(str(t) for t in self.wrapped.tags))
    return f'{value_repr} {tag_str}'


@dataclasses.dataclass(frozen=True)
class _LeafSetting:
  """Represents a leaf configuration setting."""
  path: daglish.Path
  annotation: Optional[Type[Any]]
  value: Any


def _get_annotation(cfg: config.Buildable,
                    path: daglish.Path) -> Optional[Type[Any]]:
  """Gets the type annotation associated with a Daglish path."""
  if not path or not isinstance(path[-1], daglish.Attr):
    return None
  value = cfg
  for path_element in path[:-1]:
    value = path_element.follow(value)
  if isinstance(value, config.Buildable):
    try:
      param = value.__signature__.parameters[path[-1].name]
    except KeyError:
      # Try to use the kwarg annotation
      for param in value.__signature__.parameters.values():
        if param.kind == param.VAR_KEYWORD:
          return None if param.annotation is param.empty else param.annotation
      return None  # probably a kwarg
    return None if param.annotation is param.empty else param.annotation


def _rearrange_buildable_args_and_insert_unset_sentinels(
    value: config.Buildable) -> config.Buildable:
  """Returns a copy of a Buildable with normalized arguments.

  This normalizes arguments by re-creating the __arguments__ dictionary in the
  order of the configured function or class' signature. It also inserts "unset"
  sentinels for values in the signature that don't have a value set.

  Args:
    value: Buildable to copy and normalize.

  Returns:
    Copy of `value` with arguments normalized.
  """
  # TODO: Consider pulling part of this function into a shared
  # module, or achieving the same effect by modifying traversal order.
  value = copy.copy(value)
  old_arguments = dict(value.__arguments__)
  new_arguments = {}
  for param_name, param in value.__signature__.parameters.items():
    if param.kind in {param.VAR_KEYWORD, param.VAR_POSITIONAL}:
      continue
    elif param_name in old_arguments:
      new_arguments[param_name] = old_arguments.pop(param_name)
    else:
      new_arguments[param_name] = _UnsetValue(param)
  new_arguments.update(old_arguments)  # Add in kwargs, in current order.
  object.__setattr__(value, '__arguments__', new_arguments)
  return value


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

  def generate(value, state=None) -> Iterator[_LeafSetting]:
    state = state or daglish_traversal.BasicTraversal.begin(generate, value)

    # Rearrange parameters in signature order, and add "unset" sentinels.
    if isinstance(value, config.Buildable):
      value = _rearrange_buildable_args_and_insert_unset_sentinels(value)

    if isinstance(value, tagging.TaggedValueCls):
      value = _TaggedValueWrapper(value)
      annotation = _get_annotation(cfg, state.current_path)
      yield _LeafSetting(state.current_path, annotation, value)
    elif not _has_nested_builder(value):
      annotation = _get_annotation(cfg, state.current_path)
      yield _LeafSetting(state.current_path, annotation, value)
    elif state.is_traversable(value):
      for sub_result in state.flattened_map_children(value).values:
        yield from sub_result
    else:
      annotation = _get_annotation(cfg, state.current_path)
      yield _LeafSetting(state.current_path, annotation, value)

  def format_line(line: _LeafSetting):
    type_annotation = ''
    if include_types and line.annotation is not None:
      try:
        type_annotation = f': {line.annotation.__qualname__}'
      except AttributeError:
        # Certain types, such as Union, do not have a __qualname__ attribute.
        type_annotation = f': {line.annotation}'
    value = _format_value(line.value, raw_value_repr=raw_value_repr)
    path_str = daglish.path_str(line.path)
    assert path_str.startswith('.'), 'Expected root to be a buildable'
    path_str = path_str[1:]
    return f'{path_str}{type_annotation} = {value}'

  return '\n'.join(map(format_line, generate(cfg)))


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
    value = _format_value(entry.new_value, raw_value_repr=raw_value_repr)
    return f'  - previously: {value} @ {entry.location}'

  # TODO: Add support for printing tags changes.
  value_history = [
      entry for entry in param_history
      if entry.kind == history.ChangeKind.NEW_VALUE
  ]
  if len(value_history) > 1:
    value_updates = [
        make_previous_text(entry) for entry in reversed(value_history[:-1])
    ]
    past = '\n'.join(value_updates)
    past = '\n' + past  # prefix with a newline.
  else:
    past = ''
  current_value = _format_value(
      value_history[-1].new_value, raw_value_repr=raw_value_repr)
  current = f'{current_value} @ {value_history[-1].location}'
  return f'{_path_to_str(path)} = {current}{past}'
