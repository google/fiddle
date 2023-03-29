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

"""Helper functions for visualization of Fiddle configurations.

By default, you should be able to call APIs in `graphviz` and `printing` with no
special handling. But some real-world configurations become gigantic, and so
having a few helper functions for trimming them for interactive demos and such
can be valuable.
"""

import copy
import inspect
import textwrap
from typing import Any, Dict, Iterable, List, Optional, TypeVar

from fiddle import daglish
from fiddle import diffing
from fiddle import graphviz
from fiddle._src import config as config_lib
from fiddle._src.experimental import daglish_legacy


def _raise_error():
  """Helper function for `Trimmed` that raises an error."""
  raise ValueError('Please do not call fdl.build() on a config tree with '
                   'Trimmed() nodes. These nodes are for visualization only.')


class Trimmed(config_lib.Config[type(None)], graphviz.CustomGraphvizBuildable):
  """Represents a configuration that has been trimmed."""

  def __init__(self):
    super().__init__(_raise_error)

  def __render_value__(self, api: graphviz.GraphvizRendererApi) -> Any:
    return api.tag('i')('(trimmed...)')

  @classmethod
  def __unflatten__(
      cls,
      values: Iterable[Any],
      metadata: config_lib.BuildableTraverserMetadata,
  ):
    return cls()


_T = TypeVar('_T')


def trimmed(config: _T, trim: List[config_lib.Buildable]) -> _T:
  """Returns a deep copy of a Buildable, with certain nodes replaced.

  This copy should have the same DAG structure as the original configuration.

  Example usage:

  graphviz.render(visualize.trimmed(config, [config.my_large_model_sub_config]))

  Args:
    config: Root buildable object, or collection of buildables.
    trim: List of sub-buildables that will be replaced by Trimmed() instances.

  Returns:
    Copy of the Buildable with nodes replaced.
  """

  return copy.deepcopy(config,
                       {id(sub_config): Trimmed() for sub_config in trim})


def with_defaults_trimmed(config: _T) -> _T:
  """Trims arguments that match their default values.

  Args:
    config: Root buildable object, or nested structure of Buildables.

  Returns:
    Copy of the Buildable with args matching defaults removed.
  """

  # TODO(b/241261427): Remove args matching dataclasses' default_factory too.
  def traverse_fn(_, value):
    new_value = (yield)
    if isinstance(value, config_lib.Buildable):
      for name, attr_value in list(new_value.__arguments__.items()):
        param = value.__signature__.parameters.get(name, None)
        if (param is not None and
            param.kind != inspect.Parameter.VAR_KEYWORD and
            param.default == attr_value):
          delattr(new_value, name)
      return new_value
    else:
      return new_value

  return daglish_legacy.memoized_traverse(traverse_fn, config)


def depth_over(config: Any, depth: int) -> List[config_lib.Buildable]:
  """Returns a list of sub-Buildable objects over a given depth from the root.

  If a sub-Buildable has multiple paths from the root to it, its depth is the
  *minimum* distance of those paths.

  Furthermore, only Buildable nodes count towards the depth; if a
  sub-configuration is nested in containers like lists, tuples, or dicts, those
  do not count. This is because this function is primarily geared towards
  visualization, and those containers are usually displayed inline.

  Args:
    config: Root buildable object, or collection of buildable objects.
    depth: Depth value.

  Returns:
    List of all sub-Buildables with depth strictly greater than `depth`.
  """
  node_to_depth: Dict[int, int] = {}  # Buildable hash --> min depth.
  id_to_node: Dict[int,
                   config_lib.Buildable] = {}  # Buildable hash --> Buildable.

  def _path_len(path: daglish.Path) -> int:
    return sum(1 if isinstance(elt, daglish.Attr) else 0 for elt in path)

  def traverse(node, state: daglish.State) -> None:
    if isinstance(node, config_lib.Buildable):
      all_paths = state.get_all_paths(allow_caching=True)
      node_to_depth[id(node)] = min(_path_len(path) for path in all_paths)
      id_to_node[id(node)] = node
    if state.is_traversable(node):
      state.flattened_map_children(node)

  daglish.MemoizedTraversal.run(traverse, config)

  return [
      id_to_node[key]
      for key, node_depth in node_to_depth.items()
      if node_depth > depth
  ]


_any_value = diffing.AnyValue()


def structure(config: _T) -> _T:
  """Returns a skeleton of a configuration.

  Args:
    config: Buildable item, or collection of Buildable items.

  Returns:
    Copy of `config` that has all of the original Buildable's, plus any fields
    and containers necessary to make all of those original Buildable's linked.
    Any leaf arguments (primitives, containers of primitives) are removed. In
    cases where there is a list with primitives and buildables, then the
    primitive will be replaced with AnyValue and the buildable will be
    maintained, like [1, fdl.Config(Foo)] --> [AnyValue, fdl.Config(Foo)].
  """

  def traverse(value, state: daglish.State):
    if not state.is_traversable(value):
      return _any_value
    elif isinstance(value, config_lib.Buildable):
      result = state.map_children(value)
      for name, sub_value in config_lib.ordered_arguments(result).items():
        if sub_value is _any_value:
          delattr(result, name)
      return result
    else:
      result = state.flattened_map_children(value)
      if any(item is not _any_value for item in result.values):
        return result.unflatten()
      else:
        return _any_value

  return daglish.MemoizedTraversal.run(traverse, config)


def trim_fields_to(
    config: _T,
    top_level_fields: Optional[List[str]] = None,
    *,
    fields_by_config_id: Optional[Dict[int, List[str]]] = None,
) -> _T:
  """Trims fields to a list of desired fields.

  Args:
    config: Configuration object.
    top_level_fields: Fields on the top-level configuration object to keep.
    fields_by_config_id: Fields to keep, indexed by id(sub_config).

  Returns:
    Deep copy of configuration object, with defaults trimmed.
  """

  fields_by_id = copy.copy(fields_by_config_id) if fields_by_config_id else {}
  for key, value in fields_by_id.items():
    if not isinstance(key, int):
      raise TypeError(
          f'fields_by_config_id should have integer keys, found {key}'
      )
    if not isinstance(value, (list, tuple)):
      raise TypeError(
          'Expected fields_by_config_id values to be lists or tuples.'
      )
  if top_level_fields is not None:
    if not isinstance(top_level_fields, (list, tuple)):
      raise TypeError('Expected top_level_fields to be a list or tuple.')
    fields_by_id[id(config)] = top_level_fields

  if not fields_by_id:
    return copy.deepcopy(config)

  def traverse(value, state: daglish.State):
    if isinstance(value, config_lib.Buildable) and id(value) in fields_by_id:
      to_keep = fields_by_id[id(value)]
      value = copy.copy(value)  # Shallow copy to avoid mutating original.
      for argument in set(config_lib.ordered_arguments(value)) - set(to_keep):
        setattr(value, argument, Trimmed())
    return state.map_children(value)

  return daglish.MemoizedTraversal.run(traverse, config)


class _TruncatedRepr:
  """Represent the truncated fields for visualization purpose only."""

  def __init__(self, prefix: str):
    self._prefix = prefix

  def __repr__(self):
    return self._prefix


def trim_long_fields(
    config: _T,
    threshold: int = 60,
) -> _T:
  """Trims fields to a bounded length.

  Args:
    config: Configuration object.
    threshold: Long fields will be trimmed to the threshold length.

  Returns:
    Deep copy of configuration object, with long fileds trimmed.
  """

  if not isinstance(threshold, int) or threshold <= 0:
    raise ValueError(f'threshold must be a positive int, got {threshold}.')

  def traverse(value, state: daglish.State):
    if isinstance(value, config_lib.Buildable):
      for argument in set(config_lib.ordered_arguments(value)):
        field = getattr(value, argument)
        if not isinstance(field, (config_lib.Buildable, list, tuple, dict)):
          field_repr = repr(field)
          if len(field_repr) > threshold:
            s = textwrap.shorten(
                repr(field), width=threshold, placeholder='...'
            )
            prefix = _TruncatedRepr(s)
            setattr(value, argument, prefix)
    return state.map_children(value)

  return daglish.MemoizedTraversal.run(traverse, config)