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

"""Utilities for traversing and converting dataclasses."""

import dataclasses
from typing import Any, Optional, Type

from fiddle._src import config
from fiddle._src import daglish


def _names(value_or_type):
  return [field.name for field in dataclasses.fields(value_or_type)]


def _flatten_dataclass(value):
  names = _names(value)
  return [getattr(value, name) for name in names], (type(value), names)


def _unflatten_dataclass(values, metadata):
  typ, names = metadata
  assert len(names) == len(values)
  values_dict = {name: value for name, value in zip(names, values)}
  return typ(**values_dict)


def _path_elements_fn(value):
  return tuple(daglish.Attr(name) for name in _names(value))


dataclass_traverser = daglish.NodeTraverser(
    flatten=_flatten_dataclass,
    unflatten=_unflatten_dataclass,
    path_elements=_path_elements_fn,
)


class DataclassTraverserRegistry(daglish.NodeTraverserRegistry):

  def find_node_traverser(
      self, node_type: Type[Any]
  ) -> Optional[daglish.NodeTraverser]:
    traverser = self.fallback_registry.find_node_traverser(node_type)
    if traverser is None and dataclasses.is_dataclass(node_type):
      traverser = dataclass_traverser
    return traverser


daglish_dataclass_registry = DataclassTraverserRegistry(use_fallback=True)


def convert_dataclasses_to_configs(
    root: Any, allow_post_init: bool = False
) -> Any:
  """Converts a dataclass or nested structure of dataclasses to Fiddle configs.

  Args:
    root: Root dataclass instance or structure (lists, tuples, dicts, etc.) of
      dataclasses.
    allow_post_init: Allow conversion of dataclasses with a __post_init__
      method.

  Returns:
    Fiddle config that, when built, should return an object equal to `root`
    (presuming `root` does not already contain Fiddle configs).
  """

  def traverse(value, state: daglish.State):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
      if hasattr(type(value), "__post_init__") and not allow_post_init:
        raise ValueError(
            "Dataclasses with __post_init__ are not allowed, "
            "unless allow_post_init is set to True."
        )
      value = config.Config(
          type(value),
          **{
              field.name: getattr(value, field.name)
              for field in dataclasses.fields(value)
              if field.init
          },
      )
    return state.map_children(value)

  state = daglish.MemoizedTraversal(
      traverse, root, daglish_dataclass_registry
  ).initial_state()
  return traverse(root, state)
