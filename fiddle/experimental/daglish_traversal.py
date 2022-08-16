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

"""This approach outlines a re-implementation of Daglish methods.

It may eventually replace Daglish.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, TypeVar

from fiddle.experimental import daglish

Self = TypeVar("Self")


@dataclasses.dataclass
class Traversal(metaclass=abc.ABCMeta):
  """Defines an API that traversers must implement.

  Please note that users of a traverser will mostly interact with the State
  API, which bundles a Traverser with the current paths in the traversal.
  """

  traversal_fn: Callable[..., Any]
  root_obj: Any
  registry: daglish.NodeTraverserRegistry = daglish._default_traverser_registry  # pylint: disable=protected-access

  def find_node_traverser(
      self, node_type: Type[Any]) -> Optional[daglish.NodeTraverser]:
    """Uses the configured registry to find a NodeTraverser."""
    return self.registry.find_node_traverser(node_type)

  @abc.abstractmethod
  def apply(self, value: Any, state: State) -> Any:
    """Calls the underlying function bound to the traversal."""
    raise NotImplementedError()

  @abc.abstractmethod
  def all_paths_to_object(self, object_id: int,
                          allow_caching: bool) -> List[daglish.Path]:
    """Returns all paths to a container."""
    raise NotImplementedError()

  def initial_state(self) -> State:
    """Returns an initial state for this traversal.

    The initial state has a reference to the traversal, the empty path, and no
    parent states.
    """
    return State(self, (), self.root_obj, _parent=None)  # pytype: disable=attribute-error

  @classmethod
  def begin(cls, fn: Callable[..., Any], root_obj: Any) -> State:
    """Creates a new traversal and returns the initial state.

    Args:
      fn: Function which is applied at each node during the traversal.
      root_obj: Root object being traversed.

    Returns:
      The initial state (from init_state) of a new traversal instance.
    """
    return cls(traversal_fn=fn, root_obj=root_obj).initial_state()


_T = TypeVar("_T")


@dataclasses.dataclass(frozen=True)
class SubTraversalResult:
  __slots__ = ("values", "metadata", "path_elements")
  values: List[Any]
  metadata: Any
  path_elements: daglish.PathElements


@dataclasses.dataclass(frozen=True)
class State:
  """Contains a current traversal state.

  Attributes:
    traversal: Reference to main traversal object.
    current_path: A path that can be followed to the current object. In the case
      of shared objects, there will be other paths to the current object, and
      often these are determined by a somewhat arbitrary DAG traversal order.
  """
  __slots__ = ("traversal", "current_path", "_value", "_parent")

  traversal: Traversal
  current_path: daglish.Path

  # Implementation note: Please don't use _value outside of the @property
  # accessors.
  _value: Any  # pylint: disable=invalid-name

  _parent: Optional[State]

  @property
  def _object_id(self) -> int:
    return id(self._value)

  @property
  def _is_memoizable(self) -> bool:
    return daglish.is_memoizable(self._value)

  @property
  def ancestors_inclusive(self) -> Iterable[State]:
    """Gets ancestors, including the current state."""
    yield self
    if self._parent is not None:
      yield from self._parent.ancestors_inclusive

  def get_all_paths(self, allow_caching: bool = True) -> List[daglish.Path]:
    """Gets all paths to the current value.

    Args:
      allow_caching: Whether paths can be cached. Most traverers will compute
        paths to all objects from the root config object. But this can fail to
        reflect objects that are being added by the traversal, so occasionally
        one may want to take a big performance hit to reflect any newly added
        objects.

    Returns:
      List of paths.
    """
    base_paths: List[daglish.Path] = [()]
    suffix_len = 0  # number of elements from `self.current_path` to append
    # pylint: disable=protected-access
    for state in self.ancestors_inclusive:
      if state._is_memoizable:
        base_paths = self.traversal.all_paths_to_object(
            state._object_id, allow_caching=allow_caching)
        break
      else:
        suffix_len += 1
    # pylint: enable=protected-access
    path_suffix = self.current_path[-suffix_len:] if suffix_len > 0 else ()
    return [path + path_suffix for path in base_paths]

  def is_traversable(self, value: Any) -> bool:
    """Returns whether a value is traversable."""
    return self.traversal.find_node_traverser(type(value)) is not None

  def _flattened_map_children(
      self, value: Any,
      node_traverser: daglish.NodeTraverser) -> SubTraversalResult:
    """Shared function for map_children / flattened_map_children."""
    subvalues, metadata = node_traverser.flatten(value)
    path_elements = node_traverser.path_elements(value)
    new_subvalues = [
        self.call(subvalue, path_element)
        for subvalue, path_element in zip(subvalues, path_elements)
    ]
    return SubTraversalResult(
        values=new_subvalues, metadata=metadata, path_elements=path_elements)

  def map_children(self, value: _T) -> _T:
    """Maps over children for traversable values, otherwise returns it.

    Args:
      value: Value to map over. Non-traversable values are returned unmodified.

    Returns:
      A mapped value of the same type.
    """
    node_traverser = self.traversal.find_node_traverser(type(value))
    if node_traverser is None:
      return value
    result = self._flattened_map_children(value, node_traverser)
    return node_traverser.unflatten(result.values, result.metadata)

  def flattened_map_children(self, value: Any) -> SubTraversalResult:
    """Maps over children for traversable values, but doesn't unflatten results.

    Unlike `map_children`, we've found that there's not an easy natural choice
    for handling the case where `value` is not a traversable type. Generally
    calling code expects a SubTraversalResult, which contains an iterable of
    values and metadata. The transformation of these values and choice of
    metadata is non-trivial, and client code would probably have to branch on it
    anyway. Therefore, we throw an error for non-traversable values; please test
    whether `value` is traversable beforehand.

    Args:
      value: Value to map over.

    Returns:
      Sub-traversal results.

    Raises:
      ValueError: If `value` is not traversable. Please test beforehand by
      calling `state.is_traversable()`.
    """
    node_traverser = self.traversal.find_node_traverser(type(value))
    if node_traverser is None:
      raise ValueError("Please handle non-traversable values yourself.")
    return self._flattened_map_children(value, node_traverser)

  def call(self, value, *additional_path: daglish.PathElement):
    """Low-level function to execute a sub-traversal.

    This creates a new state, and then applies the function bound to the
    traverser.

    Args:
      value: Sub-value to run the traversal on.
      *additional_path: Additional path elements relating the sub-value to the
        value referred to by `self`.

    Returns:
      Result of sub-traversal (whatever `self.traversal.traversal_fn` returns).
    """
    new_state = State(self.traversal, (*self.current_path, *additional_path),
                      value, self)
    return self.traversal.apply(value, new_state)


@dataclasses.dataclass
class BasicTraversal(Traversal):
  """Basic traversal.

  This traversal will go through shared objects multiple times, for each path
  that they are reachable from.
  """

  paths_cache: Dict[int, List[daglish.Path]] = dataclasses.field(
      default_factory=dict)

  def apply(self, value, state):
    return self.traversal_fn(value, state)

  def all_paths_to_object(self, object_id: int,
                          allow_caching: bool) -> List[daglish.Path]:
    if allow_caching and object_id in self.paths_cache:
      return self.paths_cache[object_id]
    else:
      all_paths = self.paths_cache = daglish.collect_paths_by_id(
          self.root_obj, memoizable_only=True)
      return all_paths[object_id]


@dataclasses.dataclass
class MemoizedTraversal(BasicTraversal):
  """Traversal that memoizes results."""

  memo: Dict[int, Any] = dataclasses.field(default_factory=dict)

  def apply(self, value, state):
    if id(value) in self.memo:
      return self.memo[id(value)]
    else:
      result = self.memo[id(value)] = self.traversal_fn(value, state)
      return result
