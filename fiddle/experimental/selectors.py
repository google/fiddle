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

"""Library for manipulating selections of a Buildable DAG.

A common need for configuration libraries is to override settings in some kind
of base configuration, and these APIs allow such overrides to take place
imperatively.
"""

import abc
import copy
import dataclasses
from typing import Any, Callable, Iterator, Optional, Type, Union

from fiddle import config
from fiddle import daglish
from fiddle import tag_type
from fiddle._src import mutate_buildable

# Maybe DRY up with type declaration in autobuilders.py?
FnOrClass = Union[Callable[..., Any], Type[Any]]


class Selection(metaclass=abc.ABCMeta):
  """Base class for selections of nodes/objects/values in a config DAG."""

  def __iter__(self) -> Iterator[Any]:
    """Iterates over the selected values."""
    raise NotImplementedError(f"Iteration is not supported for {type(self)}")

  @abc.abstractmethod
  def replace(self, value, deepcopy: bool = True) -> None:
    """Replaces all selected nodes/objects/values with a new value.

    Args:
      value: Value to replace selected nodes/objects/values with.
      deepcopy: Whether to deepcopy `value` every time it is set.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def set(self, /, **kwargs) -> None:
    """Sets attributes on nodes matching this selection.

    Args:
      **kwargs: Attributes to set on matching nodes.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def get(self, name: str) -> Iterator[Any]:
    """Gets all values for a particular attribute.

    Args:
      name: Name of the attribute on matching nodes.

    Yields:
      Values configured for the attribute with name `name` on matching nodes.
    """
    raise NotImplementedError()


def _memoized_walk_leaves_first(value, state=None):
  """Yields all values (memoized) from a configuration DAG."""
  state = state or daglish.MemoizedTraversal.begin(_memoized_walk_leaves_first,
                                                   value)
  if state.is_traversable(value):
    for sub_result in state.flattened_map_children(value).values:
      yield from sub_result
  yield value


@dataclasses.dataclass(frozen=True)
class NodeSelection(Selection):
  """Represents a selection of nodes.

  This selection is declarative, so if subtrees / subgraphs of `cfg` change and
  later match or don't match, a different set of nodes will be returned.

  Generally this class is intended for modifying attributes of a buildable DAG
  in a way that doesn't alter its structure. We do not pay particular attention
  to structure-altering modifications right now; please do not depend on such
  behavior.
  """
  cfg: config.Buildable
  fn_or_cls: Optional[FnOrClass]
  match_subclasses: bool
  buildable_type: Type[config.Buildable]

  def _matches(self, node: config.Buildable) -> bool:
    """Helper for __iter__ function, determining if a node matches."""

    # Implementation note: To allow for future expansion of this class, checks
    # here should be expressed as `if not my_matcher.match(x): return False`.

    if not isinstance(node, self.buildable_type):
      return False

    if self.fn_or_cls is not None:
      if self.fn_or_cls != node.__fn_or_cls__:
        # Determines if subclass matching is allowed, and if the node is a
        # subclass of `self.fn_or_cls`. We check whether both are instances
        # of `type` to avoid `issubclass` errors when either side is actually a
        # function.
        is_subclass = (
            self.match_subclasses  #
            and isinstance(self.fn_or_cls, type)  #
            and isinstance(node.__fn_or_cls__, type)  #
            and issubclass(node.__fn_or_cls__, self.fn_or_cls))
        if not is_subclass:
          return False

    return True

  def __iter__(self) -> Iterator[config.Buildable]:
    """Yields all selected nodes.

    Nodes that are reachable via multiple paths are yielded only once.
    """

    for value in _memoized_walk_leaves_first(self.cfg):
      if self._matches(value):
        yield value

  def replace(self, value: Any, deepcopy: bool = True) -> None:
    if self._matches(self.cfg):
      raise ValueError(
          "NodeSelection.replace() is not supported on selections that "
          "match the root Buildable, because select() is primarily a "
          "mutation-based API that does not return new/replacement configs.")

    def traverse(node, state: daglish.State):
      if self._matches(node):
        return copy.deepcopy(value) if deepcopy else value
      elif state.is_traversable(node):
        # TODO: Consider moving this into Daglish.
        result = state.map_children(node)
        if isinstance(node, config.Buildable):
          mutate_buildable.move_buildable_internals(
              source=result, destination=node)
        else:
          node = result
      return node

    new_config = traverse(self.cfg,
                          daglish.MemoizedTraversal.begin(traverse, self.cfg))
    mutate_buildable.move_buildable_internals(
        source=new_config, destination=self.cfg)

  def set(self, /, **kwargs) -> None:
    """Sets multiple attributes on nodes matching this selection.

    Args:
      **kwargs: Properties to set on matching nodes.
    """
    for matching in self:
      for name, value in kwargs.items():
        setattr(matching, name, value)

  def get(self, name: str) -> Iterator[Any]:
    """Gets all values for a particular attribute.

    Args:
      name: Name of the attribute on matching nodes.

    Yields:
      Values configured for the attribute with name `name` on matching nodes.
    """
    for matching in self:
      yield getattr(matching, name)


@dataclasses.dataclass(frozen=True)
class TagSelection(Selection):
  """Represents a selection of fields tagged by a given tag."""

  cfg: config.Buildable
  tag: tag_type.TagType

  def __iter__(self) -> Iterator[Any]:
    """Yields all values for the selected tag."""

    for value in _memoized_walk_leaves_first(self.cfg):
      if isinstance(value, config.Buildable):
        for name, tags in value.__argument_tags__.items():
          if any(issubclass(tag, self.tag) for tag in tags):
            yield getattr(value, name)

  def replace(self, value: Any, deepcopy: bool = True) -> None:

    for node_value in _memoized_walk_leaves_first(self.cfg):
      if isinstance(node_value, config.Buildable):
        for name, tags in node_value.__argument_tags__.items():
          if any(issubclass(tag, self.tag) for tag in tags):
            to_set = value if not deepcopy else copy.deepcopy(value)
            setattr(node_value, name, to_set)

  def get(self, name: str) -> Iterator[Any]:
    raise NotImplementedError(
        "To iterate through values of a TagSelection, use __iter__ instead "
        "of get().")

  def set(self, /, **kwargs) -> None:
    raise NotImplementedError(
        "You can't set named attributes on tagged values, you can only replace "
        "them. Please call replace() instead of set().")


def select(
    cfg: config.Buildable,
    fn_or_cls: Optional[FnOrClass] = None,
    *,
    tag: Optional[tag_type.TagType] = None,
    match_subclasses: bool = True,
    buildable_type: Type[config.Buildable] = config.Buildable,
) -> Selection:
  """Selects sub-buildables or fields within a configuration DAG.

  Example configuring attention classes:

  select(my_config, MyDenseAttention).set(num_heads=12, head_dim=512)

  Example configuring all activation dtypes:

  select(my_config, tag=DType).set(value=jnp.float32)

  Args:
    cfg: Configuraiton to traverse.
    fn_or_cls: Select by a given function or class that is being configured.
    tag: If set, selects all attributes tagged by `tag`. This will return a
      TagSelection instead of a Selection, which has a slightly different API.
    match_subclasses: If fn_or_cls is provided and a class, then also match
      subclasses of `fn_or_cls`.
    buildable_type: Restrict the selection to a particular buildable type. Not
      valid for tag selections.

  Returns:
    A Selection, which is a TagSelection if `tag` is set, and a NodeSelection
    otherwise.
  """
  if tag is not None:
    if fn_or_cls is not None:
      raise NotImplementedError(
          "Selecting by tag and fn_or_cls is not supported yet.")
    if not match_subclasses:
      raise NotImplementedError(
          "match_subclasses is ignored when selecting by tag.")
    return TagSelection(cfg, tag)
  else:
    return NodeSelection(
        cfg,
        fn_or_cls,
        match_subclasses=match_subclasses,
        buildable_type=buildable_type)
