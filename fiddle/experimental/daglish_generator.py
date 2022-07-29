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

"""An interative API to traverse a Fiddle data structure."""

import dataclasses
import enum
from typing import Any, Dict, Generator, List, Tuple, Union

import fiddle as fdl
from fiddle.experimental import daglish

Structure = Union[fdl.Buildable, List, Tuple, Dict[Any, Any], Any]


@dataclasses.dataclass
class IterationItem:
  """A pointer into a Fiddle DAG data structure.

  An iteration item represents a particular entry that is being iterated over
  with the `iterate` API. While logically the IterationItem is merely a tuple of
  path and parent, it contains myriad convenience APIs.

  Note: when collections, buildables, and leaves are aliased, the IterationItem
  only represents one of the paths.

  Attributes:
    path: the path from root to the item referenced by this `IterationItem`.
    parent: the parent collection that contains the item referenced by this.
    path_element: the key, index, or attribute name associated with this item.
    parent_path: the path to the parent collection.
    is_set: Some collection types (e.g. `fdl.Buildable`, dictionaries) support
      the concept of paths that don't have values; `is_set` is true if the
      iterated path references a defined value.
    is_collection: If the item contains nested items, it is considered a
      collection.
    is_leaf: if the item contains no children that are iterated over with the
      daglish generator.
    value: A mutable reference to the value in the collection.
  """
  path: daglish.Path
  parent: Structure

  @property
  def path_element(self) -> daglish.PathElement:
    return self.path[-1]

  @property
  def parent_path(self) -> daglish.Path:
    return self.path[:-1]

  @property
  def is_set(self) -> bool:
    return self.path_element.is_in(self.parent)

  @property
  def is_collection(self) -> bool:
    if not self.is_set:
      return False
    # TODO: Make traverser registry pluggable!
    return daglish.find_node_traverser(type(self.value)) is not None

  @property
  def is_leaf(self) -> bool:
    return not self.is_collection

  @property
  def value(self) -> Any:
    return self.path_element.get_in(self.parent)

  @value.setter
  def value(self, new_value: Any):
    self.path_element.set_in(self.parent, new_value)

  @value.deleter
  def value(self):
    self.path_element.delete_in(self.parent)


class IterateOptions(enum.Enum):  # Should this be enum.Flag?
  SHALLOW = 1
  PRE_ORDER = 2
  POST_ORDER = 3
  # Is a PRE_AND_POST_ORDER option required?


def iterate(
    structure: Structure,
    *,
    options: IterateOptions = IterateOptions.SHALLOW
) -> Generator[IterationItem, bool, None]:
  """Iterates over `structure`."""
  yield from _iterate_implementation(structure, tuple(), options=options)


def _iterate_implementation(
    structure: Structure, path: daglish.Path, *,
    options: IterateOptions) -> Generator[IterationItem, bool, None]:
  """Recursive coroutine iteration of `structure`."""
  traverser = daglish.find_node_traverser(type(structure))
  if traverser:
    path_elements = traverser.all_path_elements(structure)
    for path_element in path_elements:
      should_recurse = None
      if options != IterateOptions.POST_ORDER:
        item_path = path + (path_element,)
        should_recurse = yield IterationItem(item_path, structure)
      if options == IterateOptions.SHALLOW:
        continue
      if should_recurse or should_recurse is None:
        try:
          child = path_element.get_in(structure)
        except Exception:  # pylint: disable=broad-except
          continue
        else:
          yield from _iterate_implementation(
              child, path + (path_element,), options=options)
      if options == IterateOptions.POST_ORDER:
        item_path = path + (path_element,)
        yield IterationItem(item_path, structure)


# Note: this is a re-implementation of `daglish.collect_paths_by_id` using
# the generaor API (for comparison purposes).
def all_paths_by_id(structure: Structure) -> dict[int, list[daglish.Path]]:
  """..."""
  paths_by_id = {}
  for item in iterate(structure):
    if item.is_set and daglish.is_memoizable(item.value):
      paths_by_id.setdefault(id(item.value), []).append(item.path)
  return paths_by_id
