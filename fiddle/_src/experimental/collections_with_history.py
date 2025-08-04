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

"""Subclasses of basic Python collections that support history tracking."""

from __future__ import annotations

import types
from typing import Mapping, NamedTuple, Tuple

from fiddle._src import daglish
from fiddle._src import history


class HistoryTrackingDict(dict):
  """A dictionary that tracks history of key/value assignments."""

  def __init__(self, values=None, *, __history__=None):
    values = {} if values is None else values
    super().__init__(values)
    if __history__ is None:
      self.__history__ = history.History()
      for key, value in self.items():
        self.__history__.add_new_value(key, value)
    else:
      self.__history__ = __history__
      for key, value in self.items():
        # Only add entries for unique values. Use `is` to avoid cost of deep
        # equality.
        if key in __history__ and __history__[key][-1].new_value is not value:
          self.__history__.add_new_value(key, value)

  def __setitem__(self, key, value):
    super().__setitem__(key, value)
    self.__history__.add_new_value(key, value)

  def __delitem__(self, key):
    super().__delitem__(key)
    self.__history__.add_deleted_value(key)

  def __flatten__(self):
    item_history = {
        key: tuple(entries) for key, entries in self.__history__.items()
    }
    return tuple(self.values()), _HistoryTrackingDictMetadata(
        tuple(self.keys()), item_history
    )

  @classmethod
  def __unflatten__(cls, values, metadata: _HistoryTrackingDictMetadata):
    return cls(zip(metadata.keys, values), __history__=metadata.history())


class _HistoryTrackingDictMetadata(NamedTuple):
  keys: tuple[str]
  item_history: Mapping[str, Tuple[history.HistoryEntry, ...]] = (
      types.MappingProxyType({})
  )

  def history(self) -> history.History:
    return history.History(
        {name: list(entries) for name, entries in self.item_history.items()}
    )


daglish.register_node_traverser(
    HistoryTrackingDict,
    flatten_fn=lambda x: x.__flatten__(),
    unflatten_fn=HistoryTrackingDict.__unflatten__,
    path_elements_fn=lambda x: [daglish.Key(key) for key in x.keys()],
)
