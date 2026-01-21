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

"""Tests for collections_with_history."""

from absl.testing import absltest
from fiddle import daglish
from fiddle._src.experimental import collections_with_history


class HistoryTrackingDictTest(absltest.TestCase):

  def test_initialization(self):
    x = collections_with_history.HistoryTrackingDict({"a": 1, "b": 2})
    self.assertLen(x.__history__["a"], 1)
    self.assertLen(x.__history__["b"], 1)

  def test_setitem(self):
    x = collections_with_history.HistoryTrackingDict()
    x["c"] = 20
    self.assertLen(x.__history__["c"], 1)

  def test_delitem(self):
    x = collections_with_history.HistoryTrackingDict({"a": 1})
    del x["a"]
    self.assertEqual(x, {})
    self.assertLen(x.__history__["a"], 2)

  def test_daglish_traversal(self):
    def traverse_and_increment(value, state):
      if isinstance(value, int):
        if value > 1:
          return value + 1
        else:
          return value
      else:
        return state.map_children(value)

    container = collections_with_history.HistoryTrackingDict({"a": 1, "b": 2})
    remapped = daglish.MemoizedTraversal.run(traverse_and_increment, container)
    self.assertIsInstance(
        remapped, collections_with_history.HistoryTrackingDict
    )
    self.assertEqual(remapped, {"a": 1, "b": 3})
    self.assertLen(remapped.__history__["a"], 1)
    self.assertLen(remapped.__history__["b"], 2)


if __name__ == "__main__":
  absltest.main()
