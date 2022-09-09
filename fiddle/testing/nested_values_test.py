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

"""Tests for nested_values."""

import random

from absl.testing import parameterized
from fiddle import daglish
from fiddle.experimental import daglish_legacy
from fiddle.testing import nested_values


class NestedValuesTest(parameterized.TestCase):

  def calculate_max_container_size(self, value):
    traverser = daglish.find_node_traverser(type(value))
    children, _ = traverser.flatten(value) if traverser else ([], None)
    child_sizes = [self.calculate_max_container_size(c) for c in children]
    return max([len(children), *child_sizes])

  @parameterized.parameters([
      (None, 0),
      ([], 0),
      ([None], 1),
      ([0, (1.0,)], 2),
      ([[[[[[None]]]]]], 6),
  ])
  def test_calculate_nested_value_depth(self, value, depth):
    self.assertEqual(nested_values.calculate_nested_value_depth(value), depth)

  @parameterized.parameters(range(1, 5))
  def test_generate_max_depth_no_sharing(self, max_depth):
    for i in range(10):
      value = nested_values.generate_nested_value(
          random.Random(i), max_depth=max_depth, share_objects=False)
      depth = nested_values.calculate_nested_value_depth(value)
      self.assertLessEqual(depth, max_depth)

  @parameterized.parameters(range(1, 5))
  def test_generate_max_container_size(self, max_container_size):
    for i in range(10):
      value = nested_values.generate_nested_value(
          random.Random(i), max_container_size=max_container_size)
      max_size = self.calculate_max_container_size(value)
      self.assertLessEqual(max_size, max_container_size)

  def test_generate_can_have_shared_values(self):
    max_references = 0
    for i in range(20):
      value = nested_values.generate_nested_value(random.Random(i))
      all_paths = daglish_legacy.collect_paths_by_id(
          value, memoizable_only=True)
      max_references = max([max_references, *map(len, all_paths.values())])
    self.assertGreater(max_references, 1)

  def test_generate_shared_values_can_be_disabled(self):
    max_references = 0
    for i in range(20):
      value = nested_values.generate_nested_value(
          random.Random(i), share_objects=False)
      all_paths = daglish_legacy.collect_paths_by_id(
          value, memoizable_only=True)
      max_references = max([max_references, *map(len, all_paths.values())])
    self.assertEqual(max_references, 1)
