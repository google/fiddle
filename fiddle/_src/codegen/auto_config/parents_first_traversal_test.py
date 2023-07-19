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

"""Tests for parents_first_traversal."""

import random
from typing import Any, List

from absl.testing import absltest
from fiddle import daglish
from fiddle._src.codegen.auto_config import parents_first_traversal
from fiddle._src.testing import nested_values


def get_traversal_order(structure) -> List[Any]:
  result = []

  def traverse_fn(value, parent_results):
    del parent_results  # unused
    result.append(value)

  parents_first_traversal.traverse_parents_first(traverse_fn, structure)
  return result


class ParentsFirstTraversalTest(absltest.TestCase):

  def test_traverse_deferral_required(self):
    shared = ["b"]
    x = [["a"], [[shared]]]
    y = [shared]
    config = [x, y]
    order = get_traversal_order(config)
    self.assertEqual(
        order,
        [
            config,
            x,
            ["a"],  # x[0]
            "a",
            [[["b"]]],  # x[1]
            [["b"]],  # x[1][0]
            [["b"]],  # y  (same in value as x[0] but a different object)
            ["b"],
            "b",
        ],
    )

    normal_order = [value for value, _ in daglish.iterate(config)]
    self.assertEqual(
        normal_order,
        [
            config,
            x,
            ["a"],
            "a",
            [[["b"]]],
            [["b"]],
            ["b"],  # Shared value is visited before all of its parents.
            "b",
            y,
        ],
    )

  def test_can_traverse_all(self):
    for rng_seed in range(100):
      with self.subTest(f"rng_seed={rng_seed}"):
        rng = random.Random(rng_seed)
        structure = nested_values.generate_nested_value(rng)
        get_traversal_order(structure)

  def test_parent_results(self):
    shared = {"a": 1, "b": 2}
    x = {"a": 3, "b": shared}
    y = {"a": 4, "b": shared}
    config = {"a": x, "b": y}

    def traverse_fn(value, parent_results):
      if value is shared:
        self.assertLen(parent_results, 2)
        self.assertIn(x, parent_results)
        self.assertIn(y, parent_results)
      return value

    parents_first_traversal.traverse_parents_first(traverse_fn, config)


if __name__ == "__main__":
  absltest.main()
