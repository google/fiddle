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

"""Tests for fixture."""

import copy
import dataclasses

from absl.testing import absltest
import fiddle as fdl
from fiddle import building
from fiddle.experimental import fixture_node


@dataclasses.dataclass
class A:
  x: int


@dataclasses.dataclass
class B:
  a: A


@dataclasses.dataclass
class C:
  a: A


@dataclasses.dataclass
class D:
  b: B
  c: C
  name: str = ""


def diamond():
  a = fdl.Config(A, x=1)
  return fdl.Config(D, b=fdl.Config(B, a=a), c=fdl.Config(C, a=a))


def list_fixture(
    item: fdl.Buildable,
    num_items: int,
):
  return [copy.deepcopy(item) for _ in range(num_items)]


def layer_fixture(
    item: fdl.Buildable,
    num_items: int,
):
  """Demo of using the fixture to achieve functions for network layers."""
  result = []
  for i in range(num_items):
    layer = copy.deepcopy(item, memo={id(item.b.a): item.b.a})
    layer.name = f"layer_{i}"
    result.append(layer)
  return result


class FixtureTest(absltest.TestCase):

  def test_build_errors(self):
    config = fixture_node.FixtureNode(list_fixture, item=diamond(), num_items=4)
    with self.assertRaisesRegex(building.BuildError,
                                "Failed to construct or call list_fixture."):
      fdl.build(config)

  def test_list_fixture(self):
    config = fixture_node.FixtureNode(list_fixture, item=diamond(), num_items=2)

    # Can modify num_layers later/dynamically.
    config.num_items = 4

    config = fixture_node.materialize(config)
    result = fdl.build(config)
    self.assertIsInstance(result, list)
    self.assertLen(result, 4)
    for i in range(4):
      self.assertIs(result[i].b.a, result[i].c.a)
    for i in range(3):
      self.assertIsNot(result[i].b.a, result[i + 1].c.a)

  def test_layer_fixture(self):
    config = fixture_node.FixtureNode(
        layer_fixture, item=diamond(), num_items=4)
    config = fixture_node.materialize(config)
    result = fdl.build(config)
    self.assertIsInstance(result, list)
    self.assertLen(result, 4)

    # The names should vary as expected.
    self.assertEqual(result[0].name, "layer_0")
    self.assertEqual(result[1].name, "layer_1")
    self.assertEqual(result[2].name, "layer_2")
    self.assertEqual(result[3].name, "layer_3")

    # All a's should be the same here!
    for i in range(4):
      self.assertIs(result[0].b.a, result[i].b.a)
      self.assertIs(result[0].b.a, result[i].c.a)


if __name__ == "__main__":
  absltest.main()
