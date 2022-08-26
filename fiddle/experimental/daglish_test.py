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

"""Tests for daglish."""

import collections
import dataclasses
from typing import Any, cast, List, NamedTuple

from absl.testing import absltest
from absl.testing import parameterized

import fiddle as fdl
from fiddle.experimental import daglish


@dataclasses.dataclass
class Foo:
  bar: Any
  baz: Any


class SampleNamedTuple(NamedTuple):
  fizz: Any
  buzz: Any


class SampleTag(fdl.Tag):
  """`fdl.Tag` to use for testing."""


class PathElementTest(absltest.TestCase):

  def test_path_fragment(self):
    path: List[daglish.PathElement] = [
        daglish.Index(index=1),
        daglish.Key(key="a"),
        daglish.Attr(name="foo"),
        daglish.Key(key=2),
        daglish.Attr(name="bar"),
        daglish.BuildableFnOrCls(),
    ]
    path_str = "".join(x.code for x in path)
    self.assertEqual(path_str, "[1]['a'].foo[2].bar.__fn_or_cls__")

  def test_follow(self):
    x = [[], {}, ()]
    self.assertIs(daglish.Index(1).follow(x), x[1])

    y = {"a": [], "b": {}, "c": ()}
    self.assertIs(daglish.Key("a").follow(y), y["a"])

    z = Foo([], {})
    self.assertIs(daglish.Attr("bar").follow(z), z.bar)

    cfg = fdl.Config(Foo)
    self.assertIs(daglish.BuildableFnOrCls().follow(cfg), Foo)

  def test_follow_path(self):
    root = [
        1, {
            "a": Foo("bar", "baz"),
            "b": SampleNamedTuple("fizz", "buzz")
        }, [3, 4, fdl.Config(Foo)]
    ]
    path1 = (daglish.Index(1), daglish.Key("a"), daglish.Attr("bar"))
    self.assertEqual(daglish.follow_path(root, path1), "bar")

    path2 = (daglish.Index(2), daglish.Index(0))
    self.assertEqual(daglish.follow_path(root, path2), 3)

    path3 = (daglish.Index(1), daglish.Key("b"), daglish.Attr("fizz"))
    self.assertEqual(daglish.follow_path(root, path3), "fizz")

    path4 = ()
    self.assertIs(daglish.follow_path(root, path4), root)

    path5 = (daglish.Index(2),)
    self.assertIs(daglish.follow_path(root, path5), root[2])

    path6 = (daglish.Index(1), daglish.Key("a"))
    self.assertIs(daglish.follow_path(root, path6), root[1]["a"])

    path7 = (daglish.Index(2), daglish.Index(2), daglish.BuildableFnOrCls())
    self.assertIs(daglish.follow_path(root, path7), root[2][2].__fn_or_cls__)

    bad_path_1 = (daglish.Key("a"), daglish.Key("b"))
    with self.assertRaisesRegex(
        ValueError, r"Key\(key='a'\) is not compatible "
        r"with root=.*"):
      daglish.follow_path(root, bad_path_1)

    bad_path_2 = (daglish.Index(2), daglish.Key("b"))
    with self.assertRaisesRegex(
        ValueError, r"Key\(key='b'\) is not compatible "
        r"with root\[2\]=\[3, 4, .*\]"):
      daglish.follow_path(root, bad_path_2)

    bad_path_3 = (daglish.Index(1), daglish.Key("a"), daglish.Attr("bam"))
    with self.assertRaisesRegex(
        ValueError, r"Attr\(name='bam'\) is not compatible "
        r"with root\[1\]\['a'\]=.*"):
      daglish.follow_path(root, bad_path_3)


class TraverserRegistryTest(parameterized.TestCase):

  @parameterized.named_parameters([
      ("config", fdl.Config(Foo, bar=1, baz=2)),
      ("tagged_value", SampleTag.new()),
      ("namedtuple", SampleNamedTuple("a", "b")),
      ("list", [1, 2, 3]),
      ("tuple", (1, 2, 3)),
      ("dict", dict(a=1, b=2)),
      ("defaultdict", collections.defaultdict(list, a=[1], b=[2])),
  ])
  def test_unflatten_flatten(self, value):
    traverser = daglish.find_node_traverser(type(value))
    self.assertIsNotNone(traverser)
    traverser = cast(daglish.NodeTraverser, traverser)
    values, metadata = traverser.flatten(value)
    unflattened_value = traverser.unflatten(values, metadata)
    self.assertIs(type(value), type(unflattened_value))
    self.assertEqual(value, unflattened_value)

  def test_unknown_node_type(self):
    self.assertIsNone(daglish.find_node_traverser(Foo))

  def test_find_node_traverser_non_type_error(self):
    with self.assertRaises(TypeError):
      daglish.find_node_traverser(cast(Any, 42))

  def test_custom_traverser_registries(self):
    registry = daglish.NodeTraverserRegistry()
    self.assertIsNone(registry.find_node_traverser(Foo))
    registry.register_node_traverser(
        Foo,
        flatten_fn=lambda x: ((x.bar, x.baz), None),
        unflatten_fn=lambda values, _: Foo(*values),
        path_elements_fn=lambda _: (daglish.Attr("bar"), daglish.Attr("baz")))
    self.assertIsNone(daglish.find_node_traverser(Foo))
    foo_traverser = registry.find_node_traverser(Foo)
    self.assertIsNotNone(foo_traverser)
    self.assertEqual(((1, 2), None), foo_traverser.flatten(Foo(1, 2)))

  def test_namedtuple_special_casing(self):
    namedtuple_traverser = daglish.find_node_traverser(daglish.NamedTupleType)
    self.assertIsNotNone(namedtuple_traverser)
    self.assertIs(namedtuple_traverser,
                  daglish.find_node_traverser(SampleNamedTuple))

  def test_register_node_traverser_non_type_error(self):
    with self.assertRaises(TypeError):
      daglish.register_node_traverser(
          cast(Any, 42),
          flatten_fn=lambda x: (tuple(x), None),
          unflatten_fn=lambda x, _: list(x),
          path_elements_fn=lambda x: (daglish.Index(i) for i in range(len(x))))

  def test_register_node_traverser_existing_registration_error(self):
    with self.assertRaises(ValueError):
      daglish.register_node_traverser(
          list,
          flatten_fn=lambda x: (tuple(x), None),
          unflatten_fn=lambda x, _: list(x),
          path_elements_fn=lambda x: (daglish.Index(i) for i in range(len(x))))

  def test_map_children(self):
    value = {"a": 1, "b": 2}
    result = daglish.map_children(lambda x: x - 1, value)
    self.assertEqual({"a": 0, "b": 1}, result)

  def test_map_children_non_traversable_error(self):
    with self.assertRaises(ValueError):
      daglish.map_children(lambda x: x, 42)


if __name__ == "__main__":
  absltest.main()
