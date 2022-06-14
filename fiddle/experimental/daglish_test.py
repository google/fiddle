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


class TraverseWithPathTest(absltest.TestCase):

  def test_is_namedtuple(self):
    self.assertTrue(daglish.is_namedtuple_subclass(SampleNamedTuple))
    typing_namedtuple = SampleNamedTuple(1, 2)
    self.assertTrue(daglish.is_namedtuple_instance(typing_namedtuple))
    collections_namedtuple_type = collections.namedtuple(
        "CollectionsNamedTuple", ["arg1", "arg2"])
    self.assertTrue(daglish.is_namedtuple_subclass(collections_namedtuple_type))
    collections_namedtuple_instance = collections_namedtuple_type(1, 2)
    self.assertTrue(
        daglish.is_namedtuple_instance(collections_namedtuple_instance))

  def test_pretraversal_return_none(self):
    config = fdl.Config(
        Foo,
        bar=[1, {
            "key": (2,)
        }],
        baz=SampleNamedTuple(
            fizz=fdl.Config(Foo, bar=(1,), baz="boop"), buzz=None))

    visited_values = {}

    def traverse(path, value):
      visited_values[daglish.path_str(path)] = value
      return (yield)  # Continue traversal.

    output = daglish.traverse_with_path(traverse, config)

    expected = {
        "": config,
        ".bar": config.bar,
        ".bar[0]": config.bar[0],
        ".bar[1]": config.bar[1],
        ".bar[1]['key']": config.bar[1]["key"],
        ".bar[1]['key'][0]": config.bar[1]["key"][0],
        ".baz": config.baz,
        ".baz.fizz": config.baz.fizz,
        ".baz.fizz.bar": config.baz.fizz.bar,
        ".baz.fizz.bar[0]": config.baz.fizz.bar[0],
        ".baz.fizz.baz": config.baz.fizz.baz,
        ".baz.buzz": config.baz.buzz
    }
    self.assertEqual(expected.items(), visited_values.items())
    self.assertIsNot(config, output)
    self.assertEqual(config, output)

  def test_pretraversal_return(self):
    config = SampleNamedTuple(fizz=[1, 2, 3], buzz=(4, 5, 6))

    def traverse(path, value):
      del value
      if daglish.path_str(path) == ".fizz":
        return "fizz!"
      elif daglish.path_str(path) == ".buzz":
        return "buzz!"
      return (yield)

    output = daglish.traverse_with_path(traverse, config)
    expected = SampleNamedTuple(fizz="fizz!", buzz="buzz!")
    self.assertEqual(expected, output)

  def test_posttraversal_return(self):
    config = fdl.Config(
        Foo,
        bar=[1, {
            "key": (2,)
        }],
        baz=SampleNamedTuple(
            fizz=fdl.Config(Foo, bar=(1,), baz="boop"), buzz=None))

    def traverse(path, value):
      new_value = yield
      output = {daglish.path_str(path): value}
      traverser = daglish.find_node_traverser(type(value))
      if traverser:
        elements, _ = traverser.flatten(new_value)
        for element in elements:
          output.update(element)
      return output

    output = daglish.traverse_with_path(traverse, config)

    expected = {
        "": config,
        ".bar": config.bar,
        ".bar[0]": config.bar[0],
        ".bar[1]": config.bar[1],
        ".bar[1]['key']": config.bar[1]["key"],
        ".bar[1]['key'][0]": config.bar[1]["key"][0],
        ".baz": config.baz,
        ".baz.fizz": config.baz.fizz,
        ".baz.fizz.bar": config.baz.fizz.bar,
        ".baz.fizz.bar[0]": config.baz.fizz.bar[0],
        ".baz.fizz.baz": config.baz.fizz.baz,
        ".baz.buzz": config.baz.buzz
    }
    self.assertEqual(expected.items(), output.items())

  def test_yield_non_none_error(self):
    config = SampleNamedTuple(fizz=[1, 2, 3], buzz=(4, 5, 6))

    def traverse(unused_path, value):
      yield value

    msg = r"The traversal function yielded a non-None value\."
    with self.assertRaisesRegex(RuntimeError, msg):
      daglish.traverse_with_path(traverse, config)

  def test_yield_twice_error(self):

    def traverse(unused_path, unused_value):
      yield
      yield

    msg = "Does the traversal function have two yields?"
    with self.assertRaisesRegex(RuntimeError, msg):
      daglish.traverse_with_path(traverse, [])

  def test_doc_example(self):
    structure = {
        "a": [1, 2],
        "b": (1, 2, 3),
    }

    def replace_twos_and_tuples(unused_path, value):
      if value == 2:
        return None
      elif isinstance(value, tuple):
        return "used to be a tuple..."

      # Provides the post-traversal value! Here, any sub-nest value of 2 has
      # already been mapped to None.
      new_value = yield

      if isinstance(new_value, list):
        return ["used to be a two..." if x is None else x for x in new_value]
      else:
        return new_value

    output = daglish.traverse_with_path(replace_twos_and_tuples, structure)

    assert output == {
        "a": [1, "used to be a two..."],
        "b": "used to be a tuple...",
    }

  def test_defaultdict(self):
    defaultdict = collections.defaultdict(list)
    defaultdict["a"].append(1)
    defaultdict["b"].extend([1, 2])

    identity_traverse = lambda path, value: (yield)
    output = daglish.traverse_with_path(identity_traverse, defaultdict)
    self.assertIsInstance(output, collections.defaultdict)
    self.assertIs(output.default_factory, list)
    self.assertEqual(defaultdict, output)

    value_by_path = daglish.collect_value_by_path(
        defaultdict, memoizable_only=False)
    value_by_path_str = {
        daglish.path_str(path): value for path, value in value_by_path.items()
    }
    expected = {
        "['a']": [1],
        "['a'][0]": 1,
        "['b']": [1, 2],
        "['b'][0]": 1,
        "['b'][1]": 2,
        "": defaultdict
    }
    self.assertEqual(expected, value_by_path_str)


class TraverseWithAllPathsTest(absltest.TestCase):

  def test_collect_paths(self):
    shared_config = fdl.Config(Foo, bar=1, baz=2)
    shared_list = [1, 2]
    config = fdl.Config(
        Foo,
        bar=(shared_list, shared_config),
        baz=[shared_list, shared_config],
    )

    path_to_all_paths = {}

    def traverse(all_paths, current_path, unused_value):
      path_to_all_paths[daglish.path_str(current_path)] = [
          daglish.path_str(path) for path in all_paths
      ]
      yield

    output = daglish.traverse_with_all_paths(traverse, config)

    expected = {
        "": [""],
        ".bar": [".bar"],
        ".bar[0]": [".bar[0]", ".baz[0]"],
        ".bar[0][0]": [".bar[0][0]", ".baz[0][0]"],
        ".bar[0][1]": [".bar[0][1]", ".baz[0][1]"],
        ".bar[1]": [".bar[1]", ".baz[1]"],
        ".bar[1].bar": [".bar[1].bar", ".baz[1].bar"],
        ".bar[1].baz": [".bar[1].baz", ".baz[1].baz"],
        ".baz": [".baz"],
        ".baz[0]": [".bar[0]", ".baz[0]"],
        ".baz[0][0]": [".bar[0][0]", ".baz[0][0]"],
        ".baz[0][1]": [".bar[0][1]", ".baz[0][1]"],
        ".baz[1]": [".bar[1]", ".baz[1]"],
        ".baz[1].bar": [".bar[1].bar", ".baz[1].bar"],
        ".baz[1].baz": [".bar[1].baz", ".baz[1].baz"],
    }
    self.assertEqual(expected.items(), path_to_all_paths.items())
    self.assertIsNone(output)

  def test_posttraversal_return_old_value(self):
    shared_config = fdl.Config(Foo, bar=1, baz=2)
    shared_list = [1, 2]
    config = fdl.Config(
        Foo,
        bar=(shared_list, shared_config),
        baz=[shared_list, shared_config],
    )

    def traverse(unused_all_paths, unused_current_path, value):
      yield
      return value

    output = daglish.traverse_with_all_paths(traverse, config)
    self.assertIs(config.bar[0], output.bar[0])
    self.assertIs(config.bar[1], output.bar[1])
    self.assertIs(config.baz[0], output.baz[0])
    self.assertIs(config.baz[1], output.baz[1])
    self.assertIs(output.bar[0], output.baz[0])
    self.assertIs(output.bar[1], output.baz[1])

  def test_posttraversal_return_new_value(self):
    shared_config = fdl.Config(Foo, bar=1, baz=2)
    shared_list = [1, 2]
    config = fdl.Config(
        Foo,
        bar=(shared_list, shared_config),
        baz=[shared_list, shared_config],
    )

    def traverse(unused_all_paths, unused_current_path, unused_value):
      return (yield)

    output = daglish.traverse_with_all_paths(traverse, config)
    self.assertIsNot(config.bar[0], output.bar[0])
    self.assertIsNot(config.bar[1], output.bar[1])
    self.assertIsNot(config.baz[0], output.baz[0])
    self.assertIsNot(config.baz[1], output.baz[1])
    self.assertIsNot(output.bar[0], output.baz[0])
    self.assertIsNot(output.bar[1], output.baz[1])


class MemoizedTraverseTest(absltest.TestCase):

  def test_memoizes_values(self):
    shared_config = fdl.Config(Foo, bar=1, baz=2)
    shared_list = [1, 2]
    config = fdl.Config(
        Foo,
        bar=(shared_list, shared_config),
        baz=[shared_list, shared_config],
    )

    def traverse(unused_all_paths, unused_value):
      return (yield)

    output = daglish.memoized_traverse(traverse, config)
    self.assertIsNot(config.bar[0], output.bar[0])
    self.assertIsNot(config.bar[1], output.bar[1])
    self.assertIsNot(config.baz[0], output.baz[0])
    self.assertIsNot(config.baz[1], output.baz[1])
    self.assertIs(output.bar[0], output.baz[0])
    self.assertIs(output.bar[1], output.baz[1])


class CollectPathsByIdTest(absltest.TestCase):

  def test_empty_structure(self):
    for root in [[], {}, fdl.Config(Foo)]:
      self.assertEqual(
          daglish.collect_paths_by_id(root, True), {id(root): [()]})
    # Emtpy tuple is not memoizable:
    self.assertEqual(daglish.collect_paths_by_id((), True), {})

  def test_tagged_value(self):
    tagged_value = SampleTag.new(1)
    expected = {
        id(tagged_value): [()],
        id(tagged_value.tags): [(daglish.Attr("tags"),)]
    }
    paths_by_id = daglish.collect_paths_by_id(tagged_value, True)
    self.assertEqual(paths_by_id, expected)

  def test_collect_paths_by_id(self):
    shared_config = fdl.Config(Foo, bar=1, baz=2)
    shared_list = [1, 2]
    config = fdl.Config(
        Foo,
        bar=(shared_list, shared_config),
        baz=[shared_list, shared_config],
    )

    paths_by_id = daglish.collect_paths_by_id(config, memoizable_only=True)
    expected = {
        id(config): [()],
        id(config.bar): [(daglish.Attr("bar"),)],
        id(config.baz): [(daglish.Attr("baz"),)],
        id(shared_list): [(daglish.Attr("bar"), daglish.Index(0)),
                          (daglish.Attr("baz"), daglish.Index(0))],
        id(shared_config): [(daglish.Attr("bar"), daglish.Index(1)),
                            (daglish.Attr("baz"), daglish.Index(1))],
    }
    self.assertEqual(paths_by_id, expected)

    with self.assertRaises(ValueError):
      paths_by_id = daglish.collect_paths_by_id(config, memoizable_only=False)


class CollectValueByIdTest(absltest.TestCase):

  def test_empty_structure(self):
    for root in [[], {}, fdl.Config(Foo)]:
      self.assertEqual(
          daglish.collect_value_by_id(root, False), {id(root): root})
      self.assertEqual(
          daglish.collect_value_by_id(root, True), {id(root): root})
    # Empty tuple is not memoizable:
    self.assertEqual(daglish.collect_value_by_id((), True), {})
    self.assertEqual(daglish.collect_value_by_id((), False), {id(()): ()})

  def test_tagged_value(self):
    tagged_value = SampleTag.new(1)
    expected = {
        id(tagged_value): tagged_value,
        id(tagged_value.tags): {SampleTag},
        id(tagged_value.value): 1
    }
    value_by_id = daglish.collect_value_by_id(tagged_value, False)
    self.assertEqual(value_by_id, expected)

  def test_collect_value_by_id(self):
    shared_config = fdl.Config(Foo, bar=1, baz=2)
    shared_list = [[], ()]
    cfg = fdl.Config(
        Foo,
        bar=(shared_list, shared_config),
        baz=[shared_list, shared_config],
    )

    id_to_value = daglish.collect_value_by_id(cfg, memoizable_only=True)
    expected = [
        cfg,
        cfg.bar,
        cfg.baz,
        shared_list,
        shared_config,
        shared_list[0],
    ]
    self.assertCountEqual(id_to_value.keys(), [id(value) for value in expected])
    for value in expected:
      self.assertIs(id_to_value[id(value)], value)

    id_to_value = daglish.collect_value_by_id(cfg, memoizable_only=False)
    expected.extend([
        shared_list[1],
        shared_config.bar,
        shared_config.baz,
    ])
    self.assertCountEqual(id_to_value.keys(), [id(value) for value in expected])
    for value in expected:
      self.assertIs(id_to_value[id(value)], value)


class CollectValueByPathTest(absltest.TestCase):

  def test_empty_structure(self):
    for root in [[], {}, fdl.Config(Foo)]:
      self.assertEqual(daglish.collect_value_by_path(root, False), {(): root})
      self.assertEqual(daglish.collect_value_by_path(root, True), {(): root})
    # Empty tuple is not memoizable:
    self.assertEqual(daglish.collect_value_by_path((), True), {})
    self.assertEqual(daglish.collect_value_by_path((), False), {(): ()})

  def test_tagged_value(self):
    tagged_value = SampleTag.new(1)
    expected = {
        (): tagged_value,
        (daglish.Attr("tags"),): {SampleTag},
        (daglish.Attr("value"),): 1
    }
    value_by_path = daglish.collect_value_by_path(tagged_value, False)
    self.assertEqual(value_by_path, expected)

  def test_collect_value_by_path(self):
    shared_config = fdl.Config(Foo, bar=1, baz=2)
    shared_list = [[], ()]
    cfg = fdl.Config(
        Foo,
        bar=(shared_list, shared_config),
        baz=[shared_list, shared_config],
    )

    value_by_path = daglish.collect_value_by_path(cfg, memoizable_only=True)
    expected = {
        ():
            cfg,
        (daglish.Attr("bar"),):
            cfg.bar,
        (daglish.Attr("bar"), daglish.Index(0)):
            cfg.bar[0],
        (daglish.Attr("bar"), daglish.Index(0), daglish.Index(0)):
            cfg.bar[0][0],
        (daglish.Attr("bar"), daglish.Index(1)):
            cfg.bar[1],
        (daglish.Attr("baz"),):
            cfg.baz,
        (daglish.Attr("baz"), daglish.Index(0)):
            cfg.baz[0],
        (daglish.Attr("baz"), daglish.Index(0), daglish.Index(0)):
            cfg.baz[0][0],
        (daglish.Attr("baz"), daglish.Index(1)):
            cfg.baz[1],
    }
    self.assertEqual(value_by_path, expected)
    for path in value_by_path:
      self.assertIs(value_by_path[path], expected[path])

    value_by_path = daglish.collect_value_by_path(cfg, memoizable_only=False)
    expected.update({
        (daglish.Attr("bar"), daglish.Index(0), daglish.Index(1)):
            cfg.bar[0][1],
        (daglish.Attr("bar"), daglish.Index(1), daglish.Attr("bar")):
            cfg.bar[1].bar,
        (daglish.Attr("bar"), daglish.Index(1), daglish.Attr("baz")):
            cfg.bar[1].baz,
        (daglish.Attr("baz"), daglish.Index(0), daglish.Index(1)):
            cfg.baz[0][1],
        (daglish.Attr("baz"), daglish.Index(1), daglish.Attr("bar")):
            cfg.baz[1].bar,
        (daglish.Attr("baz"), daglish.Index(1), daglish.Attr("baz")):
            cfg.baz[1].baz,
    })
    self.assertEqual(value_by_path, expected)
    for path in value_by_path:
      self.assertIs(value_by_path[path], expected[path])


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
