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
import json
import random
from typing import Any, Iterable, List, NamedTuple, Optional, Type, cast

from absl.testing import absltest
from absl.testing import parameterized

import fiddle as fdl
from fiddle import daglish
from fiddle import history
from fiddle.testing import nested_values
from fiddle.testing import test_util


@dataclasses.dataclass
class Foo:
  bar: Any
  baz: Any


class SampleNamedTuple(NamedTuple):
  fizz: Any
  buzz: Any


class SampleTag(fdl.Tag):
  """`fdl.Tag` to use for testing."""


def test_fn(arg, kwarg="test"):
  return arg, kwarg


@dataclasses.dataclass
class TraversalLoggingMapFunction:
  call_args: List[Any] = dataclasses.field(default_factory=list)
  some_path_args: List[Any] = dataclasses.field(default_factory=list)
  all_path_args: List[Any] = dataclasses.field(default_factory=list)

  def __call__(self, value, state: Optional[daglish.State] = None):
    state = state or daglish.MemoizedTraversal.begin(self, value)
    self.call_args.append(value)
    self.some_path_args.append(state.current_path)
    self.all_path_args.append(state.get_all_paths())
    return state.map_children(value)


def switch_buildables_to_args(value, state: Optional[daglish.State] = None):
  """Replaces buildables with their arguments dictionary.

  Args:
    value: Nested configuration structure.
    state: Traversal state.

  Returns:
    `value` with buildables replaced by their arguments dictionary.
  """
  state = state or daglish.MemoizedTraversal.begin(switch_buildables_to_args,
                                                   value)
  value = state.map_children(value)
  return value.__arguments__ if isinstance(value, fdl.Buildable) else value


def daglish_generate(value,
                     state: Optional[daglish.State] = None) -> Iterable[Any]:
  """Implements the Daglish generator API with daglish."""
  state = state or daglish.BasicTraversal.begin(daglish_generate, value)
  if not state.is_traversable(value):
    yield value, state.current_path
  else:
    for sub_iter in state.flattened_map_children(value).values:
      yield from sub_iter


# Dataclasses iterator registry
class DataclassType:
  """Sentinel class for registry lookup."""


class DataclassAwareRegistry(daglish.NodeTraverserRegistry):

  def find_node_traverser(
      self, node_type: Type[Any]) -> Optional[daglish.NodeTraverser]:
    if dataclasses.is_dataclass(node_type):
      node_type = DataclassType
    return super().find_node_traverser(node_type)


dataclass_registry = DataclassAwareRegistry()

# Copy existing traversers.
dataclass_registry._node_traversers = (
    daglish._default_traverser_registry._node_traversers.copy())


def _flatten_dataclass(value):
  as_dict = dataclasses.asdict(value)
  return tuple(as_dict.values()), (tuple(as_dict.keys()), type(value))


def _unflatten_dataclass(values, metadata):
  keys, class_type = metadata
  return class_type(**dict(zip(keys, values)))


def _dataclass_path_elements(value):
  return [daglish.Attr(field.name) for field in dataclasses.fields(value)]


dataclass_registry.register_node_traverser(
    DataclassType,
    flatten_fn=_flatten_dataclass,
    unflatten_fn=_unflatten_dataclass,
    path_elements_fn=_dataclass_path_elements,
)


class PathTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="exact_match",
          prefix_path=".foo.bar",
          containing_path=".foo.bar",
          expected=True),
      dict(
          testcase_name="prefix",
          prefix_path=".foo.bar",
          containing_path=".foo.bar.baz",
          expected=True),
      dict(
          testcase_name="prefix_with_index",
          prefix_path=".foo.bar",
          containing_path=".foo.bar[0]",
          expected=True),
      dict(
          testcase_name="not_prefix",
          prefix_path=".foo.bar.baz",
          containing_path=".foo.bar",
          expected=False),
      dict(
          testcase_name="not_prefix_with_index",
          prefix_path=".foo.bar[0]",
          containing_path=".foo.bar",
          expected=False),
  )
  def test_is_prefix(self, prefix_path, containing_path, expected):
    prefix_path = test_util.parse_path(prefix_path)
    containing_path = test_util.parse_path(containing_path)
    self.assertEqual(daglish.is_prefix(prefix_path, containing_path), expected)


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


class BasicStructuredMappingTest(parameterized.TestCase):

  def test_memoizes_basic_diamond(self):
    shared = object()
    diamond = [shared, shared]
    fcn = TraversalLoggingMapFunction()
    fcn(diamond)
    self.assertEqual(fcn.call_args, [[shared, shared], shared])
    self.assertEqual(fcn.some_path_args, [(), (daglish.Index(index=0),)])
    self.assertEqual(fcn.all_path_args, [
        [()],
        [(daglish.Index(index=0),), (daglish.Index(index=1),)],
    ])

  def test_non_memoized_basic_diamond(self):
    shared = object()
    diamond = [shared, shared]
    fcn = TraversalLoggingMapFunction()
    fcn(diamond, daglish.BasicTraversal.begin(fcn, diamond))
    self.assertEqual(fcn.call_args, [[shared, shared], shared, shared])
    self.assertEqual(fcn.some_path_args, [
        (),
        (daglish.Index(index=0),),
        (daglish.Index(index=1),),
    ])
    self.assertEqual(fcn.all_path_args, [
        [()],
        [(daglish.Index(index=0),), (daglish.Index(index=1),)],
        [(daglish.Index(index=0),), (daglish.Index(index=1),)],
    ])

  @parameterized.named_parameters(
      {
          "testcase_name": "basic",
          "value": 3,
          "expected_paths": [[()],]
      },
      {
          "testcase_name":
              "two_primitives",
          "value": [3, 3],
          "expected_paths": [
              [()],
              [(daglish.Index(index=0),)],
              [(daglish.Index(index=1),)],
          ]
      },
      {
          "testcase_name":
              "shared",
          "value": (lambda shared: [[shared, shared], ()])(shared=[()]),
          "expected_paths": [
              [()],

              # Traversal into `[shared, shared]`.
              [(daglish.Index(index=0),)],

              # Traversal into `shared`.
              [(daglish.Index(index=0), daglish.Index(index=0)),
               (daglish.Index(index=0), daglish.Index(index=1))],

              # Traversal into the `()` of `shared`.
              [(daglish.Index(index=0), daglish.Index(index=0),
                daglish.Index(index=0)),
               (daglish.Index(index=0), daglish.Index(index=1),
                daglish.Index(index=0))],

              # Traversal into `shared`.
              [(daglish.Index(index=0), daglish.Index(index=0)),
               (daglish.Index(index=0), daglish.Index(index=1))],

              # Traversal into the `()` of `shared`.
              [(daglish.Index(index=0), daglish.Index(index=0),
                daglish.Index(index=0)),
               (daglish.Index(index=0), daglish.Index(index=1),
                daglish.Index(index=0))],

              # Traversal into `()`, not in `shared.`
              [(daglish.Index(index=1),)],
          ]
      })
  def test_all_paths(self, value, expected_paths):
    """Primitive values should not be seen as shared, but containers are."""
    fcn = TraversalLoggingMapFunction()
    fcn(value, daglish.BasicTraversal.begin(fcn, value))
    self.assertEqual(fcn.all_path_args, expected_paths)

    with self.subTest("run() classmethod alias"):
      fcn = TraversalLoggingMapFunction()
      daglish.BasicTraversal.run(fcn, value)
      self.assertEqual(fcn.all_path_args, expected_paths)

  def test_argument_history(self):
    cfg = fdl.Config(Foo)
    cfg.bar = 4
    cfg.bar = 5
    copied = TraversalLoggingMapFunction()(cfg)
    self.assertEqual(
        copied.__argument_history__["bar"][-1].location.line_number,
        cfg.__argument_history__["bar"][-1].location.line_number)
    self.assertEqual(copied.__argument_history__["bar"][0].new_value, 4)
    self.assertEqual(copied.__argument_history__["bar"][1].new_value, 5)
    self.assertEqual(copied.__argument_history__["bar"][-1].kind,
                     history.ChangeKind.NEW_VALUE)


class ArgsSwitchingFuzzTest(parameterized.TestCase):

  def test_fuzz(self):
    # Note: We could use `parameterized.test_cases`, but subTest shows up more
    # compactly in test output.
    for rng_seed in range(100):
      with self.subTest(f"rng_seed={rng_seed}"):
        rng = random.Random(rng_seed)
        structure = nested_values.generate_nested_value(rng)
        result = switch_buildables_to_args(structure)
        # Test that the result can be JSON-dumped. This will fail if there are
        # any unconverted items.
        json.dumps(result)


class GenerateTest(absltest.TestCase):

  def test_basic_walking(self):
    shared = fdl.Config(test_fn, "arg")
    config = {"tuple": (shared, None, shared), "int": 7}
    result = list(daglish_generate(config))
    self.assertEqual(result, [
        ("arg", (daglish.Key(key="tuple"), daglish.Index(index=0),
                 daglish.Attr(name="arg"))),
        (None, (daglish.Key(key="tuple"), daglish.Index(index=1))),
        ("arg", (daglish.Key(key="tuple"), daglish.Index(index=2),
                 daglish.Attr(name="arg"))),
        (7, (daglish.Key(key="int"),)),
    ])

  def test_walk_dataclass_default_traverser(self):
    config = {"dataclass": Foo(3, fdl.Config(test_fn, "arg"))}
    result = list(daglish_generate(config))
    self.assertEqual(result, [
        (config["dataclass"], (daglish.Key(key="dataclass"),)),
    ])

  def test_walk_dataclass_dataclass_aware_traverser(self):
    config = {"dataclass": Foo(3, fdl.Config(test_fn, "arg"))}
    state = daglish.MemoizedTraversal(
        daglish_generate, config, registry=dataclass_registry).initial_state()
    result = list(daglish_generate(config, state=state))
    self.assertEqual(result, [
        (3, (daglish.Key(key="dataclass"), daglish.Attr(name="bar"))),
        ("arg", (daglish.Key(key="dataclass"), daglish.Attr(name="baz"),
                 daglish.Attr(name="arg"))),
    ])


if __name__ == "__main__":
  absltest.main()
