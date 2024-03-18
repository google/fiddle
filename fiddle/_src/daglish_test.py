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
import enum
import json
import random
from typing import Any, List, NamedTuple, Optional, Tuple, cast

from absl.testing import absltest
from absl.testing import parameterized
import fiddle as fdl
from fiddle import daglish
from fiddle import history
from fiddle._src.experimental import dataclasses as fdl_dc
from fiddle._src.testing import nested_values
from fiddle._src.testing import test_util


@dataclasses.dataclass
class Foo:
  bar: Any
  baz: Any


@dataclasses.dataclass
class Foo2:
  bloop: Any


class SampleNamedTuple(NamedTuple):
  fizz: Any
  buzz: Any


class SampleTag(fdl.Tag):
  """`fdl.Tag` to use for testing."""


def sample_fn(arg, kwarg="test"):
  return arg, kwarg


def fn_with_position_args(  # pylint: disable=keyword-arg-before-vararg
    a, b, /, c=1, *args, kwarg1=None, karg2=None, **kwargs
):  # pylint: disable=unused-argument
  return locals()


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


@dataclasses.dataclass
class NonRecursingLoggingFunction:
  values_and_paths: List[Tuple[Any, Any]] = dataclasses.field(
      default_factory=list
  )

  def __call__(self, value, state: daglish.State):
    self.values_and_paths.append((value, state.current_path))


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
  if isinstance(value, fdl.Buildable):
    return fdl.ordered_arguments(value)
  elif isinstance(value, enum.Enum):
    return str(value)
  else:
    return value


@dataclasses.dataclass
class MyRange:
  start: int
  end: int


def myrange_flatten_with_paths_fn(myrange: MyRange):
  return (
      ({"value": i}, daglish.Index(i))
      for i in range(myrange.start, myrange.end)
  )


daglish.register_node_traverser(
    MyRange,
    flatten_fn=NotImplemented,  # pytype: disable=wrong-arg-types
    unflatten_fn=NotImplemented,  # pytype: disable=wrong-arg-types
    path_elements_fn=NotImplemented,  # pytype: disable=wrong-arg-types
    flatten_with_paths_fn=myrange_flatten_with_paths_fn,
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
    self.assertIs(
        daglish.follow_path(root, path7), fdl.get_callable(root[2][2]))

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

  def test_path_elements_order(self):
    a = daglish.Attr("bar")
    b = daglish.Index(1)
    c = daglish.Key("a")
    self.assertLess(a, b)
    self.assertLess(b, c)


class TraverserRegistryTest(parameterized.TestCase):

  @parameterized.named_parameters([
      ("config", fdl.Config(Foo, bar=1, baz=2)),
      (
          "positional_config",
          fdl.Config(
              fn_with_position_args,
              0,
              1,
              2,
              3,
              4,
              kwarg1=5,
              var_kwarg1=6,
              var_kwarg2=7,
          ),
      ),
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
    registry = daglish.NodeTraverserRegistry(use_fallback=False)
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

  def test_node_traverser_registry_with_fallback(self):
    registry = daglish.NodeTraverserRegistry(use_fallback=True)
    self.assertIsNotNone(registry.find_node_traverser(dict))
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

  def test_stacked_node_traverser_registry_with_fallback(self):
    registry1 = daglish.NodeTraverserRegistry(use_fallback=True)
    registry1.register_node_traverser(
        Foo,
        flatten_fn=lambda x: ((x.bar, x.baz), None),
        unflatten_fn=lambda values, _: Foo(*values),
        path_elements_fn=lambda _: (daglish.Attr("bar"), daglish.Attr("baz")))

    registry2 = daglish.NodeTraverserRegistry(use_fallback=registry1)
    registry2.register_node_traverser(
        Foo2,
        flatten_fn=lambda x: ((x.bloop,), None),
        unflatten_fn=lambda values, _: Foo(*values),
        path_elements_fn=lambda _: (daglish.Attr("bloop"),))

    self.assertIsNone(daglish.find_node_traverser(Foo))
    self.assertIsNone(daglish.find_node_traverser(Foo2))

    self.assertIsNotNone(registry1.find_node_traverser(dict))
    self.assertIsNotNone(registry1.find_node_traverser(Foo))
    self.assertIsNone(registry1.find_node_traverser(Foo2))

    self.assertIsNotNone(registry2.find_node_traverser(dict))
    self.assertIsNotNone(registry2.find_node_traverser(Foo))
    self.assertIsNotNone(registry2.find_node_traverser(Foo2))


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


_eager_map_fns = [
    lambda state, obj: state.map_children(obj),
    lambda state, obj: list(state.yield_map_child_values(obj)),
    lambda state, obj: state.flattened_map_children(obj),
]


class StateApiTest(parameterized.TestCase):

  @parameterized.parameters(_eager_map_fns)
  def test_map_dict(self, map_fn):
    obj = {"a": 1, "b": 2}
    log_calls = NonRecursingLoggingFunction()
    traversal = daglish.BasicTraversal(log_calls, obj)
    state = traversal.initial_state()
    map_fn(state, obj)
    self.assertEqual(
        log_calls.values_and_paths,
        [(1, (daglish.Key(key="a"),)), (2, (daglish.Key(key="b"),))],
    )

  @parameterized.parameters(_eager_map_fns)
  def test_map_tuple(self, map_fn):
    obj = ((), (1, 2), 3)
    log_calls = NonRecursingLoggingFunction()
    traversal = daglish.BasicTraversal(log_calls, obj)
    state = traversal.initial_state()
    map_fn(state, obj)
    self.assertEqual(
        log_calls.values_and_paths,
        [
            ((), (daglish.Index(index=0),)),
            ((1, 2), (daglish.Index(index=1),)),
            (3, (daglish.Index(index=2),)),
        ],
    )

  @parameterized.parameters(_eager_map_fns)
  def test_map_memoized(self, map_fn):
    shared = {"foo": 123}
    obj = [shared, shared, shared]
    log_calls = NonRecursingLoggingFunction()
    traversal = daglish.MemoizedTraversal(log_calls, obj)
    state = traversal.initial_state()
    map_fn(state, obj)
    self.assertEqual(
        log_calls.values_and_paths, [({"foo": 123}, (daglish.Index(index=0),))]
    )

  def test_fast_map_calls_flatten_with_paths(self):
    obj = MyRange(3, 7)
    log_calls = NonRecursingLoggingFunction()
    traversal = daglish.MemoizedTraversal(log_calls, obj)
    state = traversal.initial_state()
    for _ in state.yield_map_child_values(obj):
      pass  # Run lazy iterator.
    self.assertEqual(
        log_calls.values_and_paths,
        [
            ({"value": 3}, (daglish.Index(index=3),)),
            ({"value": 4}, (daglish.Index(index=4),)),
            ({"value": 5}, (daglish.Index(index=5),)),
            ({"value": 6}, (daglish.Index(index=6),)),
        ],
    )


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


def _iterate_path_strings(*args, **kwargs):
  return [(value, daglish.path_str(path))
          for value, path in daglish.iterate(*args, **kwargs)]


class IterateTest(absltest.TestCase):

  def test_basic_walking(self):
    shared = fdl.Config(sample_fn, "arg")
    config = {"tuple": (shared, None, shared), "int": 7}
    self.assertEqual(
        _iterate_path_strings(config), [
            (config, ""),
            ((shared, None, shared), "['tuple']"),
            (shared, "['tuple'][0]"),
            ("arg", "['tuple'][0].arg"),
            (None, "['tuple'][1]"),
            (7, "['int']"),
        ])

  def test_basic_walking_non_memoized(self):
    shared = fdl.Config(sample_fn, "arg")
    config = {"tuple": (shared, None, shared), "int": 7}
    self.assertEqual(
        _iterate_path_strings(config, memoized=False), [
            (config, ""),
            ((shared, None, shared), "['tuple']"),
            (shared, "['tuple'][0]"),
            ("arg", "['tuple'][0].arg"),
            (None, "['tuple'][1]"),
            (shared, "['tuple'][2]"),
            ("arg", "['tuple'][2].arg"),
            (7, "['int']"),
        ])

  def test_walk_dataclass_default_traverser(self):
    config = {"dataclass": Foo(3, fdl.Config(sample_fn, "arg"))}
    self.assertEqual(
        _iterate_path_strings(config), [
            (config, ""),
            (config["dataclass"], "['dataclass']"),
        ])

  def test_walk_dataclass_dataclass_aware_traverser(self):
    config = {"dataclass": Foo(3, fdl.Config(sample_fn, "arg"))}
    self.assertEqual(
        _iterate_path_strings(
            config, registry=fdl_dc.daglish_dataclass_registry
        ),
        [
            (config, ""),
            (config["dataclass"], "['dataclass']"),
            (3, "['dataclass'].bar"),
            (fdl.Config(sample_fn, "arg"), "['dataclass'].baz"),
            ("arg", "['dataclass'].baz.arg"),
        ],
    )

  def test_walk_positional_arguments(self):
    config = fdl.Config(
        fn_with_position_args,
        0,
        1,
        2,
        3,
        4,
        kwarg1=5,
        var_kwarg1=6,
        var_kwarg2=7,
    )
    self.assertEqual(
        _iterate_path_strings(config),
        [
            (config, ""),
            (config[0], "[0]"),
            (config[1], "[1]"),
            (config.c, ".c"),
            (config[3], "[3]"),
            (config[4], "[4]"),
            (config.kwarg1, ".kwarg1"),
            (config.var_kwarg1, ".var_kwarg1"),
            (config.var_kwarg2, ".var_kwarg2"),
        ],
    )


class MemoizedTraversalTest(absltest.TestCase):

  def test_unique_traversals_for_different_ids(self):
    # This test generates a bunch of new objects during traversal, which if
    # cached incorrectly will result in false positive cache hits. (This is
    # because the Python interpreter will garbage collect and reuse the
    # memory IDs.)

    def traverse(value, state: daglish.State):
      if isinstance(value, fdl.Buildable):
        return value.bar
      elif isinstance(value, list):
        result = 0
        for i in value:
          to_visit = fdl.Config(Foo, bar=i + 1, baz=2 * i + 1)
          result += state.call(to_visit, daglish.Key(i))  # pytype: disable=unsupported-operands
        return result
      else:
        raise AssertionError("Should not be reached, got " + str(value))

    result = daglish.MemoizedTraversal.run(traverse, list(range(10_000)))
    self.assertEqual(result, 50_005_000)

  def test_cycle_detection(self):
    def traverse(value, state: daglish.State):
      return state.map_children(value)

    x = [1, 2, 3]
    x.append(x)
    with self.assertRaisesRegex(
        ValueError,
        "Fiddle detected a cycle while traversing a value: "
        r"<root>\[3\] is <root>\[3\]\[3\]\.",
    ):
      daglish.MemoizedTraversal.run(traverse, x)

  def test_cycle_detection_in_fdl_build(self):
    cfg = fdl.Config(Foo, bar=fdl.Config(Foo))
    cfg.bar.bar = cfg  # pytype: disable=not-writable  # use-fiddle-overlay
    with self.assertRaisesRegex(
        ValueError,
        "Fiddle detected a cycle while traversing a value: "
        r"<root>\.bar is <root>\.bar\.bar\.bar\.",
    ):
      fdl.build(cfg)

  def test_get_all_paths_dataclass_traverser(self):
    config = {"dataclass": Foo(3, fdl.Config(sample_fn, "arg"))}
    config["alt"] = config["dataclass"].baz

    def traverse(value, state: daglish.State):
      if state.is_traversable(value):
        return {
            "paths": ", ".join(
                daglish.path_str(path)
                for path in state.get_all_paths(allow_caching=True)
            ),
            "sub_values": list(state.yield_map_child_values(value)),
        }
      else:
        return "leaf value"

    traverser = daglish.MemoizedTraversal(
        traverse, config, fdl_dc.daglish_dataclass_registry
    )
    result = traverse(config, traverser.initial_state())
    self.assertDictEqual(
        result,
        {
            "paths": "",
            "sub_values": [
                {
                    "paths": "['dataclass']",
                    "sub_values": [
                        "leaf value",
                        {
                            "paths": "['dataclass'].baz, ['alt']",
                            "sub_values": ["leaf value"],
                        },
                    ],
                },
                # This one comes from the config['alt']. It is returned as one
                # of the values from flattened_map_chidlren() but is the same
                # object as above.
                {
                    "paths": "['dataclass'].baz, ['alt']",
                    "sub_values": ["leaf value"],
                },
            ],
        },
    )


if __name__ == "__main__":
  absltest.main()
