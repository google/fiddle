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

"""Tests for daglish_traversal."""

import dataclasses
import json
import random
from typing import Any, Iterable, List, Optional, Type

from absl.testing import absltest
from absl.testing import parameterized
import fiddle as fdl
from fiddle.experimental import daglish
from fiddle.experimental import daglish_traversal
from fiddle.testing import nested_values


def test_fn(arg, kwarg='test'):
  return arg, kwarg


@dataclasses.dataclass
class TraversalLoggingMapFunction:
  call_args: List[Any] = dataclasses.field(default_factory=list)
  some_path_args: List[Any] = dataclasses.field(default_factory=list)
  all_path_args: List[Any] = dataclasses.field(default_factory=list)

  def __call__(self, value, state: Optional[daglish_traversal.State] = None):
    state = state or daglish_traversal.MemoizedTraversal.begin(self, value)
    self.call_args.append(value)
    self.some_path_args.append(state.current_path)
    self.all_path_args.append(state.get_all_paths())
    return state.map_children(value)


def switch_buildables_to_args(value,
                              state: Optional[daglish_traversal.State] = None):
  """Replaces buildables with their arguments dictionary.

  Args:
    value: Nested configuration structure.
    state: Traversal state.

  Returns:
    `value` with buildables replaced by their arguments dictionary.
  """
  state = state or daglish_traversal.MemoizedTraversal.begin(
      switch_buildables_to_args, value)
  value = state.map_children(value)
  return value.__arguments__ if isinstance(value, fdl.Buildable) else value


def daglish_generate(value,
                     state: Optional[daglish_traversal.State] = None
                    ) -> Iterable[Any]:
  """Implements the Daglish generator API with daglish_traversal."""
  state = state or daglish_traversal.BasicTraversal.begin(
      daglish_generate, value)
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


@dataclasses.dataclass(frozen=True)
class Foo:
  bar: int
  baz: Any


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
    fcn(diamond, daglish_traversal.BasicTraversal.begin(fcn, diamond))
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
          'testcase_name': 'basic',
          'value': 3,
          'expected_paths': [[()],]
      },
      {
          'testcase_name':
              'two_primitives',
          'value': [3, 3],
          'expected_paths': [
              [()],
              [(daglish.Index(index=0),)],
              [(daglish.Index(index=1),)],
          ]
      },
      {
          'testcase_name':
              'shared',
          'value': (lambda shared: [[shared, shared], ()])(shared=[()]),
          'expected_paths': [
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
    fcn(value, daglish_traversal.BasicTraversal.begin(fcn, value))
    self.assertEqual(fcn.all_path_args, expected_paths)


class ArgsSwitchingFuzzTest(parameterized.TestCase):

  def test_fuzz(self):
    # Note: We could use `parameterized.test_cases`, but subTest shows up more
    # compactly in test output.
    for rng_seed in range(100):
      with self.subTest(f'rng_seed={rng_seed}'):
        rng = random.Random(rng_seed)
        structure = nested_values.generate_nested_value(rng)
        result = switch_buildables_to_args(structure)
        # Test that the result can be JSON-dumped. This will fail if there are
        # any unconverted items.
        json.dumps(result)


class GenerateTest(absltest.TestCase):

  def test_basic_walking(self):
    shared = fdl.Config(test_fn, 'arg')
    config = {'tuple': (shared, None, shared), 'int': 7}
    result = list(daglish_generate(config))
    self.assertEqual(result, [
        ('arg', (daglish.Key(key='tuple'), daglish.Index(index=0),
                 daglish.Attr(name='arg'))),
        (None, (daglish.Key(key='tuple'), daglish.Index(index=1))),
        ('arg', (daglish.Key(key='tuple'), daglish.Index(index=2),
                 daglish.Attr(name='arg'))),
        (7, (daglish.Key(key='int'),)),
    ])

  def test_walk_dataclass_default_traverser(self):
    config = {'dataclass': Foo(3, fdl.Config(test_fn, 'arg'))}
    result = list(daglish_generate(config))
    self.assertEqual(result, [
        (config['dataclass'], (daglish.Key(key='dataclass'),)),
    ])

  def test_walk_dataclass_dataclass_aware_traverser(self):
    config = {'dataclass': Foo(3, fdl.Config(test_fn, 'arg'))}
    state = daglish_traversal.MemoizedTraversal(
        daglish_generate, config, registry=dataclass_registry).initial_state()
    result = list(daglish_generate(config, state=state))
    self.assertEqual(result, [
        (3, (daglish.Key(key='dataclass'), daglish.Attr(name='bar'))),
        ('arg', (daglish.Key(key='dataclass'), daglish.Attr(name='baz'),
                 daglish.Attr(name='arg'))),
    ])


if __name__ == '__main__':
  absltest.main()
