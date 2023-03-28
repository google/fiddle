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

"""Tests for signatures."""

import collections
import dataclasses
import inspect
import typing

from absl.testing import absltest
from fiddle._src import signatures
from fiddle._src.signatures_test_helper import DataclassBaseWithLocalType
import typing_extensions


@dataclasses.dataclass
class ChildClassWithNonLocalType(DataclassBaseWithLocalType):
  three: int


@dataclasses.dataclass
class TaggedDataclass:
  pseudo_tagged: typing_extensions.Annotated[int, 'some_metadata'] = 5


class SignatureCacheTest(absltest.TestCase):

  def test_signature_is_cached(self):

    def some_fn(a, b: int = 1):
      del a, b

    signatures.get_signature(some_fn)
    self.assertIn(some_fn, signatures._signature_cache)

  def test_can_get_dict_signature(self):
    signature = signatures.get_signature(dict)
    parameters = signature.parameters
    self.assertLen(parameters, 2)
    self.check_parameter(
        parameters['args'],
        name='args',
        default=inspect.Parameter.empty,
        kind=inspect.Parameter.VAR_POSITIONAL)
    self.check_parameter(
        parameters['kwargs'],
        name='kwargs',
        default=inspect.Parameter.empty,
        kind=inspect.Parameter.VAR_KEYWORD)

  def test_can_get_signature_of_unhashable_object(self):

    @dataclasses.dataclass(eq=True)  # Setting `eq=True` makes this unhashable.
    class SomeClass:

      def __call__(self, a, b: int = 1):
        del a, b

    instance = SomeClass()
    self.assertTrue(signatures.has_signature(instance))
    signature = signatures.get_signature(instance)
    self.assertIn('b', signature.parameters)

  def test_nonexistent_signature_for_uncallable_object(self):

    class SomeClass:
      pass

    self.assertFalse(signatures.has_signature(SomeClass()))
    with self.assertRaises(TypeError):
      signatures.get_signature(SomeClass())

  def test_nonexistent_signature_for_builtin(self):
    self.assertFalse(signatures.has_signature(print))
    with self.assertRaises(ValueError):
      signatures.get_signature(print)

  def check_parameter(self, param: inspect.Parameter, *, name, default, kind):
    self.assertIsInstance(param, inspect.Parameter)
    self.assertEqual(param.name, name)
    self.assertEqual(param.default, default)
    self.assertEqual(param.kind, kind)

  def test_defaults_and_kwargs(self):

    def foo(x: int, y, z=4, **kwargs):
      del y, z, kwargs
      return x

    signature = signatures.get_signature(foo)
    parameters = signature.parameters
    self.assertLen(parameters, 4)
    self.check_parameter(
        parameters['x'],
        name='x',
        default=inspect.Parameter.empty,
        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.check_parameter(
        parameters['y'],
        name='y',
        default=inspect.Parameter.empty,
        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.check_parameter(
        parameters['z'],
        name='z',
        default=4,
        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.check_parameter(
        parameters['kwargs'],
        name='kwargs',
        default=inspect.Parameter.empty,
        kind=inspect.Parameter.VAR_KEYWORD)

  def test_positional_and_kw_only(self):

    def bar(x, /, y, *, z=4):
      del y, z
      return x

    signature = signatures.get_signature(bar)
    parameters = signature.parameters
    self.assertLen(parameters, 3)
    self.check_parameter(
        parameters['x'],
        name='x',
        default=inspect.Parameter.empty,
        kind=inspect.Parameter.POSITIONAL_ONLY)
    self.check_parameter(
        parameters['y'],
        name='y',
        default=inspect.Parameter.empty,
        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.check_parameter(
        parameters['z'],
        name='z',
        default=4,
        kind=inspect.Parameter.KEYWORD_ONLY)


class TypeHintsCacheTest(absltest.TestCase):

  def test_regular_class(self):
    class RegularClass:

      def __init__(self, x: int):
        pass

    hints = signatures.get_type_hints(RegularClass)
    self.assertEqual(hints, {'x': int})

  def test_dataclass(self):
    @dataclasses.dataclass
    class SomeDataclass:
      x: int

    hints = signatures.get_type_hints(SomeDataclass)
    self.assertEqual(hints, {'x': int})

  def test_namedtuple_function(self):
    SampleTuple = collections.namedtuple('SampleTuple', ['x', 'y', 'z'])
    hints = signatures.get_type_hints(SampleTuple)

    self.assertEmpty(hints)

  def test_namedtuple_class(self):
    class SampleTuple(typing.NamedTuple):
      x: int
      y: str = 'why?'

    hints = signatures.get_type_hints(SampleTuple)
    self.assertEqual(hints, {'x': int, 'y': str})

  def test_object(self):
    hints = signatures.get_type_hints(object)
    self.assertEqual(hints, {})

  def test_int(self):
    hints = signatures.get_type_hints(int)
    self.assertEqual(hints, {})

  def test_invalid_types(self):
    class ClassWithInvalidAnnotations:

      def __init__(self, oops: 'Any'):  # pytype: disable=name-error
        pass

    hints = signatures.get_type_hints(ClassWithInvalidAnnotations)
    self.assertEqual({}, hints)

  def test_annotations_with_extras(self):
    hints = signatures.get_type_hints(TaggedDataclass, include_extras=True)
    self.assertEqual(
        hints,
        {
            'pseudo_tagged': typing_extensions.Annotated[int, 'some_metadata'],
        },
    )

  def test_annotations_without_extras(self):
    hints = signatures.get_type_hints(TaggedDataclass, include_extras=False)
    self.assertEqual(hints, {'pseudo_tagged': int})
    self.assertEqual(
        hints,
        signatures.get_type_hints(TaggedDataclass),
        'Testing function defaults.',
    )

  def test_cross_module_dataclass_with_local_type(self):
    hints = signatures.get_type_hints(ChildClassWithNonLocalType)
    self.assertEqual(hints.keys(), {'one', 'two', 'three'}, hints)


if __name__ == '__main__':
  absltest.main()
