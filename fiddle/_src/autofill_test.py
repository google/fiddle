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

"""Tests for Autofill functionality.
"""

import dataclasses
import inspect

from absl.testing import absltest
import fiddle as fdl
from fiddle._src import autofill as autofill_impl
from fiddle._src import autofill_test_helper as helper
from fiddle.experimental import autofill as autofill_api
import typing_extensions


@dataclasses.dataclass
class SimpleClass:
  x: int
  y: str = 'hello'


@dataclasses.dataclass
class HigherOrderClass:
  field_one: typing_extensions.Annotated[SimpleClass, autofill_api.Autofill]
  z: float = 3.0


@dataclasses.dataclass
class TopLevelClass:
  higher: typing_extensions.Annotated[HigherOrderClass, autofill_api.Autofill]
  simple: typing_extensions.Annotated[SimpleClass, autofill_api.Autofill]


class AutofillTest(absltest.TestCase):

  def test_autofill_class_hierarchy(self):
    config = fdl.Config(TopLevelClass)
    self.assertIsInstance(config, fdl.Config)
    self.assertIsInstance(config.simple, fdl.Config)
    self.assertIsInstance(config.higher, fdl.Config)
    self.assertIsInstance(config.higher.field_one, fdl.Config)

    config.higher.field_one.x = 42
    config.simple.x = 101

    result = fdl.build(config)
    expected = TopLevelClass(
        HigherOrderClass(SimpleClass(x=42)),
        SimpleClass(x=101),
    )

    self.assertEqual(result, expected)

  def test_autofill_does_not_override_arg(self):
    higher = HigherOrderClass(field_one=SimpleClass(x=42))
    config = fdl.Config(TopLevelClass, higher=higher)
    config.simple.x = 'hello'

    result = fdl.build(config)
    self.assertIs(result.higher, higher)

  def test_autofill_parameters_function(self):
    def simple_function(
        not_me: int,
        yes_me: typing_extensions.Annotated[int, autofill_api.Autofill],
        also_not_me: str,
    ):  # pylint: disable=unused-argument
      pass

    expected_parameters = {'yes_me': int}
    actual_parameters = autofill_impl.parameters_to_autofill(
        simple_function, inspect.signature(simple_function)
    )
    self.assertEqual(expected_parameters, actual_parameters)

  def test_autofill_parameters_toplevel_class(self):
    expected_parameters = {'higher': HigherOrderClass, 'simple': SimpleClass}
    actual_parameters = autofill_impl.parameters_to_autofill(
        TopLevelClass, inspect.signature(TopLevelClass)
    )
    self.assertEqual(expected_parameters, actual_parameters)

  def test_autofill_parameters_higherorder_class(self):
    expected_parameters = {'field_one': SimpleClass}
    actual_parameters = autofill_impl.parameters_to_autofill(
        HigherOrderClass, inspect.signature(HigherOrderClass)
    )
    self.assertEqual(expected_parameters, actual_parameters)

  def test_string_annotations(self):
    def simple_function(
        not_me: 'int',
        yes_me: 'typing_extensions.Annotated[int, autofill_api.Autofill]',
        also_not_me: 'typing_extensions.Annotated[float, "a-string"]',
    ):  # pylint: disable=unused-argument
      pass

    expected_parameters = {'yes_me': int}
    actual_parameters = autofill_impl.parameters_to_autofill(
        simple_function, inspect.signature(simple_function)
    )
    self.assertEqual(expected_parameters, actual_parameters)

  def test_custom_metaclass(self):
    class CustomMetaclass(type):
      pass

    class MetaclassUser(metaclass=CustomMetaclass):

      def __init__(self, x: int):
        self.x = x

    class EndUser:

      def __init__(
          self,
          mcu: typing_extensions.Annotated[
              MetaclassUser, autofill_api.Autofill
          ],
      ):
        self.mcu = mcu

    expected_parameters = {'mcu': MetaclassUser}
    actual_parameters = autofill_impl.parameters_to_autofill(
        EndUser, inspect.signature(EndUser)
    )
    self.assertEqual(expected_parameters, actual_parameters)

  def test_future_annotations(self):
    config = fdl.Config(helper.TopLevel)

    self.assertIsInstance(config, fdl.Config)
    self.assertIsInstance(config.child, fdl.Config)
    self.assertIsInstance(config.explicit, fdl.Config)

    config.child.a = 1
    config.explicit.child.a = 2
    result = fdl.build(config)

    self.assertIsInstance(result, helper.TopLevel)

  def test_nested_partials(self):
    partial_config = fdl.Partial(TopLevelClass)

    self.assertIsInstance(partial_config, fdl.Partial)
    self.assertIsInstance(partial_config.simple, fdl.ArgFactory)

    partial_config.simple.x = 1
    partial_config.higher.field_one.x = 2
    partial_top_level = fdl.build(partial_config)
    obj1: TopLevelClass = partial_top_level()
    obj2: TopLevelClass = partial_top_level()

    self.assertIsNot(obj1, obj2)
    self.assertIsNot(obj1.higher, obj2.higher)
    self.assertIsNot(obj1.higher.field_one, obj2.higher.field_one)
    self.assertIsNot(obj1.simple, obj2.simple)


if __name__ == '__main__':
  absltest.main()
