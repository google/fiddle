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

"""Tests for dataclasses."""

import dataclasses
import types
from typing import Any, Dict

from absl.testing import absltest
import fiddle as fdl
from fiddle.experimental import auto_config
from fiddle.experimental import dataclasses as fdl_dc
from fiddle.testing import test_util


class SampleTag(fdl.Tag):
  """A tag used in testing."""


class AdditionalTag(fdl.Tag):
  """A second tag to test multiple tags."""


@dataclasses.dataclass
class ATaggedType:
  untagged: str
  tagged: str = fdl_dc.field(tags=SampleTag, default='tagged')
  double_tagged: str = fdl_dc.field(
      tags=(AdditionalTag, SampleTag), default_factory=lambda: 'other_field')

  @classmethod
  @auto_config.auto_config
  def default(cls):
    return cls(untagged='untagged_default')


def test_fn():
  return 1


@auto_config.auto_config
def nested_structure():
  return {'foo': [test_fn(), (2, 3)]}


@dataclasses.dataclass
class AnAutoconfigType:
  tagged_type: ATaggedType = fdl_dc.field(default_factory=ATaggedType.default)
  another_default: Dict[str,
                        Any] = fdl_dc.field(default_factory=nested_structure)

  # We need this for `AncestorType` below, but we might be able to make
  # `auto_config.auto_config(AnAutoconfigType)` work in the future.
  @classmethod
  @auto_config.auto_config
  def default(cls):
    return cls()


@dataclasses.dataclass
class AncestorType:
  # We might want to make this more compact.
  child: AnAutoconfigType = fdl_dc.field(
      default_factory=AnAutoconfigType.default)


@dataclasses.dataclass
class Parent:
  """A class w/ a field that uses configurable_factory=True."""
  child: AnAutoconfigType = fdl_dc.field(
      default_factory=AnAutoconfigType, configurable_factory=True)
  y: int = 0


@dataclasses.dataclass
class ParentPair:
  first: Parent = fdl_dc.field(
      default_factory=Parent, configurable_factory=True)
  second: Parent = fdl_dc.field(
      default_factory=Parent, configurable_factory=True)


class DataclassesTest(test_util.TestCase):

  def test_dataclass_tagging(self):
    config = fdl.Config(ATaggedType)

    self.assertEqual({SampleTag}, fdl.get_tags(config, 'tagged'))
    self.assertEqual({SampleTag, AdditionalTag},
                     fdl.get_tags(config, 'double_tagged'))

    fdl.set_tagged(config, tag=AdditionalTag, value='set_correctly')

    self.assertEqual(config.double_tagged, 'set_correctly')

  def test_metadata_passthrough(self):
    other_metadata = types.MappingProxyType({'something': 4})
    constructed_field = fdl_dc.field(metadata=other_metadata)

    self.assertIn('something', constructed_field.metadata)
    self.assertEqual(4, constructed_field.metadata['something'])

  def test_auto_config_basic_equality(self):
    self.assertEqual(
        fdl.build(fdl.Config(AnAutoconfigType)), AnAutoconfigType())
    self.assertEqual(fdl.build(fdl.Config(AncestorType)), AncestorType())

  def test_name_set_in_both_cases(self):
    # A slightly more concrete test of the above.
    self.assertEqual(
        fdl.build(fdl.Config(AnAutoconfigType)).tagged_type.untagged,
        'untagged_default')
    self.assertEqual(AnAutoconfigType().tagged_type.untagged,
                     'untagged_default')

  def test_auto_config_override_equality(self):
    self.assertEqual(
        AnAutoconfigType(another_default={
            '3': 4
        }).another_default, {'3': 4})
    self.assertEqual(
        fdl.build(fdl.Config(AnAutoconfigType, another_default={'3': 4})),
        AnAutoconfigType(another_default={'3': 4}))

  def test_auto_config_field_init(self):
    config = fdl.Config(AnAutoconfigType)
    config.another_default['foo'][1] += (4,)
    obj = fdl.build(config)
    self.assertEqual(obj.another_default, {'foo': [1, (2, 3, 4)]})

  def test_mandatory_fields(self):

    @dataclasses.dataclass
    class TwoMandatoryFieldsDataclass:
      foo: int = fdl_dc.field(tags=SampleTag)
      bar: int

    instance = TwoMandatoryFieldsDataclass(3, 4)
    self.assertEqual(instance.foo, 3)
    self.assertEqual(instance.bar, 4)

  def test_invalid_definition_with_defaults(self):
    with self.assertRaisesRegex(
        ValueError, 'cannot specify both default and default_factory'):
      fdl_dc.field(default_factory=nested_structure, default=4)

  def test_configurable_factory(self):
    config = fdl.Config(ParentPair)
    expected_config = fdl.Config(
        ParentPair, fdl.Config(Parent, child=fdl.Config(AnAutoconfigType)),
        fdl.Config(Parent, child=fdl.Config(AnAutoconfigType)))
    self.assertDagEqual(config, expected_config)
    self.assertEqual(fdl.build(config), ParentPair())

  def test_configurable_factory_can_be_configured(self):
    # Create a config and make some changes to it.
    config = fdl.Config(ParentPair)
    config.first.y = 100
    config.second.child.another_default = {'x': 1}
    fdl.set_tagged(config, tag=SampleTag, value='changed')

    # Create a ParentPair object and make the same changes.
    expected_result = ParentPair()
    expected_result.first.y = 100
    expected_result.second.child.another_default = {'x': 1}
    expected_result.first.child.tagged_type.tagged = 'changed'
    expected_result.first.child.tagged_type.double_tagged = 'changed'
    expected_result.second.child.tagged_type.tagged = 'changed'
    expected_result.second.child.tagged_type.double_tagged = 'changed'

    self.assertEqual(fdl.build(config), expected_result)

  def test_configurable_factory_no_unintentional_aliasing(self):
    config = fdl.Config(ParentPair)
    self.assertIsNot(config.first, config.second)
    self.assertIsNot(config.first.child, config.second.child)
    self.assertIsNot(config.first.child.tagged_type,
                     config.second.child.tagged_type)
    self.assertIsNot(config.first.child.another_default,
                     config.second.child.another_default)

    val = fdl.build(config)
    self.assertIsNot(val.first, val.second)
    self.assertIsNot(val.first.child, val.second.child)
    self.assertIsNot(val.first.child.tagged_type, val.second.child.tagged_type)
    self.assertIsNot(val.first.child.another_default,
                     val.second.child.another_default)

  def test_configurable_factory_autoconfig_error(self):
    with self.assertRaisesRegex(
        ValueError, 'configurable_factory should not '
        "be used with auto_config'ed functions"):
      fdl_dc.field(
          default_factory=AnAutoconfigType.default, configurable_factory=True)


if __name__ == '__main__':
  absltest.main()
