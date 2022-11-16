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
import functools
import types
from typing import Any, Dict, Callable

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


@dataclasses.dataclass
class AnotherTaggedType:
  tagged: str = fdl_dc.field(tags=AdditionalTag, default='tagged')


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


@dataclasses.dataclass
class ParentWithOptionalChild:
  child: Any = None


@dataclasses.dataclass
class ParentWithATaggedTypeChild:
  child: Any = fdl_dc.field(
      default_factory=ATaggedType, configurable_factory=True)


@dataclasses.dataclass
class A:
  x: int = 0


@dataclasses.dataclass
class B:
  a: A = fdl_dc.field(default_factory=A, configurable_factory=True)


@dataclasses.dataclass
class C:
  b: B = fdl_dc.field(default_factory=B, configurable_factory=True)

  @auto_config.auto_config
  @classmethod
  def factory(cls):
    return functools.partial(cls)

  @auto_config.auto_config
  @classmethod
  def factory2(cls):
    return functools.partial(cls, b=B())


@dataclasses.dataclass
class D:
  c_factory: Callable[..., C] = fdl_dc.field(default_factory=C.factory)

  @auto_config.auto_config
  @classmethod
  def factory(cls):
    return functools.partial(cls)


@dataclasses.dataclass
class D2:
  c_factory: Callable[..., C] = fdl_dc.field(default_factory=C.factory2)

  @auto_config.auto_config
  @classmethod
  def factory(cls):
    return functools.partial(cls)


@dataclasses.dataclass
class E:
  d_factory: Callable[..., D2] = fdl_dc.field(default_factory=D2.factory)


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

  def test_nested_dataclass_default_factories(self):
    with self.subTest('config_value'):
      cfg = fdl.Config(D)
      expected = fdl.Config(
          D, c_factory=fdl.Partial(C, fdl.ArgFactory(B, fdl.ArgFactory(A))))
      self.assertDagEqual(cfg, expected)

    with self.subTest('built_value_identity'):
      for d in [D(), fdl.build(fdl.Config(D))]:
        c1 = d.c_factory()
        c2 = d.c_factory()
        self.assertIsNot(c1, c2)
        self.assertIsNot(c1.b, c2.b)
        self.assertIsNot(c1.b.a, c2.b.a)

    with self.subTest('change_arg_factory_to_config'):
      cfg = fdl.Config(D)
      cfg.c_factory.b = fdl.Config(B)  # Now this will be shared.
      d = fdl.build(cfg)
      c1 = d.c_factory()
      c2 = d.c_factory()
      self.assertIsNot(c1, c2)
      self.assertIs(c1.b, c2.b)
      self.assertIs(c1.b.a, c2.b.a)

    with self.subTest('double_partial'):
      with self.assertRaisesRegex(ValueError, 'Unable to safely replace'):
        fdl.Config(E)

    with self.subTest('expand_dataclass_default_factories_docstring'):
      f = lambda x: x
      g = lambda v=0: [v]
      make_fn = auto_config.auto_config(lambda: functools.partial(f, x=g()))

      @dataclasses.dataclass
      class Test:
        fn: Callable[[], object] = fdl_dc.field(default_factory=make_fn)

      with self.assertRaisesRegex(ValueError, 'Unable to safely replace'):
        fdl.Partial(Test)

  def test_field_has_tag(self):
    self.assertTrue(
        fdl_dc.field_has_tag(fdl_dc.field(tags=SampleTag), SampleTag))
    self.assertTrue(
        fdl_dc.field_has_tag(
            fdl_dc.field(tags=(SampleTag, AdditionalTag)), SampleTag))
    self.assertFalse(
        fdl_dc.field_has_tag(fdl_dc.field(tags=AdditionalTag), SampleTag))
    self.assertFalse(fdl_dc.field_has_tag(fdl_dc.field(), SampleTag))
    self.assertFalse(fdl_dc.field_has_tag(dataclasses.field(), SampleTag))

  def test_update_callable_for_tagged_fields(self):
    cfg = fdl.Config(ATaggedType)
    self.assertEqual(fdl.get_tags(cfg, 'tagged'), {SampleTag})

    # When we switch to a new dataclass callable, any tags associated with
    # fields get added.
    fdl.update_callable(cfg, AnotherTaggedType)
    self.assertEqual(fdl.get_tags(cfg, 'tagged'), {SampleTag, AdditionalTag})

    # Even if we've manually adjusted the tags, they will get added.
    fdl.clear_tags(cfg, 'tagged')
    fdl.update_callable(cfg, ATaggedType)
    self.assertEqual(fdl.get_tags(cfg, 'tagged'), {SampleTag})

  def test_update_callable_for_configurable_factories(self):

    with self.subTest('add_configurable_factory'):
      # fdl.update_callable will add configurable factories for any fields that
      # do not have any (explicit) value.
      cfg = fdl.Config(ParentWithOptionalChild)
      self.assertIsNone(cfg.child)
      fdl.update_callable(cfg, Parent)
      self.assertEqual(fdl.get_callable(cfg.child), AnAutoconfigType)

    with self.subTest('do_not_overwrite_explicit_value'):
      # This example differs from the one above in that child is *explicitly*
      # set to `None`, so it won't get overwritten.
      cfg = fdl.Config(ParentWithOptionalChild, child=None)
      fdl.update_callable(cfg, Parent)
      self.assertIsNone(cfg.child)

    with self.subTest('do_not_overwrite_previous_configurable_factory'):
      cfg = fdl.Config(ParentWithATaggedTypeChild)
      self.assertEqual(fdl.get_callable(cfg.child), ATaggedType)
      fdl.update_callable(cfg, Parent)
      self.assertEqual(fdl.get_callable(cfg.child), ATaggedType)

    with self.subTest('do_not_delete_configruable_factory'):
      # In this test, we change to a class whose default value for `child` is
      # None; but we leave the fdl.Config built with the configurable factory.
      cfg = fdl.Config(ParentWithATaggedTypeChild)
      fdl.update_callable(cfg, ParentWithOptionalChild)
      self.assertEqual(fdl.get_callable(cfg.child), ATaggedType)


if __name__ == '__main__':
  absltest.main()
