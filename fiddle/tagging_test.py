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

"""Tests for tagging."""

import copy
import dataclasses
from unittest import mock

from absl.testing import absltest
import fiddle as fdl
from fiddle import tagging
from fiddle import tagging_test_module as tst
from fiddle.experimental import selectors


@dataclasses.dataclass
class Foo:
  bar: int
  qux: str


class RedTag(fdl.Tag):
  """A custom tag used for testing."""


class BlueTag(fdl.Tag):
  """A custom tag used for testing."""


def return_kwargs(**kwargs):
  return kwargs


def get_single_tag(tagged_value: tagging.TaggedValueCls) -> tagging.TagType:
  """Gets a single tag from a TaggedValue, errors if there are multiple."""
  assert isinstance(tagged_value, tagging.TaggedValueCls)
  assert len(tagged_value.tags) == 1
  return next(iter(tagged_value.tags))


class TaggingTest(absltest.TestCase):

  def test_tag_str(self):
    self.assertEqual(
        str(tst.ParameterDType), "#fiddle.tagging_test_module.ParameterDType")
    self.assertEqual(
        str(tst.LinearParamDType),
        "#fiddle.tagging_test_module.LinearParamDType")
    self.assertEqual(
        str(tst.MyModel.EncoderDType),
        "#fiddle.tagging_test_module.MyModel.EncoderDType")
    self.assertEqual(
        str(tst.MyModel.DecoderDType),
        "#fiddle.tagging_test_module.MyModel.DecoderDType")

  def test_tag_instantiation(self):
    with self.assertRaisesRegex(
        TypeError, ".*cannot instantiate Fiddle tags.*ParameterDType.*"):
      _ = tst.ParameterDType()
    with self.assertRaisesRegex(
        TypeError,
        "trying to instantiate fiddle.tagging_test_module.MyModel.DecoderDType"
    ):
      _ = tst.MyModel.DecoderDType()

  def test_missing_tag_description(self):
    with self.assertRaisesRegex(TypeError,
                                "You must provide a tag description"):

      class Bar(tagging.Tag):  # pylint: disable=unused-variable
        pass

  def test_inline_tag_definition_fails(self):

    with self.assertRaisesRegex(
        TypeError, "You cannot define a tag within a function or lambda"):

      class Bar(tagging.Tag):  # pylint: disable=unused-variable
        """Bar tag."""

  def test_tagged_value_identity_fn_exception(self):
    with self.assertRaisesRegex(
        tagging.TaggedValueNotFilledError,
        "Expected.*TaggedValue.*replaced.*one was not set"):
      tagging.tagged_value_identity_fn()

  def test_tagged_value_identity_fn_none_value(self):
    # Test that when the value is None, that's OK.
    self.assertIsNone(tagging.tagged_value_identity_fn(value=None))

  def test_taggedvalue_default_none(self):
    cfg = tagging.TaggedValue(tags={tst.ParameterDType}, default=None)
    self.assertIsNone(fdl.build(cfg))

  def test_one_taggedvalue_unset_in_config(self):
    cfg = fdl.Config(
        return_kwargs,
        foo=tst.ParameterDType.new(default=None),
        bar=tst.ParameterDType.new())
    with self.assertRaisesRegex(
        tagging.TaggedValueNotFilledError,
        "Expected.*TaggedValue.*replaced.*one was not set"):
      fdl.build(cfg)

  def test_set_tagged(self):
    cfg = fdl.Config(
        return_kwargs,
        foo=tst.ParameterDType.new(default=None),
        bar=tst.LinearParamDType.new())
    tagging.set_tagged(cfg, tag=tst.LinearParamDType, value=1)
    self.assertDictEqual(fdl.build(cfg), {"foo": None, "bar": 1})

  def test_set_two_taggedvalues(self):
    cfg = fdl.Config(
        return_kwargs,
        foo=tst.ParameterDType.new(default=None),
        bar=tst.LinearParamDType.new())
    tagging.set_tagged(cfg, tag=tst.ParameterDType, value=1)
    tagging.set_tagged(cfg, tag=tst.LinearParamDType, value=2)
    self.assertDictEqual(fdl.build(cfg), {"foo": 1, "bar": 2})

  def test_double_set_tagged(self):
    cfg = fdl.Config(
        return_kwargs,
        foo=tst.ParameterDType.new(default=None),
        bar=tst.LinearParamDType.new())
    tagging.set_tagged(cfg, tag=tst.LinearParamDType, value=1)
    tagging.set_tagged(cfg, tag=tst.LinearParamDType, value=2)
    self.assertDictEqual(fdl.build(cfg), {"foo": None, "bar": 2})

  def test_set_double_tags(self):
    cfg = fdl.Config(
        return_kwargs,
        foo=tagging.TaggedValue(
            tags=(tst.ParameterDType, tst.LinearParamDType), default=None),
        bar=tst.LinearParamDType.new())
    tagging.set_tagged(cfg, tag=tst.LinearParamDType, value=4)
    self.assertEqual(fdl.build(cfg), {"foo": 4, "bar": 4})

  def test_set_tagged_subclasses(self):
    cfg = fdl.Config(
        return_kwargs,
        foo=tst.LinearParamDType.new(),
        bar=tst.ParameterDType.new())
    tagging.set_tagged(cfg, tag=tst.ParameterDType, value=42)
    self.assertEqual(fdl.build(cfg), {"foo": 42, "bar": 42})

  def test_list_tags(self):
    cfg = fdl.Config(
        return_kwargs,
        foo=tagging.TaggedValue(tags=[tst.ParameterDType], default=None),
        bar=tagging.TaggedValue(tags=[tst.LinearParamDType]))
    tags = tagging.list_tags(cfg)
    self.assertEqual(tags, {tst.LinearParamDType, tst.ParameterDType})

  def test_list_tags_multiple_tags(self):
    cfg = fdl.Config(
        return_kwargs,
        foo=tagging.TaggedValue(tags=(tst.ParameterDType, tst.ActivationDType)),
        bar=tst.LinearParamDType.new())
    tags = tagging.list_tags(cfg)
    self.assertEqual(
        tags, {tst.ParameterDType, tst.ActivationDType, tst.LinearParamDType})

  def test_list_tags_superclasses(self):
    cfg = tst.LinearParamDType.new()
    tags = tagging.list_tags(cfg, add_superclasses=True)
    self.assertEqual(tags, {tst.ParameterDType, tst.LinearParamDType})

  def test_set_only_placeholders_in_subtree(self):
    cfg = fdl.Config(
        return_kwargs,
        output_logits=fdl.Config(
            return_kwargs, dtype=tst.ActivationDType.new(default="float32")),
        encoder=fdl.Config(
            return_kwargs, dtype=tst.ActivationDType.new(default="float32")),
    )

    # At first, the defaults are used.
    self.assertDictEqual(
        fdl.build(cfg), {
            "encoder": {
                "dtype": "float32"
            },
            "output_logits": {
                "dtype": "float32"
            },
        })

    # Setting the dtype for the entire tree overrides both.
    tagging.set_tagged(cfg, tag=tst.ActivationDType, value="bfloat16")
    self.assertDictEqual(
        fdl.build(cfg), {
            "encoder": {
                "dtype": "bfloat16"
            },
            "output_logits": {
                "dtype": "bfloat16"
            },
        })

    # Setting the dtype for the output logits just overrides that one.
    tagging.set_tagged(
        cfg.output_logits, tag=tst.ActivationDType, value="float64")
    self.assertDictEqual(
        fdl.build(cfg), {
            "encoder": {
                "dtype": "bfloat16"
            },
            "output_logits": {
                "dtype": "float64"
            },
        })

  def test_deepcopy_and_object_id(self):
    """Tests that deepcopy() is fine, and key IDs are used, not names."""

    def foo(a, b, c, d):
      return a, b, c, d

    cfg = fdl.Config(foo)
    cfg.a = tst.ParameterDType.new()
    cfg.b = tst.ActivationDType.new()
    cfg.c = tst.ParameterDType.new()
    cfg.d = 4
    copied = copy.deepcopy(cfg)
    copied.d = 40

    tagging.set_tagged(cfg, tag=tst.ParameterDType, value=1)
    tagging.set_tagged(cfg, tag=tst.ActivationDType, value=2)

    self.assertEqual(fdl.build(cfg), (1, 2, 1, 4))

    with self.assertRaises(tagging.TaggedValueNotFilledError):
      fdl.build(copied)

    tagging.set_tagged(copied, tag=tst.ParameterDType, value=10)
    tagging.set_tagged(copied, tag=tst.ActivationDType, value=20)
    self.assertEqual(fdl.build(copied), (10, 20, 10, 40))

  def test_deepcopy_linked_value_object(self):

    def foo(a, b):
      return a, b

    shared_value = fdl.Config(foo, 1, 2)
    cfg = fdl.Config(
        foo,
        a=fdl.TaggedValue(tags={tst.LinearParamDType}, default=shared_value),
        b=fdl.TaggedValue(tags={tst.LinearParamDType}, default=shared_value),
    )
    copied = copy.deepcopy(cfg)
    self.assertIsNot(copied.a, cfg.a)
    self.assertIsNot(copied.a.value, cfg.a.value)

    self.assertIs(get_single_tag(copied.a), get_single_tag(cfg.a))
    self.assertIs(get_single_tag(copied.b), get_single_tag(cfg.b))
    self.assertIs(cfg.a.value, cfg.b.value)
    self.assertIs(copied.a.value, copied.b.value)

  def test_shallow_copy_tagged_values(self):
    cfg = tst.ParameterDType.new()
    copied = copy.copy(cfg)

    self.assertIs(get_single_tag(copied), tst.ParameterDType)
    self.assertIs(copied.value, tagging.NO_VALUE)

    with self.assertRaises(tagging.TaggedValueNotFilledError):
      fdl.build(copied)
    tagging.set_tagged(cfg, tag=tst.ParameterDType, value=1)
    self.assertEqual(fdl.build(cfg), 1)

    with self.assertRaises(tagging.TaggedValueNotFilledError):
      fdl.build(copied)
    tagging.set_tagged(copied, tag=tst.ParameterDType, value=2)
    self.assertEqual(fdl.build(cfg), 1)
    self.assertEqual(fdl.build(copied), 2)

  def test_multiple_keys(self):
    cfg = fdl.TaggedValue(
        tags={tst.ParameterDType, tst.MyModel.EncoderDType}, default=4)
    self.assertEqual(fdl.build(cfg), 4)

    # Set the value using the first key.
    tagging.set_tagged(cfg, tag=tst.ParameterDType, value=5)
    self.assertEqual(fdl.build(cfg), 5)

    # Now set it using the second key.
    tagging.set_tagged(cfg, tag=tst.MyModel.EncoderDType, value=6)
    self.assertEqual(fdl.build(cfg), 6)

  def test_materialize_tags_keeps_tag_in_ignore_list(self):
    foo_cfg = fdl.Config(Foo)
    foo_cfg.bar = RedTag.new(5)
    foo_cfg.qux = "abc"

    foo_cfg = tagging.materialize_tags(
        foo_cfg, tags=set(tagging.list_tags(foo_cfg) - {RedTag}))
    self.assertIsInstance(foo_cfg.bar, tagging.TaggedValueCls)
    self.assertEqual(get_single_tag(foo_cfg.bar), RedTag)

  def test_materialize_tags_removes_tag_when_default_provided(self):
    foo_cfg = fdl.Config(Foo)
    foo_cfg.bar = RedTag.new(5)
    foo_cfg.qux = "abc"

    foo_cfg = tagging.materialize_tags(foo_cfg)
    self.assertEqual(foo_cfg.bar, 5)
    self.assertIsInstance(foo_cfg.bar, int)

  def test_materialize_tags_removes_tag_when_value_supplied(self):
    foo_cfg = fdl.Config(Foo)
    foo_cfg.bar = RedTag.new()
    foo_cfg.qux = "abc"

    selectors.select(foo_cfg, tag=RedTag).replace(value=5)
    foo_cfg = tagging.materialize_tags(foo_cfg)
    self.assertEqual(foo_cfg.bar, 5)
    self.assertIsInstance(foo_cfg.bar, int)

  def test_materialize_tags_keeps_tag_when_no_default_provided(self):
    foo_cfg = fdl.Config(Foo)
    foo_cfg.bar = RedTag.new()
    foo_cfg.qux = "abc"

    foo_cfg = tagging.materialize_tags(foo_cfg)
    self.assertIsInstance(foo_cfg.bar, tagging.TaggedValueCls)
    self.assertEqual(get_single_tag(foo_cfg.bar), RedTag)

  def test_materialize_tags_materializes_tag_in_tags_list(self):
    foo_cfg = fdl.Config(Foo)
    foo_cfg.bar = RedTag.new(5)
    foo_cfg.qux = "abc"

    foo_cfg = tagging.materialize_tags(foo_cfg, tags={RedTag})
    self.assertEqual(foo_cfg.bar, 5)
    self.assertIsInstance(foo_cfg.bar, int)

  def test_materialize_tags_ignores_tag_not_in_tags_list(self):
    foo_cfg = fdl.Config(Foo)
    foo_cfg.bar = RedTag.new(5)
    foo_cfg.qux = "abc"

    foo_cfg = tagging.materialize_tags(foo_cfg, tags={BlueTag})
    self.assertIsInstance(foo_cfg.bar, tagging.TaggedValueCls)
    self.assertEqual(get_single_tag(foo_cfg.bar), RedTag)


# TODO: Test set_tagged that leverages tag inheritance.


class TestWithSelectorMock(TaggingTest):

  def setUp(self):
    super().setUp()

    def new_set_impl(cfg, tag, value):
      selectors.select(cfg, tag=tag).replace(value=value)

    self.mock_set = mock.patch.object(tagging, "set_tagged", new_set_impl)
    self.mock_set.start()

  def tearDown(self):
    self.mock_set.stop()
    super().tearDown()


if __name__ == "__main__":
  absltest.main()
