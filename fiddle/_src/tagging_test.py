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
import pickle
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import cloudpickle
import fiddle as fdl
from fiddle import selectors
from fiddle._src import daglish
from fiddle._src import tagging
from fiddle._src import tagging_test_module as tst
from typing_extensions import Annotated


@dataclasses.dataclass
class Foo:
  bar: int
  qux: str


class RedTag(fdl.Tag):
  """A custom tag used for testing."""


class BlueTag(fdl.Tag):
  """A custom tag used for testing."""


class Tag1(fdl.Tag):
  """One tag."""


class Tag2(fdl.Tag):
  """Another tag."""


class SampleClass:
  arg1: Any
  arg2: Any
  kwarg1: Any
  kwarg2: Any

  def __init__(self, arg1, arg2, kwarg1=None, kwarg2=None):  # pylint: disable=unused-argument
    self.__dict__.update(locals())

  def a_method(self):
    return 4  # A random number (https://xkcd.com/221/)

  @classmethod
  def a_classmethod(cls):
    return cls(1, 2)


def variadic_positional_fn(a, b, /, c, *args):
  return a, b, c, *args


def positional_only_fn(a, b, /, c):
  return a, b, c


@dataclasses.dataclass
class DataclassAnnotated:
  one: Annotated[int, Tag1]
  two: Annotated[str, Tag1, Tag2]

  three: "Annotated[float, Tag2]"


def return_kwargs(**kwargs):
  return kwargs


def sample_function(x, y):
  return (x, y)


def get_single_tag(tagged_value: tagging.TaggedValueCls) -> tagging.TagType:
  """Gets a single tag from a TaggedValue, errors if there are multiple."""
  assert isinstance(tagged_value, tagging.TaggedValueCls)
  assert len(tagged_value.tags) == 1
  return next(iter(tagged_value.tags))


class TaggingTest(parameterized.TestCase):

  def test_tag_str(self):
    self.assertEqual(
        str(tst.ParameterDType),
        "#fiddle._src.tagging_test_module.ParameterDType",
    )
    self.assertEqual(
        str(tst.LinearParamDType),
        "#fiddle._src.tagging_test_module.LinearParamDType",
    )
    self.assertEqual(
        str(tst.MyModel.EncoderDType),
        "#fiddle._src.tagging_test_module.MyModel.EncoderDType",
    )
    self.assertEqual(
        str(tst.MyModel.DecoderDType),
        "#fiddle._src.tagging_test_module.MyModel.DecoderDType",
    )

  def test_tag_instantiation(self):
    with self.assertRaisesRegex(
        TypeError, ".*cannot instantiate Fiddle tags.*ParameterDType.*"):
      _ = tst.ParameterDType()
    with self.assertRaisesRegex(
        TypeError,
        (
            "trying to instantiate"
            " fiddle._src.tagging_test_module.MyModel.DecoderDType"
        ),
    ):
      _ = tst.MyModel.DecoderDType()

  def test_missing_tag_description(self):
    with self.assertRaisesRegex(TypeError,
                                "You must provide a tag description"):

      class Bar(tagging.Tag):  # pylint: disable=unused-variable
        pass

  def test_tag_pickle(self):
    p = pickle.dumps(RedTag)
    rt = pickle.loads(p)
    self.assertIs(rt, RedTag)

  def test_tag_cloudpickle_other_module(self):
    p = cloudpickle.dumps(tst.ParameterDType)
    tag = cloudpickle.loads(p)
    self.assertIs(tag, tst.ParameterDType)

  def test_tag_cloudpickle_main_module(self):
    p = cloudpickle.dumps(RedTag)
    tag = cloudpickle.loads(p)
    self.assertIs(tag, RedTag)

  def test_inline_tag_definition_fails(self):

    with self.assertRaisesRegex(
        TypeError, "You cannot define a tag within a function or lambda"):

      class Bar(tagging.Tag):  # pylint: disable=unused-variable
        """Bar tag."""

  def test_tagged_value_identity_fn_exception(self):
    with self.assertRaisesRegex(
        tagging.TaggedValueNotFilledError,
        "Expected.*TaggedValue.*replaced.*one was not set"):
      tagging.tagged_value_fn(fdl.NO_VALUE)

  def test_tagged_value_identity_fn_none_value(self):
    # Test that when the value is None, that's OK.
    self.assertIsNone(tagging.tagged_value_fn(value=None))

  def test_tagged_value_error_message(self):
    cfg = fdl.Config(Foo)
    cfg.bar = tagging.TaggedValue(tags={tst.ParameterDType})  # pytype: disable=annotation-type-mismatch  # use-fiddle-overlay
    with self.assertRaisesRegex(
        TypeError,
        (
            r"Tags for unset arguments:\n - bar: "
            r"#fiddle._src.tagging_test_module.ParameterDType"
        ),
    ):
      fdl.build(cfg)

  def test_taggedvalue_default_none(self):
    cfg = tagging.TaggedValue(tags={tst.ParameterDType}, default=None)
    self.assertIsNone(fdl.build(cfg))

  def test_one_taggedvalue_unset_in_config(self):
    cfg = fdl.Config(
        sample_function,
        x=tst.ParameterDType.new(default=None),
        y=tst.ParameterDType.new(),
    )
    with self.assertRaisesRegex(
        TypeError, r" - y: #fiddle._src.tagging_test_module.ParameterDType"
    ):
      fdl.build(cfg)

  def test_set_tagged(self):
    cfg = fdl.Config(
        return_kwargs,
        foo=tst.ParameterDType.new(default=None),
        bar=tst.LinearParamDType.new(),
    )
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

    with self.assertRaisesRegex(
        TypeError,
        (
            r".*Tags for unset arguments:\n - a:"
            r" #fiddle._src.tagging_test_module.ParameterDType.*"
        ),
    ):
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

    self.assertLen(fdl.get_tags(cfg, "a"), 1)
    self.assertLen(fdl.get_tags(copied, "a"), 1)
    self.assertIs(
        list(fdl.get_tags(cfg, "a"))[0], list(fdl.get_tags(copied, "a"))[0]
    )

    self.assertLen(fdl.get_tags(cfg, "b"), 1)
    self.assertLen(fdl.get_tags(copied, "b"), 1)
    self.assertIs(
        list(fdl.get_tags(cfg, "b"))[0], list(fdl.get_tags(copied, "b"))[0]
    )

    self.assertIs(cfg.a, cfg.b)
    self.assertIs(copied.a, copied.b)

  def test_shallow_copy_tagged_values(self):
    cfg = tst.ParameterDType.new()
    copied = copy.copy(cfg)

    self.assertIs(get_single_tag(copied), tst.ParameterDType)
    self.assertFalse(hasattr(copied, "value"))

    with self.assertRaises(TypeError):
      fdl.build(copied)
    tagging.set_tagged(cfg, tag=tst.ParameterDType, value=1)
    self.assertEqual(fdl.build(cfg), 1)

    with self.assertRaises(TypeError):
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
    self.assertEqual(frozenset([RedTag]), fdl.get_tags(foo_cfg, "bar"))

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
    self.assertEqual(frozenset([RedTag]), fdl.get_tags(foo_cfg, "bar"))

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
    self.assertEqual(frozenset([RedTag]), fdl.get_tags(foo_cfg, "bar"))

  def test_tagging_ops(self):
    cfg = fdl.Config(SampleClass)
    fdl.add_tag(cfg, "arg1", Tag1)

    self.assertEqual(frozenset([Tag1]), fdl.get_tags(cfg, "arg1"))
    self.assertEqual(frozenset([]), fdl.get_tags(cfg, "arg2"))

    fdl.add_tag(cfg, "arg1", Tag2)
    self.assertEqual(frozenset([Tag1, Tag2]), fdl.get_tags(cfg, "arg1"))

    fdl.remove_tag(cfg, "arg1", Tag2)
    self.assertEqual(frozenset([Tag1]), fdl.get_tags(cfg, "arg1"))

    with self.assertRaisesRegex(ValueError, ".*not set.*"):
      fdl.remove_tag(cfg, "arg1", Tag2)

  @parameterized.parameters([(variadic_positional_fn,), (positional_only_fn,)])
  def test_tagging_positional_arguments(self, fn):
    cfg = fdl.Config(fn)
    fdl.add_tag(cfg, 2, Tag1)

    self.assertEqual(frozenset([Tag1]), fdl.get_tags(cfg, 2))
    self.assertEqual(frozenset([Tag1]), fdl.get_tags(cfg, "c"))
    self.assertEqual(frozenset([]), fdl.get_tags(cfg, 1))

    fdl.add_tag(cfg, 2, Tag2)
    self.assertEqual(frozenset([Tag1, Tag2]), fdl.get_tags(cfg, 2))

    fdl.set_tags(cfg, 0, {Tag1, Tag2})
    self.assertEqual(frozenset([Tag1, Tag2]), fdl.get_tags(cfg, 0))

    fdl.remove_tag(cfg, 2, Tag2)
    self.assertEqual(frozenset([Tag1]), fdl.get_tags(cfg, 2))

    with self.assertRaisesRegex(ValueError, ".*not set.*"):
      fdl.remove_tag(cfg, 2, Tag2)

    fdl.clear_tags(cfg, 2)
    self.assertEqual(frozenset([]), fdl.get_tags(cfg, 2))

  def test_tagging_with_out_of_range_index(self):
    cfg = fdl.Config(positional_only_fn, 0, 1, 2)

    for fn in (fdl.add_tag, fdl.set_tags, fdl.remove_tag):
      with self.assertRaisesRegex(IndexError, ".*is out of range"):
        fn(cfg, 3, Tag1)

    for fn in (fdl.get_tags, fdl.clear_tags):
      with self.assertRaisesRegex(IndexError, ".*is out of range"):
        fn(cfg, 3)

  def test_tagging_with_negative_index(self):
    cfg = fdl.Config(positional_only_fn, 0, 1, 2)

    for fn in (fdl.add_tag, fdl.set_tags, fdl.remove_tag):
      with self.assertRaisesRegex(IndexError, "Cannot use negative index"):
        fn(cfg, -1, Tag1)

    for fn in (fdl.get_tags, fdl.clear_tags):
      with self.assertRaisesRegex(IndexError, "Cannot use negative index"):
        fn(cfg, -1)

  def test_tag_annotations(self):
    cfg = fdl.Config(DataclassAnnotated)

    self.assertEqual(frozenset([Tag1]), fdl.get_tags(cfg, "one"))
    self.assertEqual(frozenset([Tag1, Tag2]), fdl.get_tags(cfg, "two"))
    self.assertEqual(frozenset([Tag2]), fdl.get_tags(cfg, "three"))

  def test_tags_flattening_and_unflattening(self):
    def flatten_unflatten(config):
      values, metadata = config.__flatten__()
      return type(config).__unflatten__(values, metadata)

    with self.subTest("tagged_field_without_value"):
      cfg = fdl.Config(DataclassAnnotated)
      self.assertEqual(frozenset([Tag1]), fdl.get_tags(cfg, "one"))
      cfg2 = flatten_unflatten(cfg)
      self.assertEqual(frozenset([Tag1]), fdl.get_tags(cfg2, "one"))

    with self.subTest("untagged_field_without_value"):
      cfg = fdl.Config(DataclassAnnotated)
      fdl.clear_tags(cfg, "one")
      self.assertEqual(frozenset([]), fdl.get_tags(cfg, "one"))
      cfg2 = flatten_unflatten(cfg)
      self.assertEqual(frozenset([]), fdl.get_tags(cfg2, "one"))

    with self.subTest("tagged_field_with_value"):
      cfg = fdl.Config(DataclassAnnotated)
      cfg.one = fdl.TaggedValue([Tag2], 3)  # pytype: disable=annotation-type-mismatch  # use-fiddle-overlay
      self.assertEqual(frozenset([Tag1, Tag2]), fdl.get_tags(cfg, "one"))
      cfg2 = flatten_unflatten(cfg)
      self.assertEqual(frozenset([Tag1, Tag2]), fdl.get_tags(cfg2, "one"))

    with self.subTest("untagged_field_with_value"):
      cfg = fdl.Config(DataclassAnnotated)
      cfg.one = 3
      fdl.clear_tags(cfg, "one")
      self.assertEqual(frozenset([]), fdl.get_tags(cfg, "one"))
      cfg2 = flatten_unflatten(cfg)
      self.assertEqual(frozenset([]), fdl.get_tags(cfg2, "one"))

  def test_setting_tagged_values_with_daglish_traversal(self):
    cfg = fdl.Config(DataclassAnnotated, one=3)

    def traverse(value, state):
      if isinstance(value, int):
        return fdl.TaggedValue([Tag2], value)
      else:
        return state.map_children(value)

    # We expect the TaggedValue class *not* to be unwrapped by the constructor
    # now, because __new__ is called directly.
    cfg2 = traverse(cfg, daglish.MemoizedTraversal.begin(traverse, cfg))
    self.assertEqual(frozenset([Tag1]), fdl.get_tags(cfg2, "one"))
    self.assertEqual(frozenset([Tag2]), fdl.get_tags(cfg2.one, "value"))

  def test_clear_tags(self):
    cfg = fdl.Config(SampleClass)
    fdl.add_tag(cfg, "arg1", Tag1)
    fdl.add_tag(cfg, "arg1", Tag2)
    fdl.clear_tags(cfg, "arg1")
    self.assertEqual(frozenset([]), fdl.get_tags(cfg, "arg1"))

  def test_set_tags(self):
    cfg = fdl.Config(SampleClass)
    fdl.add_tag(cfg, "arg1", Tag1)
    fdl.add_tag(cfg, "arg1", Tag2)
    fdl.set_tags(cfg, "arg1", {Tag2})
    self.assertEqual(frozenset([Tag2]), fdl.get_tags(cfg, "arg1"))

  def test_flatten_unflatten_tags(self):
    cfg = fdl.Config(SampleClass)
    fdl.add_tag(cfg, "arg1", Tag1)
    values, metadata = cfg.__flatten__()
    copied = fdl.Config.__unflatten__(values, metadata)
    fdl.add_tag(copied, "arg1", Tag2)
    self.assertEqual(frozenset([Tag1]), fdl.get_tags(cfg, "arg1"))
    self.assertEqual(frozenset([Tag1, Tag2]), fdl.get_tags(copied, "arg1"))

  def test_build_reraises_nice_error_with_tag_information(self):
    cfg = fdl.Config(SampleClass)
    fdl.add_tag(cfg, "arg2", Tag1)

    with self.assertRaisesRegex(
        TypeError,
        r".*Tags for unset arguments:\n - arg2: #{module}.Tag1.*".format(
            module=__name__
        ),
    ):
      fdl.build(cfg)

  def test_build_reraises_nice_error_multiple_tags(self):
    cfg = fdl.Config(SampleClass)
    fdl.add_tag(cfg, "arg1", Tag2)
    fdl.add_tag(cfg, "arg2", Tag1)
    fdl.add_tag(cfg, "arg2", Tag2)

    expected_match = (
        r".* - arg1: #{module}.Tag2\n - arg2: #{module}.Tag1 #{module}.Tag2.*"
        .format(module=__name__)
    )
    with self.assertRaisesRegex(TypeError, expected_match):
      fdl.build(cfg)

  def test_repr_class_tags(self):
    config = fdl.Config(
        SampleClass,
        1,
        kwarg1="kwarg1",
        kwarg2=fdl.Config(
            SampleClass,
            "nested value might be large so "
            + "put tag next to param, not after value.",
        ),
    )
    fdl.add_tag(config, "arg1", Tag1)
    fdl.add_tag(config, "arg2", Tag2)
    fdl.add_tag(config, "kwarg2", Tag1)
    fdl.add_tag(config, "kwarg2", Tag2)
    expected_repr = """<Config[SampleClass(
  arg1[#{module}.Tag1]=1,
  arg2[#{module}.Tag2],
  kwarg1='kwarg1',
  kwarg2[#{module}.Tag1, #{module}.Tag2]=<Config[SampleClass(
    arg1='nested value might be large so put tag next to param, not after value.')]>)]>"""
    self.assertEqual(repr(config), expected_repr.format(module=__name__))


# TODO(b/272077830): Test set_tagged that leverages tag inheritance.


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
