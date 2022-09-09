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

"""Tests for print."""

import dataclasses
import re
import textwrap
from typing import Union

from absl.testing import absltest
import fiddle as fdl
from fiddle import daglish
from fiddle import printing
from fiddle import tagging


class SampleTag(tagging.Tag):
  """Sample tag for testing."""


class SampleTag2(tagging.Tag):
  """Second tag, for fun & profit!"""


def fn_x_y(x, y):  # pylint: disable=unused-argument
  pass


def fn_with_kwargs(x, y, **kwargs):  # pylint: disable=unused-argument
  pass


class SampleClass:

  def __init__(self, a, b):  # pylint: disable=unused-argument
    pass


def fn_with_type_annotations(x: int, y: str, z: float):  # pylint: disable=unused-argument
  pass


@dataclasses.dataclass
class DataclassHelper:
  x: int
  y: str


def advanced_annotations_helper(x: SampleClass, **kwargs: DataclassHelper):  # pylint: disable=unused-argument
  pass


def annotated_kwargs_helper(**kwargs: int):  # pylint: disable=unused-argument
  pass


# Sometimes the module of local symbols shows up differently, depending on how
# this test is imported/run.
_local_module_regex = r'(__main__|fiddle\.printing_test)'


class PathStrTest(absltest.TestCase):

  def test_empty(self):
    self.assertEqual(printing._path_str(()), '')

  def test_nested_attr(self):
    path = (daglish.Attr('foo'), daglish.Attr('bar'))
    self.assertEqual(printing._path_str(path), 'foo.bar')

  def test_list(self):
    path = (daglish.Index(0),)
    self.assertEqual(printing._path_str(path), '[0]')


class GetTypeAnnotationTest(absltest.TestCase):

  def test_get_type_annotation_root(self):
    config = fdl.Config(fn_with_type_annotations)
    self.assertIsNone(printing._get_annotation(config, ()))


class AsStrFlattenedTests(absltest.TestCase):

  def check_result(self, actual, expected):
    expected = re.escape(textwrap.dedent(expected))
    self.assertRegex(actual, expected)

  def test_simple_printing(self):
    cfg = fdl.Config(fn_x_y, 1, 'abc')
    output = printing.as_str_flattened(cfg)

    self.check_result(output, """\
        x = 1
        y = 'abc'""")

  def test_unset_argument(self):
    cfg = fdl.Config(fn_x_y, 3.14)
    output = printing.as_str_flattened(cfg)

    expected = textwrap.dedent("""\
        x = 3.14
        y = <[unset]>""")
    self.assertEqual(output, expected)

  def test_nested(self):
    cfg = fdl.Config(fn_x_y, 'x', fdl.Config(fn_x_y, 'nest_x', 123))
    output = printing.as_str_flattened(cfg)

    expected = textwrap.dedent("""\
        x = 'x'
        y.x = 'nest_x'
        y.y = 123""")
    self.assertEqual(output, expected)

  def test_class(self):
    cfg = fdl.Config(SampleClass, 'a_param', b=123)
    output = printing.as_str_flattened(cfg)

    expected = textwrap.dedent("""\
        a = 'a_param'
        b = 123""")
    self.assertEqual(output, expected)

  def test_kwargs(self):
    cfg = fdl.Config(fn_with_kwargs, 1, abc='extra kwarg value')
    output = printing.as_str_flattened(cfg)

    expected = textwrap.dedent("""\
        x = 1
        y = <[unset]>
        abc = 'extra kwarg value'""")
    self.assertEqual(output, expected)

  def test_nested_kwargs(self):
    cfg = fdl.Config(
        fn_with_kwargs,
        extra=fdl.Config(fn_with_kwargs, 1, nested_extra='whee'))
    output = printing.as_str_flattened(cfg)

    expected = textwrap.dedent("""\
        x = <[unset]>
        y = <[unset]>
        extra.x = 1
        extra.y = <[unset]>
        extra.nested_extra = 'whee'""")
    self.assertEqual(output, expected)

  def test_nested_collections(self):
    cfg = fdl.Config(fn_x_y,
                     [fdl.Config(fn_x_y, 1, '1'),
                      fdl.Config(SampleClass, 2)])
    output = printing.as_str_flattened(cfg)

    expected = textwrap.dedent("""\
        x[0].x = 1
        x[0].y = '1'
        x[1].a = 2
        x[1].b = <[unset]>
        y = <[unset]>""")
    self.assertEqual(output, expected)

  def test_multiple_nested_collections(self):
    cfg = fdl.Config(fn_x_y, {
        'a': fdl.Config(fn_with_kwargs, abc=[1, 2, 3]),
        'b': [3, 2, 1]
    }, [fdl.Config(fn_x_y, [fdl.Config(fn_x_y, 1, 2)])])
    output = printing.as_str_flattened(cfg)

    expected = textwrap.dedent("""\
        x['a'].x = <[unset]>
        x['a'].y = <[unset]>
        x['a'].abc = [1, 2, 3]
        x['b'] = [3, 2, 1]
        y[0].x[0].x = 1
        y[0].x[0].y = 2
        y[0].y = <[unset]>""")
    self.assertEqual(output, expected)

  def test_default_values(self):

    def test_fn(w, x, y=3, z='abc'):  # pylint: disable=unused-argument
      pass

    cfg = fdl.Config(test_fn, 1)
    output = printing.as_str_flattened(cfg)

    expected = textwrap.dedent("""\
        w = 1
        x = <[unset]>
        y = <[unset; default: 3]>
        z = <[unset; default: 'abc']>""")
    self.assertEqual(output, expected)

  def test_tagged_values(self):
    cfg = fdl.Config(fn_x_y, x=SampleTag.new(), y=SampleTag.new(default='abc'))
    output = printing.as_str_flattened(cfg)

    self.check_result(
        output, f"""\
        x = <[unset]> {SampleTag}
        y = 'abc' {SampleTag}""")

    tagging.set_tagged(cfg, tag=SampleTag, value='cba')
    output = printing.as_str_flattened(cfg)

    self.check_result(
        output, f"""\
        x = 'cba' {SampleTag}
        y = 'cba' {SampleTag}""")

  def test_tagged_values_multiple_tags(self):
    cfg = fdl.Config(
        fn_x_y,
        x=tagging.TaggedValue(tags=(SampleTag, SampleTag2)),
        y=tagging.TaggedValue(tags=(SampleTag, SampleTag2), default='abc'))
    output = printing.as_str_flattened(cfg)

    self.check_result(
        output, f"""\
        x = <[unset]> {SampleTag} {SampleTag2}
        y = 'abc' {SampleTag} {SampleTag2}""")

    tagging.set_tagged(cfg, tag=SampleTag, value='cba')
    output = printing.as_str_flattened(cfg)

    self.check_result(
        output, f"""\
        x = 'cba' {SampleTag} {SampleTag2}
        y = 'cba' {SampleTag} {SampleTag2}""")

  def test_tagged_config(self):
    cfg = fdl.Config(
        fn_x_y,
        x=tagging.TaggedValue(
            tags=(SampleTag,), default=fdl.Config(SampleClass)))
    output = printing.as_str_flattened(cfg)
    self.check_result(
        output, f"""\
        x = <Config[SampleClass()]> {SampleTag}
        y = <[unset]>""")

  def test_argument_tags(self):
    cfg = fdl.Config(fn_x_y, y='abc')
    fdl.add_tag(cfg, 'x', SampleTag)
    fdl.add_tag(cfg, 'y', SampleTag)
    output = printing.as_str_flattened(cfg)

    self.check_result(
        output, f"""\
        x = <[unset]> {SampleTag}
        y = 'abc' {SampleTag}""")

    tagging.set_tagged(cfg, tag=SampleTag, value='cba')
    output = printing.as_str_flattened(cfg)

    self.check_result(
        output, f"""\
        x = 'cba' {SampleTag}
        y = 'cba' {SampleTag}""")

  def test_argument_tags_multiple_tags(self):
    cfg = fdl.Config(fn_x_y, y='abc')
    fdl.set_tags(cfg, 'x', (SampleTag, SampleTag2))
    fdl.set_tags(cfg, 'y', (SampleTag, SampleTag2))
    output = printing.as_str_flattened(cfg)

    self.check_result(
        output, f"""\
        x = <[unset]> {SampleTag} {SampleTag2}
        y = 'abc' {SampleTag} {SampleTag2}""")

    tagging.set_tagged(cfg, tag=SampleTag, value='cba')
    output = printing.as_str_flattened(cfg)

    self.check_result(
        output, f"""\
        x = 'cba' {SampleTag} {SampleTag2}
        y = 'cba' {SampleTag} {SampleTag2}""")

  def test_argument_tags_tagged_config(self):
    cfg = fdl.Config(fn_x_y, x=fdl.Config(SampleClass))
    fdl.add_tag(cfg, 'x', SampleTag)
    output = printing.as_str_flattened(cfg)
    self.check_result(
        output, f"""\
        x = <Config[SampleClass()]> {SampleTag}
        y = <[unset]>""")

  def test_partial(self):
    partial = fdl.Partial(fn_x_y)
    partial.x = 'abc'
    output = printing.as_str_flattened(partial)

    expected = textwrap.dedent("""\
        x = 'abc'
        y = <[unset]>""")
    self.assertEqual(output, expected)

  def test_builtin_types_annotations(self):
    cfg = fdl.Config(fn_with_type_annotations, 1)
    cfg.y = 'abc'
    output = printing.as_str_flattened(cfg)

    expected = textwrap.dedent("""\
        x: int = 1
        y: str = 'abc'
        z: float = <[unset]>""")
    self.assertEqual(output, expected)

  def test_advanced_type_annotations(self):
    cfg = fdl.Config(advanced_annotations_helper)
    cfg.abc = fdl.Config(DataclassHelper)
    output = printing.as_str_flattened(cfg)

    expected = textwrap.dedent("""\
        x: SampleClass = <[unset]>
        abc.x: int = <[unset]>
        abc.y: str = <[unset]>""")
    self.assertEqual(output, expected)

  def test_annotated_kwargs(self):
    cfg = fdl.Config(annotated_kwargs_helper, x=1, y='oops')
    output = printing.as_str_flattened(cfg)

    expected = textwrap.dedent("""\
        x: int = 1
        y: int = 'oops'""")
    self.assertEqual(output, expected)

  def test_disabling_type_annotations(self):
    cfg = fdl.Config(fn_with_type_annotations, 1)
    cfg.y = 'abc'
    output = printing.as_str_flattened(cfg, include_types=False)

    expected = textwrap.dedent("""\
        x = 1
        y = 'abc'
        z = <[unset]>""")
    self.assertEqual(output, expected)

  def test_can_print_union_type(self):

    def to_integer(x: Union[int, str]):
      return int(x)

    cfg = fdl.Config(to_integer, 1)
    output = printing.as_str_flattened(cfg, include_types=True)

    expected = textwrap.dedent("""\
        x: typing.Union[int, str] = 1""")
    self.assertEqual(output, expected)

  def test_materialized_default_values(self):

    def test_fn(w, x, y=3, z='abc'):
      del w, x, y, z  # Unused.

    cfg = fdl.Config(test_fn, 1)
    fdl.materialize_defaults(cfg)
    output = printing.as_str_flattened(cfg)
    expected = textwrap.dedent("""\
        w = 1
        x = <[unset]>
        y = 3
        z = 'abc'""")
    self.assertEqual(output, expected)


class HistoryPerLeafParamTests(absltest.TestCase):

  def test_simple_history(self):
    cfg = fdl.Config(fn_x_y, 1, 'abc')
    cfg.x = 2
    output = printing.history_per_leaf_parameter(cfg)
    expected = textwrap.dedent(r"""
        x = 2 @ .*/printing_test.py:\d+:test_simple_history
          - previously: 1 @ .*/printing_test.py:\d+:test_simple_history
        y = 'abc' @ .*/printing_test.py:\d+:test_simple_history""".strip('\n'))
    self.assertRegex(output, expected)

  def test_nested_in_collections(self):
    cfg = fdl.Config(fn_x_y,
                     [fdl.Config(fn_x_y, 1, '1'),
                      fdl.Config(SampleClass, 2)])
    cfg.x[0].x = 3
    cfg.x[1].a = 2  # Reset to same value.
    cfg.x[0].y = 'abc'
    cfg.x[0].x = 4
    output = printing.history_per_leaf_parameter(cfg)
    self.assertTrue(printing._has_nested_builder(cfg.x))
    expected = textwrap.dedent(rf"""
        __fn_or_cls__ = .*fn_x_y .+/printing_test.py:\d+:test_nested_in_collections
        x\[0\].__fn_or_cls__ = .*fn_x_y .+/printing_test.py:\d+:test_nested_in_collections
        x\[0\].x = 4 @ .*/printing_test.py:\d+:test_nested_in_collections
          - previously: 3 @ .*/printing_test.py:\d+:test_nested_in_collections
          - previously: 1 @ .*/printing_test.py:\d+:test_nested_in_collections
        x\[0\].y = 'abc' @ .*/printing_test.py:\d+:test_nested_in_collections
          - previously: '1' @ .*/printing_test.py:\d+:test_nested_in_collections
        x\[1\].__fn_or_cls__ = .*{_local_module_regex}.SampleClass.*/printing_test.py:\d+:test_nested_in_collections
        x\[1\].a = 2 @ .*/printing_test.py:\d+:test_nested_in_collections
          - previously: 2 @ .*/printing_test.py:\d+:test_nested_in_collections
        x\[1\].b = <\[unset\]>
        y = <\[unset\]>""".strip('\n'))
    self.assertRegex(output, expected)

  def test_update_callable_history(self):
    cfg = fdl.Config(fn_x_y, x=1, y=2)
    fdl.update_callable(cfg, fn_with_kwargs)
    cfg.abc = '123'
    output = printing.history_per_leaf_parameter(cfg)
    expected = textwrap.dedent(r"""
        __fn_or_cls__ = .*fn_with_kwargs .+/printing_test.py:\d+:test_update_callable_history
          - previously: .*fn_x_y .+/printing_test.py:\d+:test_update_callable_history
        x = 1 @ .+/printing_test.py:\d+:test_update_callable_history
        y = 2 @ .+/printing_test.py:\d+:test_update_callable_history
        abc = '123' @ .+/printing_test.py:\d+:test_update_callable_history
        kwargs = <\[unset\]>""".strip('\n'))
    self.assertRegex(output, expected)

  def test_materialize_defaults_history(self):

    def test_defaulting_helper(w, x, y=1, z=2):
      del w, x, y, z  # Unused.

    cfg = fdl.Config(test_defaulting_helper, w=0)
    cfg.y = 5
    cfg.y = 6
    fdl.materialize_defaults(cfg)
    output = printing.history_per_leaf_parameter(cfg)
    expected = textwrap.dedent(r"""
        __fn_or_cls__ = .*test_defaulting_helper .+/printing_test.py:\d+:test_materialize_defaults_history
        w = 0 @ .*/printing_test.py:\d+:test_materialize_defaults_history
        y = 6 @ .*/printing_test.py:\d+:test_materialize_defaults_history
          - previously: 5 .*
        z = 2 @ .*/printing_test.py:\d+:test_materialize_defaults_history
        x = <\[unset\]>""".strip('\n'))

    self.assertRegex(output, expected)

  def test_collection_of_two_buildables_history(self):
    config_a = fdl.Config(fn_x_y, x=1)
    config_a.x = 2
    config_b = fdl.Config(fn_x_y, y=3)
    config_b.x = 10
    config_b.y = 4
    output = printing.history_per_leaf_parameter([config_a, config_b])
    name = r'.*printing_test.py:\d+:test_collection_of_two_buildables_history'
    expected = textwrap.dedent(rf"""
        \[0\].__fn_or_cls__ = <function fn_x_y at .*> @ {name}
        \[0\].x = 2 @ {name}
          - previously: 1 @ {name}
        \[0\].y = <\[unset\]>
        \[1\].__fn_or_cls__ = <function fn_x_y at .*> @ {name}
        \[1\].y = 4 @ {name}
          - previously: 3 @ {name}
        \[1\].x = 10 @ {name}""".strip('\n'))
    self.assertRegex(output, expected)


if __name__ == '__main__':
  absltest.main()
