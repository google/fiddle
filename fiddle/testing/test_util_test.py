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

"""Tests for fiddle.testing.test_util."""

import copy
import dataclasses
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized

import fiddle as fdl
from fiddle import daglish
from fiddle import tagging
from fiddle import testing
from fiddle.experimental import diff
from fiddle.testing import test_util


def test_func(**kwargs):
  return kwargs


class SampleTag(tagging.Tag):
  """Tag for testing."""


@dataclasses.dataclass
class SampleClass:
  a: Any = 5
  b: Any = 6


def make_test_value():
  shared_list = [1, 2]
  shared_object = SampleClass(5)
  shared_config = fdl.Config(test_func, x=22)
  nested_shared_list = [3, shared_list]
  return fdl.Config(
      test_func,
      a=shared_list,
      b=shared_object,
      c=shared_config,
      d=[shared_list, shared_object, shared_config],
      e=nested_shared_list,
      f=SampleTag.new([5]))


class ParsePathTest(parameterized.TestCase):

  @parameterized.parameters([
      # Empty path:
      ('', ()),
      # Length-1 paths:
      ('.foo', (daglish.Attr('foo'),)),
      ('[0]', (daglish.Index(0),)),
      ('["foo"]', (daglish.Key('foo'),)),
      ("['a b c']", (daglish.Key('a b c'),)),
      ('.__fn_or_cls__', (daglish.BuildableFnOrCls(),)),
      # Length 2 paths:
      ('.foo.bar', (daglish.Attr('foo'), daglish.Attr('bar'))),
      ('.foo[22]', (daglish.Attr('foo'), daglish.Index(22))),
      ('[5]["x"]', (daglish.Index(5), daglish.Key('x'))),
      # Long path:
      ('.foo[8].bar.baz["x"].__fn_or_cls__',
       (daglish.Attr('foo'), daglish.Index(8), daglish.Attr('bar'),
        daglish.Attr('baz'), daglish.Key('x'), daglish.BuildableFnOrCls())),
  ])
  def test_parse_path(self, path_str, path):
    self.assertEqual(testing.parse_path(path_str), path)

  @parameterized.parameters([
      '.@#$',
      'foo.bar',
      '["]',
      '[foo]',
  ])
  def test_parse_path_error(self, path_str):
    with self.assertRaisesRegex(ValueError, 'Unable to parse path'):
      testing.parse_path(path_str)


class ParseReferenceTest(parameterized.TestCase):

  @parameterized.parameters([
      ('old', '', diff.Reference('old', ())),
      ('my_root', '.foo', diff.Reference('my_root', (daglish.Attr('foo'),))),
      ('$', '.foo.bar',
       diff.Reference('$', (daglish.Attr('foo'), daglish.Attr('bar')))),
      # Long path:
      ('root', '.foo[8].bar.baz["x"].__fn_or_cls__',
       diff.Reference(
           'root',
           (daglish.Attr('foo'), daglish.Index(8), daglish.Attr('bar'),
            daglish.Attr('baz'), daglish.Key('x'), daglish.BuildableFnOrCls()))
      ),
  ])
  def test_parse_reference(self, root, path_str, path):
    self.assertEqual(testing.parse_reference(root, path_str), path)


class DescribeDagDiffsTest(parameterized.TestCase):

  maxDiff = 5000

  shared_list = [1]

  def test_no_diffs(self):
    x = make_test_value()
    y = make_test_value()
    self.assertEqual(test_util.describe_dag_diffs(x, y), [])

  def test_type_diff(self):
    self.assertEqual(
        test_util.describe_dag_diffs([0], ['0']),
        ["* type(x[0]) != type(y[0]): <class 'int'> vs <class 'str'>"])

  def test_sharing_diff(self):
    shared_list = [0]
    a = [shared_list, shared_list]
    b = [shared_list, [0]]
    self.assertEqual(
        test_util.describe_dag_diffs(a, b),
        ['* Sharing diff: x[1] is x[0] but y[1] is not y[0]'])
    self.assertEqual(
        test_util.describe_dag_diffs(b, a),
        ['* Sharing diff: y[1] is y[0] but x[1] is not x[0]'])

  def test_leaf_diff(self):
    x = fdl.Config(test_func, a=5)
    y = fdl.Config(test_func, a=6)
    self.assertEqual(test_util.describe_dag_diffs(x, y), ['* x.a=5 but y.a=6'])

  def test_big_leaf_diff(self):
    x = fdl.Config(test_func, a='x' * 40)
    y = fdl.Config(test_func, a='y' * 40)
    self.assertEqual(
        test_util.describe_dag_diffs(x, y), [
            "* x.a='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' but\n" +
            "  y.a='yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy'"
        ])

  def test_set_diff(self):
    # Sets are considered leaves -- not traversed by daglish.
    x = fdl.Config(test_func, a=set([1, 2]))
    y = fdl.Config(test_func, a=set([2, 3]))
    self.assertEqual(
        test_util.describe_dag_diffs(x, y), ['* x.a={1, 2} but y.a={2, 3}'])

  def test_list_len_diff(self):
    self.assertEqual(
        test_util.describe_dag_diffs([1], [2, 3]),
        ['* y[1] has a value but x[1] does not.'])
    self.assertEqual(
        test_util.describe_dag_diffs([[1]], [[2, 3]]),
        ['* y[0][1] has a value but x[0][1] does not.'])

  def test_tuple_len_diff(self):
    self.assertEqual(
        test_util.describe_dag_diffs((1,), (2, 3)),
        ['* y[1] has a value but x[1] does not.'])
    self.assertEqual(
        test_util.describe_dag_diffs([(1,)], [(2, 3)]),
        ['* y[0][1] has a value but x[0][1] does not.'])

  def test_dict_keys_diff(self):
    self.assertEqual(
        test_util.describe_dag_diffs(dict(a=1), dict(b=1)), [
            "* x['a'] has a value but y['a'] does not.",
            "* y['b'] has a value but x['b'] does not."
        ])
    self.assertEqual(
        test_util.describe_dag_diffs([dict(a=2)], [dict(b=2)]), [
            "* x[0]['a'] has a value but y[0]['a'] does not.",
            "* y[0]['b'] has a value but x[0]['b'] does not."
        ])

  def test_config_arg_names_diff(self):
    x = fdl.Config(test_func, a=3)
    y = fdl.Config(test_func, b=3)
    self.assertEqual(
        test_util.describe_dag_diffs(x, y), [
            '* x.a has a value but y.a does not.',
            '* y.b has a value but x.b does not.'
        ])

  def test_config_callable_diff(self):
    # Note: Config.__fn_or_cls__ is metadata (not a traversed child).
    x = fdl.Config(SampleClass)
    y = fdl.Config(test_func)
    self.assertEqual(
        test_util.describe_dag_diffs(x, y), [
            '* x=<Config[SampleClass()]> but y=<Config[test_func()]>',
        ])

  def test_replace_value_with_unrelated_value(self):
    x = make_test_value()
    y = make_test_value()
    y.a = 12
    self.assertEqual(
        test_util.describe_dag_diffs(x, y), [
            '* Sharing diff: x.d[0] is x.a but y.d[0] is not y.a',
            '* Sharing diff: x.e[1] is x.a but y.e[1] is not y.a',
            '* Sharing diff: y.e[1] is y.d[0] but x.e[1] is not x.d[0]',
            "* type(x.a) != type(y.a): <class 'list'> vs <class 'int'>",
        ])

  def test_replace_value_with_copy(self):
    x = make_test_value()
    y = make_test_value()
    y.a = copy.copy(y.a)
    self.assertEqual(
        test_util.describe_dag_diffs(x, y), [
            '* Sharing diff: x.d[0] is x.a but y.d[0] is not y.a',
            '* Sharing diff: x.e[1] is x.a but y.e[1] is not y.a',
            '* Sharing diff: y.e[1] is y.d[0] but x.e[1] is not x.d[0]',
        ])

  def test_multiple_difference(self):
    x = make_test_value()
    y = make_test_value()
    y.a = copy.copy(y.a)
    y.c = copy.copy(y.c)
    y.d.append(5)
    y.e = [y.e]
    self.assertEqual(
        test_util.describe_dag_diffs(x, y), [
            '* x.e[1] has a value but y.e[1] does not.',
            '* y.d[3] has a value but x.d[3] does not.',
        ])

  def test_config_default_is_different(self):
    x = fdl.Config(SampleClass)
    y = fdl.Config(SampleClass, a=5)
    self.assertEqual(x, y)
    self.assertEqual(
        test_util.describe_dag_diffs(x, y), [
            '* y.a has a value but x.a does not.',
        ])

  def test_sharing_not_detected_in_non_traversable_object(self):
    # There is no NodeTraverser registered for SampleClass, so we don't
    # recurse inside it when looking for shared objects.
    shared_list = [1]
    x = SampleClass(a=shared_list, b=shared_list)
    y = SampleClass(a=shared_list, b=copy.copy(shared_list))
    self.assertEmpty(test_util.describe_dag_diffs(x, y))


class TestCaseTest(testing.TestCase):

  def test_dag_equal(self):
    x = make_test_value()
    y = make_test_value()
    self.assertDagEqual(x, y)

  def test_dag_not_equal(self):
    x = make_test_value()
    y = make_test_value()
    y.a = 5
    # pylint: disable=g-error-prone-assert-raises
    with self.assertRaises(self.failureException):
      self.assertDagEqual(x, y)

  def test_dag_sharing_diff(self):
    x = make_test_value()
    y = make_test_value()
    y.a = copy.copy(y.a)
    # pylint: disable=g-error-prone-assert-raises
    with self.assertRaisesRegex(self.failureException, 'Sharing diff'):
      self.assertDagEqual(x, y)


if __name__ == '__main__':
  absltest.main()
