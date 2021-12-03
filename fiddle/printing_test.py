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

from absl.testing import absltest
import fiddle as fdl
from fiddle import placeholders
from fiddle import printing

test_key = placeholders.PlaceholderKey('test_key')


def test_helper(x, y):  # pylint: disable=unused-argument
  pass


def test_kwarg_helper(x, y, **kwargs):  # pylint: disable=unused-argument
  pass


class TestHelper:

  def __init__(self, a, b):  # pylint: disable=unused-argument
    pass


class PrintTest(absltest.TestCase):

  def test_path_to_str(self):
    self.assertEqual(printing._path_to_str(()), '')
    self.assertEqual(
        printing._path_to_str((printing._ParamName('x'), 0)), 'x[0]')
    self.assertEqual(
        printing._path_to_str((printing._ParamName('a'), 2, 'x')), "a[2]['x']")

  def test_simple_printing(self):
    cfg = fdl.Config(test_helper, 1, 'abc')
    output = printing.as_str_flattened(cfg)
    expected = """
x = 1
y = 'abc'
""".strip()
    self.assertEqual(output, expected)

  def test_unset_argument(self):
    cfg = fdl.Config(test_helper, 3.14)
    output = printing.as_str_flattened(cfg)
    expected = """
x = 3.14
y = <[unset]>
""".strip()
    self.assertEqual(output, expected)

  def test_nested(self):
    cfg = fdl.Config(test_helper, 'x', fdl.Config(test_helper, 'nest_x', 123))
    output = printing.as_str_flattened(cfg)
    expected = """
x = 'x'
y.x = 'nest_x'
y.y = 123
""".strip()
    self.assertEqual(output, expected)

  def test_class(self):
    cfg = fdl.Config(TestHelper, 'a_param', b=123)
    output = printing.as_str_flattened(cfg)
    expected = """
a = 'a_param'
b = 123
""".strip()
    self.assertEqual(output, expected)

  def test_kwargs(self):
    cfg = fdl.Config(test_kwarg_helper, 1, abc='extra kwarg value')
    output = printing.as_str_flattened(cfg)
    expected = """
x = 1
y = <[unset]>
abc = 'extra kwarg value'
""".strip()
    self.assertEqual(output, expected)

  def test_nested_kwargs(self):
    cfg = fdl.Config(
        test_kwarg_helper,
        extra=fdl.Config(test_kwarg_helper, 1, nested_extra='whee'))
    output = printing.as_str_flattened(cfg)
    expected = """
x = <[unset]>
y = <[unset]>
extra.x = 1
extra.y = <[unset]>
extra.nested_extra = 'whee'
""".strip()
    self.assertEqual(output, expected)

  def test_nested_collections(self):
    cfg = fdl.Config(
        test_helper,
        [fdl.Config(test_helper, 1, '1'),
         fdl.Config(TestHelper, 2)])
    output = printing.as_str_flattened(cfg)
    expected = """
x[0].x = 1
x[0].y = '1'
x[1].a = 2
x[1].b = <[unset]>
y = <[unset]>
""".strip()
    self.assertEqual(output, expected)

  def test_multiple_nested_collections(self):
    cfg = fdl.Config(test_helper, {
        'a': fdl.Config(test_kwarg_helper, abc=[1, 2, 3]),
        'b': [3, 2, 1]
    }, [fdl.Config(test_helper, [fdl.Config(test_helper, 1, 2)])])
    output = printing.as_str_flattened(cfg)
    expected = """
x['a'].x = <[unset]>
x['a'].y = <[unset]>
x['a'].abc = [1, 2, 3]
x['b'][0] = 3
x['b'][1] = 2
x['b'][2] = 1
y[0].x[0].x = 1
y[0].x[0].y = 2
y[0].y = <[unset]>
""".strip()
    self.assertEqual(output, expected)

  def test_default_values(self):

    def test_fn(w, x, y=3, z='abc'):  # pylint: disable=unused-argument
      pass

    cfg = fdl.Config(test_fn, 1)
    output = printing.as_str_flattened(cfg)
    expected = """
w = 1
x = <[unset]>
y = <[unset; default: 3]>
z = <[unset; default: 'abc']>
""".strip()
    self.assertEqual(output, expected)

  def test_placeholders(self):
    cfg = fdl.Config(
        test_helper,
        x=placeholders.Placeholder(key=test_key),
        y=placeholders.Placeholder(key=test_key, default='abc'))
    output = printing.as_str_flattened(cfg)
    expected = """
x = fdl.Placeholder('test_key', value=<[unset]>)
y = fdl.Placeholder('test_key', value='abc')
""".strip()
    self.assertEqual(output, expected)
    placeholders.set_placeholder(cfg, test_key, 'cba')
    output = printing.as_str_flattened(cfg)
    expected = """
x = fdl.Placeholder('test_key', value='cba')
y = fdl.Placeholder('test_key', value='cba')
""".strip()
    self.assertEqual(output, expected)

  def test_partial(self):
    partial = fdl.Partial(test_helper)
    partial.x = 'abc'
    output = printing.as_str_flattened(partial)
    expected = """
x = 'abc'
y = <[unset]>
""".strip()
    self.assertEqual(output, expected)


if __name__ == '__main__':
  absltest.main()
