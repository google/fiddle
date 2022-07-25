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

"""Tests for fiddle.codegen.py_val_to_cst_converter."""

import functools
import inspect
import logging.config
import re
import typing

from absl.testing import absltest
from absl.testing import parameterized

import fiddle as fdl
from fiddle.codegen import py_val_to_cst_converter
import fiddle.config as fdl_config

import libcst as cst


class SampleNamedTuple(typing.NamedTuple):
  x: typing.Any
  y: typing.Any


class SampleTag(fdl.Tag):
  """Fiddle tag for testing."""


class AnotherTag(fdl.Tag):
  """Fiddle tag for testing."""


def sample_fn(x, y, z):
  return (x, y, z)


def _get_cst_code(cst_module):
  # Depending on how the tests are run, objects defined in this file might be
  # named "x" or "fiddle.codegen.py_val_to_cst_converter_test.x"; normalize.
  return cst_module.code.replace('fiddle.codegen.py_val_to_cst_converter_test.',
                                 '')


class PyValToCstConverterTest(parameterized.TestCase):

  @parameterized.parameters([
      # Builtin value types:
      (5, '5'),
      (2.0, '2.0'),
      (2 + 3j, '(2.0+3.0j)'),
      (3j, '3j'),
      (True, 'True'),
      ('foo', "'foo'"),
      (b'bar', "b'bar'"),
      (None, 'None'),
      (Ellipsis, '...'),
      # Builtin container types:
      ([1, 2, 3], '[1, 2, 3]'),
      ((4, 5), '(4, 5)'),
      ((4,), '(4,)'),
      ({6}, '{6}'),
      (dict(a=8), "{'a': 8}"),
      ({}, '{}'),
      # Nested containers:
      ([1, (2, {3}), {
          4: 5,
          6: [7]
      }], '[1, (2, {3}), {4: 5, 6: [7]}]'),
      # Fiddle types:
      (fdl.Config(SampleNamedTuple,
                  3), 'fiddle.config.Config(SampleNamedTuple, x=3)'),
      (fdl.Partial(SampleNamedTuple,
                   [4]), 'fiddle.config.Partial(SampleNamedTuple, x=[4])'),
      (fdl.Config(re.match,
                  'a|b'), "fiddle.config.Config(re.match, pattern='a|b')"),
      (SampleTag.new(123), 'SampleTag.new(123)'),
      # NamedTuples:
      (SampleNamedTuple(1, 2), 'SampleNamedTuple(x=1, y=2)'),
      # Modules:
      (re, 're'),
      (logging.config, 'logging.config'),
      (py_val_to_cst_converter, 'fiddle.codegen.py_val_to_cst_converter'),
      # Importable values:
      (re.search, 're.search'),
      (logging.config.dictConfig, 'logging.config.dictConfig'),
      (inspect.Signature.bind, 'inspect.Signature.bind'),
      # Builtins:
      (sum, 'sum'),
      (slice, 'slice'),
      # Partials:
      (functools.partial(sample_fn), 'functools.partial(sample_fn)'),
      (functools.partial(sample_fn, 3,
                         z=4), 'functools.partial(sample_fn, 3, z=4)'),
  ])
  def test_convert(self, pyval, expected):
    cst_expr = py_val_to_cst_converter.convert_py_val_to_cst(pyval)
    cst_module = cst.Module([cst.SimpleStatementLine([cst.Expr(cst_expr)])])
    self.assertEqual(_get_cst_code(cst_module), expected + '\n')

  def test_convert_multiple_tags(self):
    pyval = fdl.TaggedValue([SampleTag, AnotherTag], 3)
    cst_expr = py_val_to_cst_converter.convert_py_val_to_cst(pyval)
    cst_module = cst.Module([cst.SimpleStatementLine([cst.Expr(cst_expr)])])
    self.assertEqual(
        _get_cst_code(cst_module), 'AnotherTag.new(SampleTag.new(3))\n')

  def test_convert_new_tags(self):
    pyval = fdl.Config(SampleNamedTuple, x=1)
    fdl_config.add_tag(pyval, 'x', SampleTag)
    cst_expr = py_val_to_cst_converter.convert_py_val_to_cst(pyval)
    cst_module = cst.Module([cst.SimpleStatementLine([cst.Expr(cst_expr)])])
    self.assertEqual(
        _get_cst_code(cst_module),
        'fiddle.config.Config(SampleNamedTuple, x=SampleTag.new(1))\n')

  def test_convert_empty_set(self):
    cst_expr = py_val_to_cst_converter.convert_py_val_to_cst(set())
    cst_module = cst.Module([cst.SimpleStatementLine([cst.Expr(cst_expr)])])
    self.assertEqual(_get_cst_code(cst_module), 'set()\n')

  def test_convert_unsupported_type(self):
    with self.assertRaisesRegex(ValueError, 'has no registered converter for'):
      py_val_to_cst_converter.convert_py_val_to_cst(object())

  def test_additional_converters(self):
    x = [1]
    pyval = [1, {2: x}, fdl.Config(re.match, 'a|b')]

    def convert_named_value(value, convert_child, id_to_name):
      del convert_child  # Unused.
      if id(value) in id_to_name:
        return cst.Name(id_to_name[id(value)])
      else:
        return None

    id_to_name = {id(x): 'x', id(fdl.Config): 'MyFiddleConfig'}

    custom_converter = py_val_to_cst_converter.ValueConverter(
        matcher=lambda value: True,
        priority=200,  # Use priority>100 to run before all standard converters.
        converter=functools.partial(convert_named_value, id_to_name=id_to_name))

    cst_expr = py_val_to_cst_converter.convert_py_val_to_cst(
        pyval, [custom_converter])
    cst_module = cst.Module([cst.SimpleStatementLine([cst.Expr(cst_expr)])])
    self.assertEqual(
        _get_cst_code(cst_module),
        "[1, {2: x}, MyFiddleConfig(re.match, pattern='a|b')]\n")

    cst_expr = py_val_to_cst_converter.convert_py_val_to_cst(pyval)
    cst_module = cst.Module([cst.SimpleStatementLine([cst.Expr(cst_expr)])])
    self.assertEqual(
        _get_cst_code(cst_module),
        "[1, {2: [1]}, fiddle.config.Config(re.match, pattern='a|b')]\n")


if __name__ == '__main__':
  absltest.main()
