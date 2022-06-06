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

"""Tests for fiddle.codegen.py_val_to_ast_converter."""

import ast
import functools
import inspect
import logging.config
import re
import typing

from absl.testing import absltest
from absl.testing import parameterized

import fiddle as fdl
from fiddle.codegen import py_val_to_ast_converter


class TestNamedTuple(typing.NamedTuple):
  x: typing.Any
  y: typing.Any


class TestTag(fdl.Tag):
  """Fiddle tag for testing."""


class AnotherTag(fdl.Tag):
  """Fiddle tag for testing."""


class PyValToAstConverterTest(parameterized.TestCase):

  @parameterized.parameters([
      # Builtin value types:
      (5, '5'),
      (2.0, '2.0'),
      (2 + 3j, '(2+3j)'),
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
      (fdl.Config(TestNamedTuple,
                  3), 'fiddle.config.Config(TestNamedTuple, x=3)'),
      (fdl.Partial(TestNamedTuple,
                   [4]), 'fiddle.config.Partial(TestNamedTuple, x=[4])'),
      (fdl.Config(re.match,
                  'a|b'), "fiddle.config.Config(re.match, pattern='a|b')"),
      (TestTag.new(123), 'TestTag.new(123)'),
      # NamedTuples:
      (TestNamedTuple(1, 2), 'TestNamedTuple(x=1, y=2)'),
      # Modules:
      (re, 're'),
      (logging.config, 'logging.config'),
      (py_val_to_ast_converter, 'fiddle.codegen.py_val_to_ast_converter'),
      # Importable values:
      (re.search, 're.search'),
      (logging.config.dictConfig, 'logging.config.dictConfig'),
      (inspect.Signature.bind, 'inspect.Signature.bind'),
      # Builtins:
      (sum, 'sum'),
      (slice, 'slice'),
  ])
  def test_convert(self, pyval, expected):
    if not hasattr(ast, 'unparse'):
      self.skipTest('ast.unparse requires Python 3.9+')
    ast_node = py_val_to_ast_converter.convert_py_val_to_ast(pyval)
    self.assertEqual(ast.unparse(ast_node), expected)

  def test_convert_multiple_tags(self):
    if not hasattr(ast, 'unparse'):
      self.skipTest('ast.unparse requires Python 3.9+')
    pyval = fdl.TaggedValue([TestTag, AnotherTag], 3)
    ast_node = py_val_to_ast_converter.convert_py_val_to_ast(pyval)
    self.assertIn(
        ast.unparse(ast_node),
        ['TestTag.new(AnotherTag.new(3))', 'AnotherTag.new(TestTag.new(3))'])

  def test_convert_empty_set(self):
    if not hasattr(ast, 'unparse'):
      self.skipTest('ast.unparse requires Python 3.9+')
    ast_node = py_val_to_ast_converter.convert_py_val_to_ast(set())
    self.assertEqual(ast.unparse(ast_node), '{*()}')

  def test_convert_unsupported_type(self):
    with self.assertRaisesRegex(ValueError, 'has no registered converter for'):
      py_val_to_ast_converter.convert_py_val_to_ast(object())

  def test_additional_converters(self):
    if not hasattr(ast, 'unparse'):
      self.skipTest('ast.unparse requires Python 3.9+')
    x = [1]
    pyval = [1, {2: x}, fdl.Config(re.match, 'a|b')]

    def convert_named_value(value, convert_child, id_to_name):
      del convert_child  # Unused.
      if id(value) in id_to_name:
        return ast.Name(id_to_name[id(value)])
      else:
        return None

    id_to_name = {id(x): 'x', id(fdl.Config): 'MyFiddleConfig'}

    custom_converter = py_val_to_ast_converter.ValueConverter(
        matcher=lambda value: True,
        priority=200,  # Use priority>100 to run before all standard converters.
        converter=functools.partial(convert_named_value, id_to_name=id_to_name))

    with_custom_converter = py_val_to_ast_converter.convert_py_val_to_ast(
        pyval, [custom_converter])
    self.assertEqual(
        ast.unparse(with_custom_converter),
        "[1, {2: x}, MyFiddleConfig(re.match, pattern='a|b')]")

    without_custom_converter = py_val_to_ast_converter.convert_py_val_to_ast(
        pyval)
    self.assertEqual(
        ast.unparse(without_custom_converter),
        "[1, {2: [1]}, fiddle.config.Config(re.match, pattern='a|b')]")


if __name__ == '__main__':
  absltest.main()
