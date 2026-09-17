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

"""Tests for lazy_import."""

import textwrap
from typing import List

from absl.testing import absltest
import fiddle as fdl
from fiddle._src.experimental import lazy_import_test_example
from fiddle.experimental import lazy_import


# pylint: disable=unused-argument
# pytype: disable=bad-return-type

_EXAMPLE_MODULE = 'fiddle._src.experimental.lazy_import_test_example'


def build_lazy_my_function():
  @lazy_import.lazy_import(_EXAMPLE_MODULE, 'my_function')
  def my_function(x: int, y: float = 1.0) -> List[float]:
    ...

  return my_function


def build_lazy_my_class():
  @lazy_import.lazy_import(_EXAMPLE_MODULE, 'MyClass')
  class MyClass:

    def __init__(self, name: str):
      ...

  return MyClass


def build_lazy_my_dataclass():
  @lazy_import.lazy_import(_EXAMPLE_MODULE, 'MyDataclass')
  class MyDataclass:

    def __init__(self, a: int, b: float):
      ...

  return MyDataclass


def build_lazy_another_function(my_class, my_dataclass):
  @lazy_import.lazy_import(_EXAMPLE_MODULE, 'another_function')
  def another_funciton(a: my_class, b: List[my_dataclass], **kwargs):
    ...

  return another_funciton


class LazyImportTest(absltest.TestCase):

  def test_lazy_function(self):
    my_function = build_lazy_my_function()
    self.assertIsNone(my_function.fn_or_cls)
    self.assertEqual(my_function(2, 4.0), [4.0, 4.0])
    self.assertIsNotNone(my_function.fn_or_cls)

  def test_lazy_class(self):
    MyClass = build_lazy_my_class()
    self.assertIsNone(MyClass.lazy_fn.fn_or_cls)
    self.assertEqual(MyClass('matthew').reversed_name(), 'wehttam')
    self.assertIsNotNone(MyClass.lazy_fn.fn_or_cls)

  def test_lazy_dataclass(self):
    MyDataclass = build_lazy_my_dataclass()
    self.assertIsNone(MyDataclass.lazy_fn.fn_or_cls)
    inst = MyDataclass(1, 2)
    self.assertEqual(repr(inst), 'MyDataclass(a=1, b=2)')
    self.assertIsNotNone(MyDataclass.lazy_fn.fn_or_cls)

  def test_lazy_another_function(self):
    MyClass = build_lazy_my_class()
    MyDataclass = build_lazy_my_dataclass()
    another_function = build_lazy_another_function(MyClass, MyDataclass)
    result = another_function(MyClass('foo'), b=[MyDataclass(1, 2.0)], c='c')
    self.assertEqual(
        repr(result), "(MyClass('foo'), [MyDataclass(a=1, b=2.0)], {'c': 'c'})"
    )

  def test_lazy_function_config(self):
    my_function = build_lazy_my_function()
    self.assertIsNone(my_function.fn_or_cls)
    cfg = fdl.Config(my_function, x=5, y=2.0)
    self.assertEqual(fdl.build(cfg), [2.0, 2.0, 2.0, 2.0, 2.0])
    self.assertIsNotNone(my_function.fn_or_cls)

  def test_lazy_class_config(self):
    MyClass = build_lazy_my_class()
    cfg = fdl.Config(MyClass, name='matthew')
    self.assertIsNone(MyClass.lazy_fn.fn_or_cls)
    self.assertEqual(fdl.build(cfg).reversed_name(), 'wehttam')
    self.assertIsNotNone(MyClass.lazy_fn.fn_or_cls)

  def test_lazy_another_function_config(self):
    MyClass = build_lazy_my_class()
    MyDataclass = build_lazy_my_dataclass()
    another_function = build_lazy_another_function(MyClass, MyDataclass)
    cfg = fdl.Config(
        another_function,
        a=fdl.Config(MyClass, 'foo'),
        b=[fdl.Config(MyDataclass, 1, 2.0)],
        c='c',
    )
    self.assertIsNone(MyClass.lazy_fn.fn_or_cls)
    self.assertIsNone(MyDataclass.lazy_fn.fn_or_cls)
    self.assertIsNone(another_function.fn_or_cls)
    result = fdl.build(cfg)
    self.assertEqual(
        repr(result), "(MyClass('foo'), [MyDataclass(a=1, b=2.0)], {'c': 'c'})"
    )

  def test_ellipsis_for_default_value(self):
    # It's ok to use ellipsis for default values.  Note that pytype does
    # not complain about the type mismatch between `float` and `...`.
    # (pytype has a special case for `...` in signature.)
    @lazy_import.lazy_import(_EXAMPLE_MODULE, 'my_function')
    def my_function(x: int, y: float = ...) -> List[float]:
      ...

    self.assertEqual(my_function(2), [1.0, 1.0])

  def test_error_bad_function_stub(self):
    with self.assertRaisesRegex(
        ValueError,
        'The body for functions decorated with `lazy_import` '
        r'should be an ellipsis mark or a docstring\.',
    ):

      @lazy_import.lazy_import(_EXAMPLE_MODULE, 'my_function')
      def my_function(x: int, y: float = 1.0):
        return x + y

  def test_error_bad_class_stub(self):
    with self.assertRaisesRegex(
        ValueError,
        'Classes decorated with `lazy_import` should define '
        r'a stub method __init__ \(and nothing else\)\.',
    ):

      @lazy_import.lazy_import(_EXAMPLE_MODULE, 'MyClass')
      class MyClass:  # pylint: disable=unused-variable
        x: int

  def test_error_module_does_not_exist(self):
    @lazy_import.lazy_import('module_does_not_exist', 'my_function')
    def my_function():
      ...

    with self.assertRaisesRegex(
        ImportError, "No module named 'module_does_not_exist"
    ):
      my_function()

  def test_error_symbol_does_not_exist(self):
    @lazy_import.lazy_import('re', 'function_does_not_exist')
    def my_function():
      ...

    with self.assertRaisesRegex(
        ImportError, 'cannot import function_does_not_exist from re'
    ):
      my_function()

  def test_error_import_not_callable(self):
    @lazy_import.lazy_import('re', 'MULTILINE')
    def my_function():
      ...

    with self.assertRaisesRegex(
        TypeError, 'Expected re:MULTILINE to be callable; got re.MULTILINE'
    ):
      my_function()

  def test_error_signature_mismatch(self):
    @lazy_import.lazy_import('re', 'sub')
    def re_sub(foo, bar):
      ...

    with self.assertRaisesRegex(
        ValueError,
        r'Expected re:sub to have signature \(foo, bar\); '
        r'got .* with signature \(pattern, repl, string, count=0, flags=0\)',
    ):
      re_sub('a', 'b')

  def test_error_bad_module_name(self):
    with self.assertRaisesRegex(
        ValueError, r"Expected `module` to be a dotted name, got ':\)'\."
    ):

      @lazy_import.lazy_import(':)', 'my_function')
      def my_function():
        ...

  def test_error_bad_qualname(self):
    with self.assertRaisesRegex(
        ValueError,
        "Expected `qualname` to be a dotted name, got 'foo.<locals>.bar'.",
    ):

      @lazy_import.lazy_import('my_module', 'foo.<locals>.bar')
      def my_function():
        ...

  def test_codegen_lazy_import_module(self):
    code = lazy_import.codegen_lazy_import_module(lazy_import_test_example)
    expected = """\
      # Fiddle lazy_import wrappers for fiddle._src.experimental.lazy_import_test_example
      # THIS FILE WAS AUTOMATICALLY GENERATED.

      from fiddle.experimental import lazy_import as _lazy_import

      # pylint: disable=unused-argument
      # pytype: disable=bad-return-type


      @_lazy_import.lazy_import(fiddle._src.experimental.lazy_import_test_example, List)
      def List(*args, **kwargs):
        ...


      @_lazy_import.lazy_import(fiddle._src.experimental.lazy_import_test_example, my_function)
      def my_function(x: int, y: float = Ellipsis) -> List[float]:
        ...


      @_lazy_import.lazy_import(fiddle._src.experimental.lazy_import_test_example, MyClass)
      class MyClass:
        def __init__(name: str):
          ...


      @_lazy_import.lazy_import(fiddle._src.experimental.lazy_import_test_example, MyDataclass)
      class MyDataclass:
        def __init__(a: int, b: float) -> None:
          ...


      @_lazy_import.lazy_import(fiddle._src.experimental.lazy_import_test_example, another_function)
      def another_function(a: fiddle._src.experimental.lazy_import_test_example.MyClass, b: List[fiddle._src.experimental.lazy_import_test_example.MyDataclass], **kwargs):
        ...
      """
    self.assertEqual(code.strip(), textwrap.dedent(expected).strip())


if __name__ == '__main__':
  absltest.main()
