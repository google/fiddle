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

"""Tests for auto_config."""

import dataclasses
import functools
import inspect
import sys
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized

import fiddle as fdl
from fiddle import testing
from fiddle.experimental import auto_config
from fiddle.experimental import autobuilders as ab


def basic_fn(arg, kwarg='test'):
  return {'arg': arg, 'kwarg': kwarg}


def fn_with_kwargs(**kwargs):
  return {'kwargs': kwargs}


@dataclasses.dataclass(frozen=True)
class SampleClass:
  arg1: Any
  arg2: Any

  def method(self):
    return 42


class TestTag(fdl.Tag):
  """Sample Tag used for tests."""


class TestTag2(fdl.Tag):
  """Sample Tag used for tests."""


def explicit_config_building_fn(x: int) -> fdl.Config:
  return fdl.Config(basic_fn, 5, kwarg=x)


def pass_through(arg):
  return arg


def _line_number():
  return sys._getframe(1).f_lineno


class AutoConfigTest(testing.TestCase, parameterized.TestCase):

  def test_create_basic_config(self):
    expected_config = fdl.Config(SampleClass, 1, arg2=2)

    @auto_config.auto_config
    def test_class_config():
      return SampleClass(1, 2)

    self.assertEqual(expected_config, test_class_config.as_buildable())
    self.assertEqual(SampleClass(1, 2), test_class_config())

  def test_create_basic_config_parents(self):
    expected_config = fdl.Config(SampleClass, 1, arg2=2)

    @auto_config.auto_config()  # Note the parenthesis!
    def test_class_config():
      return SampleClass(1, 2)

    self.assertEqual(expected_config, test_class_config.as_buildable())
    self.assertEqual(SampleClass(1, 2), test_class_config())

  def test_create_basic_partial(self):
    expected_config = fdl.Partial(basic_fn, 1, kwarg='kwarg')

    @auto_config.auto_config
    def test_fn_config():
      return functools.partial(basic_fn, 1, 'kwarg')

    self.assertEqual(expected_config, test_fn_config.as_buildable())

  def test_create_config_with_args(self):
    expected_config = fdl.Config(SampleClass, 'positional', arg2='default')

    @auto_config.auto_config
    def test_class_config(arg1, arg2='default'):
      return SampleClass(arg1, arg2)

    self.assertEqual(expected_config,
                     test_class_config.as_buildable('positional'))
    self.assertEqual(
        SampleClass('positional', 'default'), test_class_config('positional'))

  def test_create_config_with_kwonly_args(self):
    expected_config = fdl.Config(SampleClass, 'positional', arg2='default')

    @auto_config.auto_config
    def test_class_config(arg1, *, arg2='default'):
      return SampleClass(arg1, arg2)

    self.assertEqual(expected_config,
                     test_class_config.as_buildable('positional'))
    self.assertEqual(
        SampleClass('positional', 'default'), test_class_config('positional'))

  def test_calling_auto_config(self):
    expected_config = fdl.Config(
        basic_fn, 1, kwarg=fdl.Config(SampleClass, 1, 2))

    @auto_config.auto_config
    def test_class_config():
      return SampleClass(1, arg2=2)

    @auto_config.auto_config
    def test_fn_config():
      return basic_fn(1, test_class_config())

    self.assertEqual(expected_config, test_fn_config.as_buildable())
    self.assertEqual({
        'arg': 1,
        'kwarg': SampleClass(1, arg2=2)
    }, test_fn_config())

  def test_nested_calls(self):
    expected_config = fdl.Config(
        SampleClass, 1, arg2=fdl.Config(basic_fn, 2, 'kwarg'))

    @auto_config.auto_config
    def test_class_config():
      return SampleClass(1, basic_fn(2, 'kwarg'))

    self.assertEqual(expected_config, test_class_config.as_buildable())
    self.assertEqual(
        SampleClass(1, {
            'arg': 2,
            'kwarg': 'kwarg'
        }), test_class_config())

  def test_calling_explicit_function(self):
    expected_config = fdl.Config(
        SampleClass, 1, arg2=fdl.Config(basic_fn, 5, 10))

    @auto_config.auto_config
    def test_nested_call():
      return SampleClass(1, explicit_config_building_fn(10))

    self.assertEqual(expected_config, test_nested_call.as_buildable())

  def test_reference_nonlocal(self):
    nonlocal_var = 3
    expected_config = fdl.Config(basic_fn, 1, kwarg=nonlocal_var)

    @auto_config.auto_config
    def test_fn_config():
      return basic_fn(1, nonlocal_var)

    self.assertEqual(expected_config, test_fn_config.as_buildable())

  def test_closures_in_multi_level_nesting(self):
    var_level_0 = 0

    def outer():
      var_level_1 = 1

      @auto_config.auto_config
      def inner():
        var_level_2 = 2
        return pass_through((var_level_0, var_level_1, var_level_2))

      return inner

    inner_fn = outer()
    cfg = inner_fn.as_buildable()
    expected_result = (0, 1, 2)
    self.assertEqual(cfg.arg, expected_result)
    self.assertEqual(expected_result, inner_fn())
    self.assertEqual(expected_result, fdl.build(cfg))

  def test_calling_builtins(self):
    expected_config = fdl.Config(SampleClass, [0, 1, 2], ['a', 'b'])

    @auto_config.auto_config
    def test_config():
      return SampleClass(list(range(3)), list({'a': 0, 'b': 1}.keys()))

    self.assertEqual(expected_config, test_config.as_buildable())

  def test_auto_config_eligibility(self):
    # Some common types that have no signature.
    for builtin_type in (range, dict):
      self.assertFalse(auto_config._is_auto_config_eligible(builtin_type))
    # Some common builtins.
    for builtin in (list, tuple, sum, any, all, iter):
      self.assertFalse(auto_config._is_auto_config_eligible(builtin))
    self.assertFalse(auto_config._is_auto_config_eligible([].append))
    self.assertFalse(auto_config._is_auto_config_eligible({}.keys))
    # A method.
    test_class = SampleClass(1, 2)
    self.assertFalse(auto_config._is_auto_config_eligible(test_class.method))
    # Buildable subclasses.
    self.assertFalse(auto_config._is_auto_config_eligible(fdl.Config))
    self.assertFalse(auto_config._is_auto_config_eligible(fdl.Partial))
    # Auto-config annotations are not eligible, and this case is tested in
    # `test_calling_auto_config` above.

  def test_autobuilders_in_auto_config(self):
    expected_config = fdl.Config(
        basic_fn, arg=ab.config(SampleClass, require_skeleton=False))

    @auto_config.auto_config
    def autobuilder_using_fn():
      x = ab.config(SampleClass, require_skeleton=False)
      return basic_fn(x)

    self.assertEqual(expected_config, autobuilder_using_fn.as_buildable())

  def test_auto_configuring_non_function(self):
    with self.assertRaisesRegex(ValueError, 'only compatible with functions'):
      auto_config.auto_config(3)

  def test_return_structure(self):
    expected_config = {
        'test_key1': fdl.Config(SampleClass, 1, arg2=2),
        'test_key2': [fdl.Config(SampleClass, 3, 4), 5],
        'test_key3': (fdl.Partial(pass_through, 'arg'), 6),
    }

    @auto_config.auto_config
    def test_config():
      return {
          'test_key1': SampleClass(1, 2),
          'test_key2': [SampleClass(3, 4), 5],
          'test_key3': (functools.partial(pass_through, 'arg'), 6),
      }

    buildable = test_config.as_buildable()
    self.assertEqual(expected_config, buildable)

  def test_require_buildable_in_return_type(self):

    @auto_config.auto_config
    def test_config():
      return {'no buildable': ['inside here']}

    with self.assertRaisesRegex(
        TypeError, r'The `auto_config` rewritten version of '
        r'`AutoConfigTest\.\w+\.<locals>\.test_config` returned a `dict`, '
        r'which is not \(or did not contain\) a `fdl\.Buildable`\.'):
      test_config.as_buildable()

  def test_staticmethod_nullary(self):

    class MyClass:

      @staticmethod
      @auto_config.auto_config
      def my_fn():
        return SampleClass(2, 1)

    self.assertEqual(
        fdl.Config(SampleClass, 2, 1), MyClass.my_fn.as_buildable())
    self.assertEqual(SampleClass(2, 1), MyClass.my_fn())

    instance = MyClass()
    self.assertEqual(
        fdl.Config(SampleClass, 2, 1), instance.my_fn.as_buildable())
    self.assertEqual(SampleClass(2, 1), instance.my_fn())

  def test_staticmethod_arguments(self):

    class MyClass:

      @staticmethod
      @auto_config.auto_config
      def my_fn(x):
        return SampleClass(x, x + 1)

    self.assertEqual(
        fdl.Config(SampleClass, 5, 6), MyClass.my_fn.as_buildable(5))
    self.assertEqual(SampleClass(5, 6), MyClass.my_fn(5))

    instance = MyClass()

    self.assertEqual(
        fdl.Config(SampleClass, 5, 6), instance.my_fn.as_buildable(5))
    self.assertEqual(SampleClass(5, 6), instance.my_fn(5))

  def test_staticmethod_not_on_top(self):

    class MyClass:  # pylint: disable=unused-variable

      @auto_config.auto_config
      @staticmethod
      def my_fn(x, y):
        return SampleClass(x, y)

    instance = MyClass()
    cfg = instance.my_fn.as_buildable(x=3, y=5)

    self.assertEqual(SampleClass, cfg.__fn_or_cls__)
    self.assertEqual(3, cfg.arg1)
    self.assertEqual(5, cfg.arg2)

  def test_staticmethod_ctor(self):

    @dataclasses.dataclass(frozen=True)
    class ClassWithStaticmethodCtor:
      a: int
      b: float

      @staticmethod
      @auto_config.auto_config
      def staticmethod_init(b=1.0):
        return ClassWithStaticmethodCtor(1, b)

    cfg = ClassWithStaticmethodCtor.staticmethod_init.as_buildable(2.0)
    self.assertEqual(cfg.a, 1)
    self.assertEqual(cfg.b, 2.0)

  def test_classmethod(self):

    @dataclasses.dataclass
    class MyClass:
      x: int
      y: str
      z: float = 2.0

      @classmethod
      @auto_config.auto_config
      def simple(cls):
        return cls(x=1, y='1', z=1.0)

    cfg = MyClass.simple.as_buildable()
    self.assertEqual(MyClass, cfg.__fn_or_cls__)
    self.assertEqual(1, cfg.x)
    self.assertEqual('1', cfg.y)
    self.assertEqual(1.0, cfg.z)

    class_instance = MyClass(1, '2')
    cfg = class_instance.simple.as_buildable()
    self.assertEqual(MyClass, cfg.__fn_or_cls__)
    self.assertEqual(1, cfg.x)
    self.assertEqual('1', cfg.y)
    self.assertEqual(1.0, cfg.z)

  def test_classmethod_not_on_top(self):

    class MyClass:  # pylint: disable=unused-variable

      def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

      @auto_config.auto_config
      @classmethod
      def my_fn(cls, x, y):
        return cls(x, y)

    instance = MyClass()
    cfg = instance.my_fn.as_buildable(x=3, y=5)

    self.assertEqual(MyClass, cfg.__fn_or_cls__)

    obj = fdl.build(cfg)

    self.assertEqual(3, obj.x)
    self.assertEqual(5, obj.y)

  def test_call_super(self):

    class MyClass(SampleClass):  # pylint: disable=unused-variable

      def __init__(self):
        super().__init__(1, 2)

      @auto_config.auto_config
      def my_fn(self, x):
        return basic_fn(x, kwarg=super().method())

    instance = MyClass()
    cfg = instance.my_fn.as_buildable('a')
    self.assertEqual(cfg.__fn_or_cls__, basic_fn)
    self.assertEqual(cfg.arg, 'a')
    self.assertEqual(cfg.kwarg, 42)

  def test_function_metadata(self):

    @auto_config.auto_config
    def my_helpful_function(x: int, y: str = 'y') -> SampleClass:
      """A docstring."""
      return SampleClass(x, y)

    self.assertEqual('A docstring.', inspect.getdoc(my_helpful_function))
    param_kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    expected_parameters = (inspect.Parameter(
        'x', kind=param_kind, annotation=int),
                           inspect.Parameter(
                               'y',
                               kind=param_kind,
                               default='y',
                               annotation=str))
    expected_signature = inspect.Signature(
        expected_parameters, return_annotation=SampleClass)
    self.assertEqual(expected_signature, inspect.signature(my_helpful_function))
    self.assertEqual('my_helpful_function', my_helpful_function.__name__)

  def test_control_flow_if(self):

    def test_config(condition):
      if condition:
        return pass_through('true branch')
      else:
        return pass_through('false branch')

    error_line_number = _line_number() - 5

    with self.assertRaisesRegex(
        auto_config.UnsupportedLanguageConstructError,
        r'Control flow \(If\) is unsupported by auto_config\. '
        r'\(auto_config_test\.py, line \d+\)') as context:
      auto_config.auto_config(test_config)

    self.assertEqual(error_line_number, context.exception.lineno)
    # The text and offset here don't quite match with the indentation level of
    # test_config above, due to the `dedent` performed inside `auto_config`.
    # This slight discrepancy is unlikely to cause any confusion.
    self.assertEqual('  if condition:', context.exception.text)
    self.assertEqual(2, context.exception.offset)

    build_config_fn = auto_config.auto_config(
        test_config, experimental_allow_control_flow=True).as_buildable
    self.assertEqual('true branch', fdl.build(build_config_fn(True)))
    self.assertEqual('false branch', fdl.build(build_config_fn(False)))

  def test_control_flow_for(self):

    def test_config():
      layers = []
      for i in range(3):
        layers.append(SampleClass(i, i))
      return pass_through(layers)

    with self.assertRaisesRegex(
        auto_config.UnsupportedLanguageConstructError,
        r'Control flow \(For\) is unsupported by auto_config\.'):
      auto_config.auto_config(test_config)

    expected_config = fdl.Config(
        pass_through, [fdl.Config(SampleClass, i, i) for i in range(3)])
    actual_config = auto_config.auto_config(
        test_config, experimental_allow_control_flow=True).as_buildable()
    self.assertEqual(expected_config, actual_config)

  def test_control_flow_while(self):

    def test_config():
      i = 10
      while i > 0:
        i -= 1
      return pass_through(i)

    with self.assertRaisesRegex(
        auto_config.UnsupportedLanguageConstructError,
        r'Control flow \(While\) is unsupported by auto_config\.'):
      auto_config.auto_config(test_config)

    pass_through_config = auto_config.auto_config(
        test_config, experimental_allow_control_flow=True).as_buildable()
    self.assertEqual(0, fdl.build(pass_through_config))

  @parameterized.parameters(
      (lambda: pass_through([i + 1 for i in [0, 1, 2]]), 'ListComp'),
      (lambda: pass_through({i + 1 for i in [0, 1, 2]}), 'SetComp'),
      (lambda: pass_through({i + 1: None for i in [0, 1, 2]}), 'DictComp'),
      (lambda: pass_through((i + 1 for i in [0, 1, 2])), 'GeneratorExp'),
      (lambda: pass_through(None if list() else [1, 2, 3]), 'IfExp'),
  )
  def test_control_flow_expression(self, test_config, node_name):
    with self.assertRaisesRegex(
        auto_config.UnsupportedLanguageConstructError,
        rf'Control flow \({node_name}\) is unsupported by auto_config\.'):
      auto_config.auto_config(test_config)

    pass_through_config = auto_config.auto_config(
        test_config, experimental_allow_control_flow=True).as_buildable()
    self.assertCountEqual([1, 2, 3], fdl.build(pass_through_config))

  def test_allow_control_flow_decorator(self):

    @auto_config.auto_config(experimental_allow_control_flow=True)
    def test_config():
      return pass_through([i + 1 for i in range(3)])

    self.assertEqual([1, 2, 3], fdl.build(test_config.as_buildable()))

  def test_disallow_try(self):

    def test_config():
      try:
        pass
      except:  # pylint: disable=bare-except
        pass

    with self.assertRaisesRegex(
        auto_config.UnsupportedLanguageConstructError,
        r'Control flow \(Try\) is unsupported by auto_config\.'):
      auto_config.auto_config(test_config)

  def test_disallow_raise(self):

    def test_config():
      raise Exception()

    with self.assertRaisesRegex(
        auto_config.UnsupportedLanguageConstructError,
        r'Control flow \(Raise\) is unsupported by auto_config\.'):
      auto_config.auto_config(test_config)

  def test_disallow_with(self):

    def test_config():
      with open(''):
        pass

    with self.assertRaisesRegex(
        auto_config.UnsupportedLanguageConstructError,
        r'Control flow \(With\) is unsupported by auto_config\.'):
      auto_config.auto_config(test_config)

  def test_disallow_yield(self):

    def test_config():
      yield

    with self.assertRaisesRegex(
        auto_config.UnsupportedLanguageConstructError,
        r'Control flow \(Yield\) is unsupported by auto_config\.'):
      auto_config.auto_config(test_config)

  def test_disallow_yield_from(self):

    def test_config():
      yield from (i for i in range(5))

    with self.assertRaisesRegex(
        auto_config.UnsupportedLanguageConstructError,
        r'Control flow \(YieldFrom\) is unsupported by auto_config\.'):
      auto_config.auto_config(test_config)

  def test_disallow_function_definitions(self):

    def test_config():

      def nested_fn():
        pass

      return nested_fn()

    with self.assertRaisesRegex(
        auto_config.UnsupportedLanguageConstructError,
        r'Nested function definitions are not supported by auto_config\.'):
      auto_config.auto_config(test_config)

  def test_disallow_class_definitions(self):

    def test_config():

      class NestedClass:
        pass

      return NestedClass()

    with self.assertRaisesRegex(
        auto_config.UnsupportedLanguageConstructError,
        r'Class definitions are not supported by auto_config\.'):
      auto_config.auto_config(test_config)

  def test_disallow_lambda_definitions(self):

    def test_config():
      return pass_through(lambda: SampleClass(1, 2))

    with self.assertRaisesRegex(
        auto_config.UnsupportedLanguageConstructError,
        r'Lambda definitions are not supported by auto_config\.'):
      auto_config.auto_config(test_config)

  def test_disallow_async_function_definitions(self):

    async def test_config():
      pass

    with self.assertRaisesRegex(
        auto_config.UnsupportedLanguageConstructError,
        r'Async function definitions are not supported by auto_config\.'):
      auto_config.auto_config(test_config)

  def test_tag_new(self):

    @auto_config.auto_config
    def fn():
      return SampleClass(4, TestTag.new(5))

    with self.subTest('dunder_call'):
      self.assertEqual(fn(), SampleClass(4, 5))

    with self.subTest('as_buildable'):
      self.assertDagEqual(fn.as_buildable(),
                          fdl.Config(SampleClass, 4, TestTag.new(5)))

  def test_tag_new_without_default(self):

    @auto_config.auto_config
    def fn():
      return SampleClass(4, TestTag.new())

    with self.subTest('dunder_call'):
      with self.assertRaisesRegex(
          ValueError, 'Expected all `TaggedValue`s to be replaced via '
          r'fdl.set_tagged\(\) calls, but one was not set\.'):
        fn()

    with self.subTest('as_buildable'):
      self.assertDagEqual(fn.as_buildable(),
                          fdl.Config(SampleClass, 4, TestTag.new()))

  def test_add_tag(self):

    @auto_config.auto_config
    def fn():
      x = SampleClass(4, 5)
      fdl.add_tag(x, 'arg1', TestTag)
      return x

    with self.subTest('dunder_call'):
      self.assertEqual(fn(), SampleClass(4, 5))

    with self.subTest('as_buildable'):
      expected_cfg = fdl.Config(SampleClass, 4, 5)
      fdl.add_tag(expected_cfg, 'arg1', TestTag)
      self.assertDagEqual(fn.as_buildable(), expected_cfg)

  def test_tag_manipulations(self):
    # Tests all 5 tag-manipulation functions (add, set, remove, clear, get).

    @auto_config.auto_config
    def inner_config():
      result = SampleClass(basic_fn(1, 2), 3)
      fdl.add_tag(result, 'arg1', TestTag)
      fdl.set_tags(result.arg1, 'arg', [TestTag, TestTag2])
      fdl.add_tag(result.arg1, 'kwarg', TestTag)
      fdl.add_tag(result.arg1, 'kwarg', TestTag2)
      return result

    @auto_config.auto_config
    def outer_config():
      result = inner_config()
      # Modify the tags that were assigned in inner_config.
      fdl.set_tags(result, 'arg2', fdl.get_tags(result, 'arg1'))
      fdl.clear_tags(result, 'arg1')
      fdl.remove_tag(result.arg1, 'kwarg', TestTag)
      return result

    with self.subTest('inner_config.dunder_call'):
      self.assertEqual(inner_config(), SampleClass(basic_fn(1, 2), 3))

    with self.subTest('outer_config.dunder_call'):
      self.assertEqual(outer_config(), SampleClass(basic_fn(1, 2), 3))

    with self.subTest('inner_config.as_buildable'):
      cfg = inner_config.as_buildable()
      expected = fdl.Config(SampleClass, fdl.Config(basic_fn, 1, 2), 3)
      fdl.set_tags(expected, 'arg1', [TestTag])
      fdl.set_tags(expected.arg1, 'arg', [TestTag, TestTag2])
      fdl.set_tags(expected.arg1, 'kwarg', [TestTag, TestTag2])
      self.assertDagEqual(cfg, expected)

    with self.subTest('outer_config.as_buildable'):
      cfg = outer_config.as_buildable()
      expected = fdl.Config(SampleClass, fdl.Config(basic_fn, 1, 2), 3)
      fdl.set_tags(expected, 'arg1', [])
      fdl.set_tags(expected, 'arg2', [TestTag])
      fdl.set_tags(expected.arg1, 'arg', [TestTag, TestTag2])
      fdl.set_tags(expected.arg1, 'kwarg', [TestTag2])
      self.assertDagEqual(cfg, expected)

  def test_debug_mode(self):

    def sort_list(x):  # Function with a side-effect.
      x.sort()

    @auto_config.auto_config
    def fn():
      lst = [3, 2, 1]
      sort_list(lst)  # Function only gets exectued in debug mode.
      result = SampleClass(lst, None)
      return result

    with self.subTest('debug_mode'):
      with auto_config.debug_auto_config():
        self.assertEqual(fn(), SampleClass([1, 2, 3], None))

    with self.subTest('dunder_call'):
      self.assertEqual(fn(), SampleClass([3, 2, 1], None))

    with self.subTest('as_buildable'):
      self.assertDagEqual(fn.as_buildable(),
                          fdl.Config(SampleClass, [3, 2, 1], None))


if __name__ == '__main__':
  absltest.main()
