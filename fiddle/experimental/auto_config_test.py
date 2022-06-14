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
import sys
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized

from fiddle import building
from fiddle import config
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
    pass


def explicit_config_building_fn(x: int) -> config.Config:
  return config.Config(basic_fn, 5, kwarg=x)


def pass_through(arg):
  return arg


def _line_number():
  return sys._getframe(1).f_lineno


class AutoConfigTest(parameterized.TestCase):

  def test_create_basic_config(self):
    expected_config = config.Config(SampleClass, 1, arg2=2)

    @auto_config.auto_config
    def test_class_config():
      return SampleClass(1, 2)

    self.assertEqual(expected_config, test_class_config.as_buildable())
    self.assertEqual(SampleClass(1, 2), test_class_config())

  def test_create_basic_config_parents(self):
    expected_config = config.Config(SampleClass, 1, arg2=2)

    @auto_config.auto_config()  # Note the parenthesis!
    def test_class_config():
      return SampleClass(1, 2)

    self.assertEqual(expected_config, test_class_config.as_buildable())
    self.assertEqual(SampleClass(1, 2), test_class_config())

  def test_create_basic_partial(self):
    expected_config = config.Partial(basic_fn, 1, kwarg='kwarg')

    @auto_config.auto_config
    def test_fn_config():
      return functools.partial(basic_fn, 1, 'kwarg')

    self.assertEqual(expected_config, test_fn_config.as_buildable())

  def test_create_config_with_args(self):
    expected_config = config.Config(SampleClass, 'positional', arg2='default')

    @auto_config.auto_config
    def test_class_config(arg1, arg2='default'):
      return SampleClass(arg1, arg2)

    self.assertEqual(expected_config,
                     test_class_config.as_buildable('positional'))
    self.assertEqual(
        SampleClass('positional', 'default'), test_class_config('positional'))

  def test_create_config_with_kwonly_args(self):
    expected_config = config.Config(SampleClass, 'positional', arg2='default')

    @auto_config.auto_config
    def test_class_config(arg1, *, arg2='default'):
      return SampleClass(arg1, arg2)

    self.assertEqual(expected_config,
                     test_class_config.as_buildable('positional'))
    self.assertEqual(
        SampleClass('positional', 'default'), test_class_config('positional'))

  def test_calling_auto_config(self):
    expected_config = config.Config(
        basic_fn, 1, kwarg=config.Config(SampleClass, 1, 2))

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
    expected_config = config.Config(
        SampleClass, 1, arg2=config.Config(basic_fn, 2, 'kwarg'))

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
    expected_config = config.Config(
        SampleClass, 1, arg2=config.Config(basic_fn, 5, 10))

    @auto_config.auto_config
    def test_nested_call():
      return SampleClass(1, explicit_config_building_fn(10))

    self.assertEqual(expected_config, test_nested_call.as_buildable())

  def test_reference_nonlocal(self):
    nonlocal_var = 3
    expected_config = config.Config(basic_fn, 1, kwarg=nonlocal_var)

    @auto_config.auto_config
    def test_fn_config():
      return basic_fn(1, nonlocal_var)

    self.assertEqual(expected_config, test_fn_config.as_buildable())

  def test_calling_builtins(self):
    expected_config = config.Config(SampleClass, [0, 1, 2], ['a', 'b'])

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
    self.assertFalse(auto_config._is_auto_config_eligible(config.Config))
    self.assertFalse(auto_config._is_auto_config_eligible(config.Partial))
    # Auto-config annotations are not eligible, and this case is tested in
    # `test_calling_auto_config` above.

  def test_autobuilders_in_auto_config(self):
    expected_config = config.Config(
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
        'test_key1': config.Config(SampleClass, 1, arg2=2),
        'test_key2': [config.Config(SampleClass, 3, 4), 5],
        'test_key3': (config.Partial(pass_through, 'arg'), 6),
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
        config.Config(SampleClass, 2, 1), MyClass.my_fn.as_buildable())
    self.assertEqual(SampleClass(2, 1), MyClass.my_fn())

    instance = MyClass()
    self.assertEqual(
        config.Config(SampleClass, 2, 1), instance.my_fn.as_buildable())
    self.assertEqual(SampleClass(2, 1), instance.my_fn())

  def test_staticmethod_arguments(self):

    class MyClass:

      @staticmethod
      @auto_config.auto_config
      def my_fn(x):
        return SampleClass(x, x + 1)

    self.assertEqual(
        config.Config(SampleClass, 5, 6), MyClass.my_fn.as_buildable(5))
    self.assertEqual(SampleClass(5, 6), MyClass.my_fn(5))

    instance = MyClass()

    self.assertEqual(
        config.Config(SampleClass, 5, 6), instance.my_fn.as_buildable(5))
    self.assertEqual(SampleClass(5, 6), instance.my_fn(5))

  def test_staticmethod_not_on_top(self):
    with self.assertRaisesRegex(TypeError, 'Please order the decorators'):

      class MyClass:  # pylint: disable=unused-variable

        @auto_config.auto_config
        @staticmethod
        def my_fn(x, y):
          return SampleClass(x, y)

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
    self.assertEqual('true branch', building.build(build_config_fn(True)))
    self.assertEqual('false branch', building.build(build_config_fn(False)))

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

    expected_config = config.Config(
        pass_through, [config.Config(SampleClass, i, i) for i in range(3)])
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
    self.assertEqual(0, building.build(pass_through_config))

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
    self.assertCountEqual([1, 2, 3], building.build(pass_through_config))

  def test_allow_control_flow_decorator(self):

    @auto_config.auto_config(experimental_allow_control_flow=True)
    def test_config():
      return pass_through([i + 1 for i in range(3)])

    self.assertEqual([1, 2, 3], building.build(test_config.as_buildable()))

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


if __name__ == '__main__':
  absltest.main()
