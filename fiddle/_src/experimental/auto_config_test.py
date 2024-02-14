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

import ast
import copy
import dataclasses
import functools
import inspect
import sys
from typing import Any, TypeVar

from absl.testing import absltest
from absl.testing import parameterized
import fiddle as fdl
from fiddle import arg_factory
from fiddle._src.experimental import auto_config
from fiddle._src.experimental import auto_config_policy
from fiddle._src.experimental import autobuilders as ab
from fiddle._src.testing import test_util
import pytype_extensions


def basic_fn(arg, kwarg='test'):
  return {'arg': arg, 'kwarg': kwarg}


def fn_with_kwargs(**kwargs):
  return {'kwargs': kwargs}


class NonDataclassSampleClass:

  def __init__(self, arg1, arg2):
    self.arg1 = arg1
    self.arg2 = arg2


@dataclasses.dataclass(frozen=False)
class MutableSampleClass:
  arg1: Any
  arg2: Any


@dataclasses.dataclass(frozen=True)
class FrozenSampleClass:
  arg1: Any
  arg2: Any

  def method(self):
    return 42

  @auto_config.auto_config
  def autoconfig_method(self):
    return basic_fn(self)


@auto_config.auto_unconfig(experimental_always_inline=True)
def explicit_config_building_fn(x: int) -> fdl.Config:
  return fdl.Config(basic_fn, 5, kwarg=x)


@auto_config.auto_config
def globals_test_fn():
  """Function to test that auto_config globals act correctly.

  In particular, we're testing that symbols that are added to `globals` *after*
  the auto_config decorator runs are visible when we call `as_buildable`.  For
  that reason, this function definition must be placed before the definition for
  `pass_through`.

  Returns:
    5
  """
  return pass_through(5)


T = TypeVar('T')


def pass_through(arg: T) -> T:
  return arg


def _line_number():
  return sys._getframe(1).f_lineno


class AutoConfigTransformerTest(parameterized.TestCase, test_util.TestCase):

  @parameterized.parameters(
      (
          'a.b = x, y',
          "__auto_config_attr_save_handler__(a, 'b', (x, y))",
      ),
      (
          'x.y.z = v',
          (
              '__auto_config_attr_save_handler__('
              "__auto_config_attr_load_handler__(x, 'y'), 'z', v)"
          ),
      ),
      (
          'a.b.c = c.d',
          (
              '__auto_config_attr_save_handler__('
              "__auto_config_attr_load_handler__(a, 'b'), 'c', "
              "__auto_config_attr_load_handler__(c, 'd'))"
          ),
      ),
      (
          'a, b.c = v',
          (
              '(a, _attr_save_temp_0) = v\n'
              "__auto_config_attr_save_handler__(b, 'c', _attr_save_temp_0)"
          ),
          (
              'a, _attr_save_temp_0 = v\n'
              "__auto_config_attr_save_handler__(b, 'c', _attr_save_temp_0)"
          ),
      ),
      (
          'a = b = c',
          'a = b = c',
      ),
      (
          'a, b.c = x.y.z = foo.bar',
          (
              '(a, _attr_save_temp_0) = _attr_save_temp_1 = '
              "__auto_config_attr_load_handler__(foo, 'bar')\n"
              "__auto_config_attr_save_handler__(b, 'c', _attr_save_temp_0)\n"
              '__auto_config_attr_save_handler__('
              "__auto_config_attr_load_handler__(x, 'y'), "
              "'z', _attr_save_temp_1)"
          ),
          (
              'a, _attr_save_temp_0 = _attr_save_temp_1 = '
              "__auto_config_attr_load_handler__(foo, 'bar')\n"
              "__auto_config_attr_save_handler__(b, 'c', _attr_save_temp_0)\n"
              '__auto_config_attr_save_handler__('
              "__auto_config_attr_load_handler__(x, 'y'), "
              "'z', _attr_save_temp_1)"
          ),
      ),
      # Test transformations when Assign.value is a function call.
      # This test will ensure calls to ``functools.partial`` are replaced by
      # ``fdl.Partial`` properly.
      (
          'a.b.c = functools.partial(foo, x.y)',
          (
              '__auto_config_attr_save_handler__('
              "__auto_config_attr_load_handler__(a, 'b'), 'c', "
              '__auto_config_call_handler__(functools.partial, foo, '
              "__auto_config_attr_load_handler__(x, 'y')))"
          ),
      ),
      (
          'a.b.c = bar = functools.partial(foo, x.y)',
          (
              '_attr_save_temp_0 = bar = __auto_config_call_handler__('
              'functools.partial, foo, __auto_config_attr_load_handler__('
              "x, 'y'))\n"
              '__auto_config_attr_save_handler__('
              "__auto_config_attr_load_handler__(a, 'b'), 'c', "
              '_attr_save_temp_0)'
          ),
      ),
      ('a.b[1] = 123', "__auto_config_attr_load_handler__(a, 'b')[1] = 123"),
      (
          # TODO(b/288479702): Add validation for ast.Starred pattern.
          'a, *a.b = foo.bar',
          "(a, *a.b) = __auto_config_attr_load_handler__(foo, 'bar')",
          "a, *a.b = __auto_config_attr_load_handler__(foo, 'bar')",
      ),
  )
  def test_attribute_save(self, source, expected, py11_expected=None):
    version = sys.version_info
    # ast.unparse is available only after Python 3.9
    if version.major >= 3 and version.minor >= 9:
      transfomer = auto_config._AutoConfigNodeTransformer(source, 'dummy.py', 0)
      node = ast.parse(source)
      node = transfomer.visit(node)
      # `ast.unparse` has different behaviors since Python 3.11, for example,
      # ast.unparse(ast.parse('a, b = c'))
      # Before Python 3.11, it returns: '(a, b) = c'
      # From Python 3.11, it returns: 'a, b = c'
      if version.minor >= 11 and py11_expected is not None:
        self.assertEqual(ast.unparse(node), py11_expected)
      else:
        self.assertEqual(ast.unparse(node), expected)


class AutoConfigTest(parameterized.TestCase, test_util.TestCase):

  def test_create_basic_config(self):
    expected_config = fdl.Config(FrozenSampleClass, 1, arg2=2)

    @auto_config.auto_config
    def test_class_config():
      return FrozenSampleClass(1, 2)

    self.assertEqual(expected_config, test_class_config.as_buildable())
    self.assertEqual(FrozenSampleClass(1, 2), test_class_config())

  def test_create_basic_config_parents(self):
    expected_config = fdl.Config(FrozenSampleClass, 1, arg2=2)

    @auto_config.auto_config()  # Note the parenthesis!
    def test_class_config():
      return FrozenSampleClass(1, 2)

    self.assertEqual(expected_config, test_class_config.as_buildable())
    self.assertEqual(FrozenSampleClass(1, 2), test_class_config())

  def test_create_basic_partial(self):
    expected_config = fdl.Partial(basic_fn, 1, kwarg='kwarg')

    @auto_config.auto_config
    def test_fn_config():
      return functools.partial(basic_fn, 1, 'kwarg')

    self.assertEqual(expected_config, test_fn_config.as_buildable())

  def test_nested_functools_partial(self):
    expected_config = fdl.Partial(fn_with_kwargs, a=1, b=2)

    @auto_config.auto_config
    def test_fn_config():
      return functools.partial(functools.partial(fn_with_kwargs, a=1), b=2)

    self.assertEqual(expected_config, test_fn_config.as_buildable())
    self.assertEqual(test_fn_config()(), {'kwargs': {'a': 1, 'b': 2}})
    self.assertEqual(
        fdl.build(test_fn_config.as_buildable())(), {'kwargs': {'a': 1, 'b': 2}}
    )

  def test_nested_functools_partial_unsupported_positional_args(self):
    @auto_config.auto_config
    def test_fn_config():
      return functools.partial(functools.partial(FrozenSampleClass, 1), 2)

    with self.assertRaisesRegex(
        ValueError, 'chained.*partial.*calls.*only.*keyword'
    ):
      test_fn_config.as_buildable()

  def test_create_basic_arg_factory(self):
    expected_config = fdl.Partial(
        basic_fn, fdl.ArgFactory(fn_with_kwargs, a='b')
    )

    @auto_config.auto_config
    def test_fn_config():
      return arg_factory.partial(
          basic_fn, functools.partial(fn_with_kwargs, a='b')
      )

    self.assertEqual(expected_config, test_fn_config.as_buildable())

  def test_arg_factory_counter(self):
    counter = {'count': 0}

    def count():
      counter['count'] += 1
      return counter['count']

    @auto_config.auto_config
    def test_fn_config():
      return arg_factory.partial(basic_fn, count)

    py_callable = test_fn_config()
    self.assertEqual(py_callable(), {'arg': 1, 'kwarg': 'test'})
    self.assertEqual(py_callable(), {'arg': 2, 'kwarg': 'test'})
    self.assertEqual(py_callable(), {'arg': 3, 'kwarg': 'test'})
    built_from_config = fdl.build(test_fn_config.as_buildable())
    self.assertEqual(built_from_config(), {'arg': 4, 'kwarg': 'test'})
    self.assertEqual(built_from_config(), {'arg': 5, 'kwarg': 'test'})
    self.assertEqual(built_from_config(), {'arg': 6, 'kwarg': 'test'})

  def test_arg_factory_functools_partial_alternation(self):
    expected_config = fdl.Partial(
        fn_with_kwargs, a=fdl.Config(basic_fn, 1), b=fdl.ArgFactory(basic_fn, 2)
    )

    @auto_config.auto_config
    def test_fn_config_1():
      return arg_factory.partial(
          functools.partial(
              fn_with_kwargs,
              a=basic_fn(1),
          ),
          b=functools.partial(basic_fn, 2),
      )

    @auto_config.auto_config
    def test_fn_config_2():
      return functools.partial(
          arg_factory.partial(
              fn_with_kwargs,
              b=functools.partial(basic_fn, 2),
          ),
          a=basic_fn(1),
      )

    self.assertEqual(expected_config, test_fn_config_1.as_buildable())
    self.assertEqual(expected_config, test_fn_config_2.as_buildable())

  def test_create_config_with_args(self):
    expected_config = fdl.Config(
        FrozenSampleClass, 'positional', arg2='default'
    )

    @auto_config.auto_config
    def test_class_config(arg1, arg2='default'):
      return FrozenSampleClass(arg1, arg2)

    self.assertEqual(expected_config,
                     test_class_config.as_buildable('positional'))
    self.assertEqual(
        FrozenSampleClass('positional', 'default'),
        test_class_config('positional'),
    )

  def test_create_config_with_kwonly_args(self):
    expected_config = fdl.Config(
        FrozenSampleClass, 'positional', arg2='default'
    )

    @auto_config.auto_config
    def test_class_config(arg1, *, arg2='default'):
      return FrozenSampleClass(arg1, arg2)

    self.assertEqual(expected_config,
                     test_class_config.as_buildable('positional'))
    self.assertEqual(
        FrozenSampleClass('positional', 'default'),
        test_class_config('positional'),
    )

  def test_calling_auto_config(self):
    expected_config = fdl.Config(
        basic_fn, 1, kwarg=fdl.Config(FrozenSampleClass, 1, 2)
    )

    @auto_config.auto_config(experimental_always_inline=True)
    def test_class_config():
      return FrozenSampleClass(1, arg2=2)

    @auto_config.auto_config
    def test_fn_config():
      return basic_fn(1, test_class_config())

    self.assertEqual(expected_config, test_fn_config.as_buildable())
    self.assertEqual(
        {'arg': 1, 'kwarg': FrozenSampleClass(1, arg2=2)}, test_fn_config()
    )

  def test_nested_calls(self):
    expected_config = fdl.Config(
        FrozenSampleClass, 1, arg2=fdl.Config(basic_fn, 2, 'kwarg')
    )

    @auto_config.auto_config
    def test_class_config():
      return FrozenSampleClass(1, basic_fn(2, 'kwarg'))

    self.assertEqual(expected_config, test_class_config.as_buildable())
    self.assertEqual(
        FrozenSampleClass(1, {'arg': 2, 'kwarg': 'kwarg'}), test_class_config()
    )

  def test_calling_explicit_function(self):
    expected_config = fdl.Config(
        FrozenSampleClass, 1, arg2=fdl.Config(basic_fn, 5, 10)
    )

    @auto_config.auto_config
    def test_nested_call():
      return FrozenSampleClass(1, explicit_config_building_fn(10))

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
    expected_config = fdl.Config(FrozenSampleClass, [0, 1, 2], ['a', 'b'])

    @auto_config.auto_config
    def test_config():
      return FrozenSampleClass(list(range(3)), list({'a': 0, 'b': 1}.keys()))

    self.assertEqual(expected_config, test_config.as_buildable())

  def test_auto_config_eligibility(self):
    # Some common types that have no signature.
    for builtin_type in (range, dict):
      self.assertTrue(auto_config_policy.v1(builtin_type))
    # Some common builtins.
    for builtin in (list, tuple, sum, any, all, iter):
      self.assertTrue(auto_config_policy.v1(builtin))
    self.assertTrue(auto_config_policy.v1([].append))
    self.assertTrue(auto_config_policy.v1({}.keys))
    # A method.
    test_class = FrozenSampleClass(1, 2)
    self.assertFalse(auto_config_policy.v1(test_class.method))
    # Buildable subclasses.
    self.assertTrue(auto_config_policy.v1(fdl.Config))
    self.assertTrue(auto_config_policy.v1(fdl.Partial))
    # Auto-config annotations are not eligible, and this case is tested in
    # `test_calling_auto_config` above.

  def test_autobuilders_in_auto_config(self):
    expected_config = fdl.Config(
        basic_fn, arg=ab.config(FrozenSampleClass, require_skeleton=False)
    )

    @auto_config.auto_config
    def autobuilder_using_fn():
      x = ab.config(FrozenSampleClass, require_skeleton=False)
      return basic_fn(x)

    self.assertEqual(expected_config, autobuilder_using_fn.as_buildable())

  def test_auto_configuring_non_function(self):
    with self.assertRaisesRegex(ValueError, 'only compatible with functions'):
      auto_config.auto_config(3)

  def test_return_structure(self):
    expected_config = {
        'test_key1': fdl.Config(FrozenSampleClass, 1, arg2=2),
        'test_key2': [fdl.Config(FrozenSampleClass, 3, 4), 5],
        'test_key3': (fdl.Partial(pass_through, 'arg'), 6),
    }

    @auto_config.auto_config
    def test_config():
      return {
          'test_key1': FrozenSampleClass(1, 2),
          'test_key2': [FrozenSampleClass(3, 4), 5],
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

  def test_experimental_config_types(self):
    class CustomConfig(fdl.Config):
      pass

    class CustomPartial(fdl.Partial):
      pass

    class CustomArgFactory(fdl.ArgFactory):
      pass

    config_types = auto_config.ConfigTypes(  # pytype: disable=wrong-arg-types  # use-fiddle-overlay
        config_cls=CustomConfig,
        partial_cls=CustomPartial,
        arg_factory_cls=CustomArgFactory,
    )

    def make_int() -> int:
      return 42

    @auto_config.auto_config(experimental_config_types=config_types)
    def fn():
      return FrozenSampleClass(
          arg1=functools.partial(basic_fn, make_int()),
          arg2=arg_factory.partial(basic_fn, make_int),
      )

    expected = CustomConfig(
        FrozenSampleClass,
        arg1=CustomPartial(basic_fn, CustomConfig(make_int)),
        arg2=CustomPartial(basic_fn, CustomArgFactory(make_int)),
    )
    cfg = fn.as_buildable()
    # This assertion ensures that the correct types were produced (in addition
    # to the leaf/parameter values in the config).
    self.assertDagEqual(cfg, expected)

  def test_staticmethod_nullary(self):

    class MyClass:

      @auto_config.auto_config
      @staticmethod
      def my_fn():
        return FrozenSampleClass(2, 1)

    self.assertEqual(
        fdl.Config(FrozenSampleClass, 2, 1), MyClass.my_fn.as_buildable()
    )
    self.assertEqual(FrozenSampleClass(2, 1), MyClass.my_fn())

    instance = MyClass()
    self.assertEqual(
        fdl.Config(FrozenSampleClass, 2, 1), instance.my_fn.as_buildable()
    )
    self.assertEqual(FrozenSampleClass(2, 1), instance.my_fn())

  def test_staticmethod_arguments(self):

    class MyClass:

      @auto_config.auto_config
      @staticmethod
      def my_fn(x):
        return FrozenSampleClass(x, x + 1)

    self.assertEqual(
        fdl.Config(FrozenSampleClass, 5, 6), MyClass.my_fn.as_buildable(5)
    )
    self.assertEqual(FrozenSampleClass(5, 6), MyClass.my_fn(5))

    instance = MyClass()

    self.assertEqual(
        fdl.Config(FrozenSampleClass, 5, 6), instance.my_fn.as_buildable(5)
    )
    self.assertEqual(FrozenSampleClass(5, 6), instance.my_fn(5))

  def test_staticmethod_on_top(self):

    expected_msg = (
        r'@staticmethod placed above @auto_config on function my_fn at '
        r'.+:\d+\. Reorder decorators so that @auto_config is placed above '
        r'@staticmethod.')
    with self.assertRaisesRegex(AssertionError, expected_msg):

      class MyClass:  # pylint: disable=unused-variable

        @staticmethod
        @auto_config.auto_config
        def my_fn(x, y):
          pass

  def test_staticmethod_ctor(self):

    @dataclasses.dataclass(frozen=True)
    class ClassWithStaticmethodCtor:
      a: int
      b: float

      @auto_config.auto_config
      @staticmethod
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

      @auto_config.auto_config
      @classmethod
      def simple(cls):
        """Test simple docstring."""
        return cls(x=1, y='1', z=1.0)

    class MySubclass(MyClass):
      pass

    cfg = MyClass.simple.as_buildable()
    self.assertEqual(MyClass, fdl.get_callable(cfg))
    self.assertEqual(1, cfg.x)
    self.assertEqual('1', cfg.y)
    self.assertEqual(1.0, cfg.z)

    class_instance = MyClass(1, '2')
    cfg = class_instance.simple.as_buildable()
    self.assertEqual(MyClass, fdl.get_callable(cfg))
    self.assertEqual(1, cfg.x)
    self.assertEqual('1', cfg.y)
    self.assertEqual(1.0, cfg.z)

    self.assertEqual(MyClass.simple.__name__, 'simple')
    self.assertEqual(MyClass.simple.__qualname__,
                     'AutoConfigTest.test_classmethod.<locals>.MyClass.simple')
    self.assertEqual(MyClass.simple.__module__, __name__)
    self.assertEqual(MyClass.simple.__doc__, 'Test simple docstring.')

    # The attributes on the subclass should be the same, except for
    # __qualname__.
    self.assertEqual(MySubclass.simple.__name__, 'simple')
    self.assertEqual(MySubclass.simple.__qualname__,
                     'AutoConfigTest.test_classmethod.<locals>.MyClass.simple')
    self.assertEqual(MySubclass.simple.__module__, __name__)
    self.assertEqual(MySubclass.simple.__doc__, 'Test simple docstring.')

  def test_classmethod_on_top(self):

    expected_msg = (
        r'@classmethod placed above @auto_config on function my_fn at .+:\d+\. '
        r'Reorder decorators so that @auto_config is placed above @classmethod.'
    )
    with self.assertRaisesRegex(AssertionError, expected_msg):

      class MyClass:  # pylint: disable=unused-variable

        @classmethod
        @auto_config.auto_config
        def my_fn(cls, x, y):
          pass

  def test_classmethod_on_top_with_auto_config_arguments(self):

    expected_msg = (
        r'@classmethod placed above @auto_config on function my_fn at .+:\d+\. '
        r'Reorder decorators so that @auto_config is placed above @classmethod.'
    )
    with self.assertRaisesRegex(AssertionError, expected_msg):

      class MyClass:  # pylint: disable=unused-variable

        @classmethod
        @auto_config.auto_config(experimental_allow_control_flow=True)
        def my_fn(cls, x, y):
          pass

  def test_copy(self):

    @auto_config.auto_config
    def test_fn():
      return FrozenSampleClass(1, 2)

    with self.subTest('shallow_copy'):
      shallow_copy = copy.copy(test_fn)
      self.assertIsNot(shallow_copy, test_fn)
      self.assertEqual(shallow_copy.as_buildable(), test_fn.as_buildable())

    with self.subTest('deep_copy'):
      deep_copy = copy.deepcopy(test_fn)
      self.assertIsNot(deep_copy, test_fn)
      # We don't actually have any additional subfields that should be
      # deepcopied to test, but this at least makes sure things run.
      self.assertEqual(deep_copy.as_buildable(), test_fn.as_buildable())

  def test_copy_classmethod(self):

    @dataclasses.dataclass(frozen=True)
    class SomeClass:
      a: int
      b: int

      @auto_config.auto_config
      @classmethod
      def test_method(cls):
        return cls(1, 2)

    with self.subTest('shallow_copy'):
      shallow_copy = copy.copy(SomeClass.test_method)
      self.assertIsNot(shallow_copy, SomeClass.test_method)
      self.assertEqual(shallow_copy.as_buildable(),
                       SomeClass.test_method.as_buildable())

    with self.subTest('deep_copy'):
      deep_copy = copy.deepcopy(SomeClass.test_method)
      self.assertIsNot(deep_copy, SomeClass.test_method)
      # We don't actually have any additional subfields that should be
      # deepcopied to test, but this at least makes sure things run.
      self.assertEqual(deep_copy.as_buildable(),
                       SomeClass.test_method.as_buildable())

  def test_call_super(self):

    class MyClass(FrozenSampleClass):  # pylint: disable=unused-variable

      def __init__(self):
        super().__init__(1, 2)

      @auto_config.auto_config
      def my_fn(self, x):
        return basic_fn(x, kwarg=super().method())

    instance = MyClass()
    cfg = instance.my_fn.as_buildable('a')
    self.assertEqual(fdl.get_callable(cfg), basic_fn)
    self.assertEqual(cfg.arg, 'a')
    self.assertEqual(cfg.kwarg, fdl.Config(super(MyClass, instance).method))  # pytype: disable=invalid-annotation  # use-fiddle-overlay

  def test_custom_policy(self):

    def make_list(max_value):
      return list(range(max_value))

    def custom_policy(fn):
      if fn is make_list:
        return True
      return auto_config_policy.v1(fn)

    def make_sample(max_value):
      return FrozenSampleClass(arg1=make_list(max_value), arg2=None)

    with_policy = auto_config.auto_config(
        experimental_exemption_policy=custom_policy)(
            make_sample)
    without_policy = auto_config.auto_config(make_sample)

    expected_without_policy = fdl.Config(
        FrozenSampleClass, fdl.Config(make_list, 5), None
    )
    expected_with_policy = fdl.Config(FrozenSampleClass, list(range(5)), None)

    self.assertEqual(with_policy.as_buildable(5), expected_with_policy)
    self.assertEqual(without_policy.as_buildable(5), expected_without_policy)

  def test_inline_exemption(self):
    @auto_config.auto_config
    def make_sample(max_value):
      # Test the case where we first create the function, then call.
      # It produces different AST.
      exempted_func = auto_config.exempt(pass_through)
      return FrozenSampleClass(
          arg1=auto_config.exempt(pass_through)(max_value),
          arg2=FrozenSampleClass(
              arg1=pass_through(max_value), arg2=exempted_func(max_value)
          ),
      )

    cfg = make_sample.as_buildable(5)
    self.assertEqual(cfg.arg1, 5)
    self.assertEqual(cfg.arg2.arg1, fdl.Config(pass_through, 5))
    self.assertEqual(cfg.arg2.arg2, 5)

  def test_exemption_type_inference(self):
    exempted_func = auto_config.exempt(pass_through)
    value = exempted_func(42)
    pytype_extensions.assert_type(value, int)

  def test_lambda_supported_in_decorator(self):
    @auto_config.auto_config(experimental_exemption_policy=lambda x: False)
    def make_sample():
      return dict(arg1=3)

    expected = fdl.Config(dict, arg1=3)
    self.assertEqual(make_sample.as_buildable(), expected)

  def test_function_metadata(self):

    @auto_config.auto_config
    def my_helpful_function(x: int, y: str = 'y') -> FrozenSampleClass:
      """A docstring."""
      return FrozenSampleClass(x, y)

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
        expected_parameters, return_annotation=FrozenSampleClass
    )
    self.assertEqual(expected_signature, inspect.signature(my_helpful_function))
    self.assertEqual('my_helpful_function', my_helpful_function.__name__)

  def test_unhashable_callable(self):
    class UnhashableCallable:
      a: int = 1

      # Class with __eq__ but not __hash__ is unhashable.
      def __eq__(self, other):
        return self is other

      def __call__(self, arg1: int) -> int:
        return arg1

    obj = UnhashableCallable()

    with self.assertRaisesRegex(TypeError, 'unhashable type'):
      _ = {obj: 1}

    @auto_config.auto_config
    def make_sample(obj):
      return obj(1)

    cfg = make_sample.as_buildable(obj)
    self.assertEqual(cfg, fdl.Config(obj, arg1=1))  # pytype: disable=invalid-annotation  # use-fiddle-overlay

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
        layers.append(FrozenSampleClass(i, i))
      return pass_through(layers)

    with self.assertRaisesRegex(
        auto_config.UnsupportedLanguageConstructError,
        r'Control flow \(For\) is unsupported by auto_config\.'):
      auto_config.auto_config(test_config)

    expected_config = fdl.Config(
        pass_through, [fdl.Config(FrozenSampleClass, i, i) for i in range(3)]
    )
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
      raise ValueError('Some message.')

    with self.assertRaisesRegex(
        auto_config.UnsupportedLanguageConstructError,
        r'Control flow \(Raise\) is unsupported by auto_config\.'):
      auto_config.auto_config(test_config)

    auto_config.auto_config(test_config, experimental_allow_control_flow=True)

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
      return pass_through(lambda: FrozenSampleClass(1, 2))

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

  def test_access_autoconfig_method_via_class(self):
    # If we access an autoconfig via an instance, we get a bound method:
    self.assertIsInstance(
        FrozenSampleClass(1, 2).autoconfig_method, auto_config.AutoConfig
    )

    # But if we access it via the class, we get the (unbound) AutoConfig:
    self.assertIsInstance(
        FrozenSampleClass.autoconfig_method, auto_config.AutoConfig
    )

  @absltest.skip('Enable this after dropping pyhon 3.7 support')
  def test_can_pass_self_as_keyword(self):
    x = FrozenSampleClass(1, 2)
    self.assertEqual(
        FrozenSampleClass.autoconfig_method(self=x), {'arg': x, 'kwarg': 'test'}
    )
    self.assertDagEqual(
        FrozenSampleClass.autoconfig_method.as_buildable(self=x),
        fdl.Config(basic_fn, x),
    )

  def test_globals(self):
    # The following will fail with `NameError: name 'pass_through' is not
    # defined` if the as_buildable globals don't reflect changes made after
    # the auto_config decorator is run.
    globals_test_fn.as_buildable()

  def test_trailing_paren_after_lambda(self):
    # Test that we can handle a lambda that has an extra trailing paren on
    # the line that contains the lambda expression.
    x = (  # do not re-format this expression.
        lambda: basic_fn(5))  # pyformat: disable
    x = auto_config.auto_config(x)
    self.assertDagEqual(x.as_buildable(), fdl.Config(basic_fn, 5))

  def test_multiple_lambdas_on_one_line(self):
    # TODO(b/258671226) Enable this test if/when we add support for this.
    self.skipTest('multiple lambdas on one line not supported yet')
    x = (lambda a: basic_fn(2 * a), lambda b: basic_fn(3 * b))
    x0 = auto_config.auto_config(x[0])
    x1 = auto_config.auto_config(x[1])
    self.assertDagEqual(x0.as_buildable(10), fdl.Config(basic_fn, 20))
    self.assertDagEqual(x1.as_buildable(10), fdl.Config(basic_fn, 30))

  def test_experimental_result_must_contain_buildable(self):
    @auto_config.auto_config(
        experimental_result_must_contain_buildable=False,
        experimental_allow_control_flow=True,
    )
    def make_dict(keys):
      # This function's result may or may not contain any buildable, depending
      # on the value of `keys`.
      return {key: basic_fn(key) for key in keys}

    cfg = make_dict.as_buildable(['foo'])
    self.assertEqual(fdl.build(cfg), {'foo': basic_fn('foo')})

    cfg = make_dict.as_buildable([])
    self.assertEqual(fdl.build(cfg), {})

  def test_non_dataclass_attribute_load(self):
    @auto_config.auto_config
    def fixture():
      foo = NonDataclassSampleClass(1, 2)
      _ = foo.arg1
      return foo

    with self.assertRaisesRegex(ValueError, 'Cannot load attribute'):
      fixture.as_buildable()

  def test_dataclass_attribute_load(self):
    def fixture():
      foo = MutableSampleClass(1, 2)
      _ = foo.arg1
      return foo

    fixture_disable = auto_config.auto_config(
        fixture, experimental_allow_dataclass_attribute_access=False
    )
    fixture_enable = auto_config.auto_config(
        fixture, experimental_allow_dataclass_attribute_access=True
    )

    with self.subTest('disable'):
      with self.assertRaisesRegex(ValueError, 'Cannot load attribute'):
        fixture_disable.as_buildable()

    with self.subTest('enable'):
      fixture_enable.as_buildable()

  def test_nested_attribute_load(self):
    @auto_config.auto_config
    def fixture():
      foo = NonDataclassSampleClass(1, 'ABC')
      # To test if inner attribute calls are validated. foo.arg2 is not a
      # Buildable but foo is. So if the attribute validation only inpsects
      # the outermost attribute access, the error will not be raised.
      lower = foo.arg2.lower
      return NonDataclassSampleClass(2, lower)

    with self.subTest('disable'):
      with self.assertRaisesRegex(ValueError, 'Cannot load attribute'):
        fixture.as_buildable()

  def test_non_dataclass_attribute_save(self):
    @auto_config.auto_config
    def fixture():
      foo = NonDataclassSampleClass(1, 2)
      foo.arg1 = 2
      return foo

    with self.assertRaisesRegex(ValueError, 'Cannot save attribute'):
      fixture.as_buildable()

  def test_dataclass_attribute_save(self):
    def fixture():
      foo = MutableSampleClass(1, 2)
      foo.arg1 = 3
      return foo

    fixture_disable = auto_config.auto_config(
        fixture, experimental_allow_dataclass_attribute_access=False
    )
    fixture_enable = auto_config.auto_config(
        fixture, experimental_allow_dataclass_attribute_access=True
    )

    with self.subTest('disable'):
      with self.assertRaisesRegex(ValueError, 'Cannot save attribute'):
        fixture_disable.as_buildable()

    with self.subTest('enable'):
      config = fixture_enable.as_buildable()
      self.assertEqual(config.arg1, 3)
      self.assertEqual(config.arg2, 2)

  def test_nested_attribute_save(self):
    @auto_config.auto_config(experimental_allow_dataclass_attribute_access=True)
    def fixture():
      foo = NonDataclassSampleClass(0, 'foo')
      bar = MutableSampleClass(foo, 'bar')
      bar.arg1.arg2 = 'baz'
      return NonDataclassSampleClass(0, bar)

    with self.assertRaisesRegex(ValueError, 'Cannot save attribute'):
      fixture.as_buildable()

  def test_nested_dataclass_attribute_save(self):
    @auto_config.auto_config(experimental_allow_dataclass_attribute_access=True)
    def fixture():
      foo = MutableSampleClass(0, 'foo')
      bar = MutableSampleClass(foo, 'bar')
      bar.arg1.arg2 = 'baz'
      return MutableSampleClass(0, bar)

    config = fixture.as_buildable()
    self.assertEqual(config.arg2.arg1.arg2, 'baz')


class AutoUnconfigTest(absltest.TestCase):

  def test_simple_inline(self):

    @auto_config.auto_unconfig(experimental_always_inline=True)
    def simple(x: int) -> fdl.Config[FrozenSampleClass]:
      cfg = fdl.Config(FrozenSampleClass)
      cfg.arg1 = x
      cfg.arg2 = str(x)
      return cfg

    @auto_config.auto_config
    def parent():
      return FrozenSampleClass(arg1=simple(42), arg2=5)

    expected = fdl.Config(
        FrozenSampleClass, arg1=fdl.Config(FrozenSampleClass, 42, '42'), arg2=5
    )

    self.assertEqual(expected, parent.as_buildable())

  def test_simple_not_inline(self):

    @auto_config.auto_unconfig(experimental_always_inline=False)
    def simple(x: int) -> fdl.Config[FrozenSampleClass]:
      cfg = fdl.Config(FrozenSampleClass)
      cfg.arg1 = x
      cfg.arg2 = str(x)
      return cfg

    @auto_config.auto_config
    def parent():
      return FrozenSampleClass(arg1=simple(42), arg2=5)

    expected = fdl.Config(
        FrozenSampleClass, arg1=fdl.Config(simple, 42), arg2=5
    )

    self.assertEqual(expected, parent.as_buildable())

  def test_arg_factory_counter(self):
    counter = {'count': 0}

    def count():
      counter['count'] += 1
      return counter['count']

    @auto_config.auto_unconfig
    def test_fn_config():
      return fdl.Partial(basic_fn, fdl.ArgFactory(count))

    py_callable = test_fn_config()
    self.assertEqual(py_callable(), {'arg': 1, 'kwarg': 'test'})
    self.assertEqual(py_callable(), {'arg': 2, 'kwarg': 'test'})
    self.assertEqual(py_callable(), {'arg': 3, 'kwarg': 'test'})
    built_from_config = fdl.build(test_fn_config.as_buildable())
    self.assertEqual(built_from_config(), {'arg': 4, 'kwarg': 'test'})
    self.assertEqual(built_from_config(), {'arg': 5, 'kwarg': 'test'})
    self.assertEqual(built_from_config(), {'arg': 6, 'kwarg': 'test'})

  def test_python_execution(self):

    @auto_config.auto_unconfig
    def simple(x: int) -> fdl.Config[FrozenSampleClass]:
      cfg = fdl.Config(FrozenSampleClass)
      cfg.arg1 = x
      cfg.arg2 = str(x)
      return cfg

    @auto_config.auto_config
    def parent():
      return FrozenSampleClass(arg1=simple(42), arg2=5)

    expected = FrozenSampleClass(arg1=FrozenSampleClass(42, '42'), arg2=5)

    self.assertEqual(expected, parent())

  def test_nested_python_execution(self):

    @auto_config.auto_unconfig
    def simple(x: int) -> fdl.Config[FrozenSampleClass]:
      cfg = fdl.Config(FrozenSampleClass)
      cfg.arg1 = x
      cfg.arg2 = str(x)
      return cfg

    def regular_python_wrapper(x) -> FrozenSampleClass:
      return simple(x - 1)

    @auto_config.auto_unconfig
    def parent() -> fdl.Config[FrozenSampleClass]:
      cfg = fdl.Config(FrozenSampleClass)
      cfg.arg1 = fdl.Config(regular_python_wrapper, 42)
      cfg.arg2 = 5
      return cfg

    expected = FrozenSampleClass(FrozenSampleClass(41, '41'), 5)
    self.assertEqual(expected, parent())


class InlineTest(test_util.TestCase):

  def test_simple(self):

    @auto_config.auto_config
    def my_fn(x: int, y: str):
      return FrozenSampleClass(basic_fn(x), y)

    cfg = fdl.Config(my_fn, x=4, y='y')
    orig = fdl.build(cfg)
    auto_config.inline(cfg)

    self.assertEqual(FrozenSampleClass, fdl.get_callable(cfg))
    self.assertEqual(cfg.arg2, 'y')
    self.assertIsInstance(cfg.arg1, fdl.Config)
    self.assertEqual(basic_fn, fdl.get_callable(cfg.arg1))
    self.assertEqual(4, cfg.arg1.arg)
    self.assertEqual(orig, fdl.build(cfg))

  def test_bound_auto_config(self):

    @dataclasses.dataclass
    class AutoConfigClass:
      x: Any
      y: Any

      @auto_config.auto_config
      @classmethod
      def default(cls, z):
        return cls(z, z + 2)

    cfg = fdl.Config(AutoConfigClass.default, z=5)
    orig = fdl.build(cfg)
    auto_config.inline(cfg)

    self.assertEqual(AutoConfigClass, fdl.get_callable(cfg))
    self.assertEqual(cfg.x, 5)
    self.assertEqual(cfg.y, 7)
    self.assertEqual(orig, fdl.build(cfg))

  def test_always_inline_default_is_true(self):

    @auto_config.auto_config
    def inlined_fn():
      return FrozenSampleClass(arg1=1, arg2=2)

    @auto_config.auto_config
    def calling_fn():
      return inlined_fn()

    expected_config = fdl.Config(FrozenSampleClass, arg1=1, arg2=2)
    actual_config = calling_fn.as_buildable()
    self.assertEqual(expected_config, actual_config)

  def test_inlining_nested_config(self):

    @auto_config.auto_config(experimental_always_inline=False)
    def make_library_type(x: int, y: str) -> FrozenSampleClass:
      """A function that forms a configuration API boundary.

      These functions are often vended by libraries.

      Args:
       x: A sample argument to be configured.
       y: An additional argument to be configured.

      Returns:
        An instance of the sample class configured per the policy encapsulated
        by this function.
      """
      return FrozenSampleClass(arg1=x + 1, arg2='Hello ' + y)

    @auto_config.auto_config
    def make_experiment():
      thing1 = basic_fn(arg=5)
      thing2 = make_library_type(5, 'world')
      return FrozenSampleClass(thing1, thing2)

    expected_config = fdl.Config(
        FrozenSampleClass,
        arg1=fdl.Config(basic_fn, arg=5),
        arg2=fdl.Config(make_library_type, 5, 'world'),
    )

    actual_config = make_experiment.as_buildable()
    self.assertEqual(expected_config, actual_config)

    auto_config.inline(actual_config.arg2)

    inline_expected_config = fdl.Config(
        FrozenSampleClass,
        arg1=fdl.Config(basic_fn, arg=5),
        arg2=fdl.Config(FrozenSampleClass, 6, 'Hello world'),
    )

    self.assertEqual(inline_expected_config, actual_config)

  # TODO(b/272374472): Investigate fixing this.
  @absltest.skip('Cannot inline a function that returns a partial currently.')
  def test_inlined_func_returns_partial(self):

    @auto_config.auto_config
    def my_fn(x: int):
      return functools.partial(FrozenSampleClass, arg1=x)

    cfg = fdl.Config(my_fn, x=2)
    auto_config.inline(cfg)
    expected = fdl.Partial(FrozenSampleClass, arg1=2)

    self.assertDagEqual(cfg, expected)
    self.assertEqual(fdl.build(cfg)(4), FrozenSampleClass(2, 4))

  @absltest.skip('Cannot inline a function that returns a non-buildable.')
  def test_inlined_func_returns_list(self):

    @auto_config.auto_config
    def my_fn(x: int):
      return [FrozenSampleClass(x, x), FrozenSampleClass(x + 1, x + 2)]

    cfg = fdl.Config(my_fn, x=2)
    auto_config.inline(cfg)


if __name__ == '__main__':
  absltest.main()
