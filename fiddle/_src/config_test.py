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

"""Tests for the `fiddle._src.config` module."""

import copy
import dataclasses
import pickle
import sys
import threading
from typing import Any, Dict, Generic, TypeVar
from absl.testing import absltest
from absl.testing import parameterized
import fiddle as fdl
from fiddle import daglish
from fiddle import history
from fiddle._src import config as config_lib
from fiddle._src.experimental import daglish_legacy
from fiddle._src.testing.example import demo_configs
import pytype_extensions


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


def basic_fn(arg1, arg2, kwarg1=None, kwarg2=None) -> Dict[str, Any]:  # pylint: disable=unused-argument
  return locals()


def fn_with_var_args(arg1, *args, kwarg1=None):  # pylint: disable=unused-argument
  return locals()


def fn_with_var_kwargs(arg1, kwarg1=None, **kwargs):  # pylint: disable=unused-argument
  return locals()


def fn_with_var_args_and_kwargs(arg1, *args, kwarg1=None, **kwargs):  # pylint: disable=unused-argument
  return locals()


def fn_with_args_and_kwargs_only(*args, **kwargs):  # pylint: disable=unused-argument
  return locals()


def fn_with_position_args(a, b=0, /, c=1, *args, kwarg1=None, **kwargs):  # pylint: disable=keyword-arg-before-vararg, unused-argument
  return locals()


def make_typed_config() -> fdl.Config[SampleClass]:
  """Helper function which returns a fdl.Config whose type is known."""
  return fdl.Config(SampleClass, arg1=1, arg2=2)


def make_untyped_config(arg_to_configure, **kwargs) -> fdl.Config:
  """Helper function which returns an untyped fdl.Config."""
  return fdl.Config(arg_to_configure, **kwargs)


class Unserializable:

  def __getstate__(self):
    raise NotImplementedError()


def _test_fn_unserializable_default(x=Unserializable()):
  return x


@dataclasses.dataclass
class DataclassChild:
  x: int = 0


@dataclasses.dataclass
class DataclassParent:
  child: DataclassChild = dataclasses.field(default_factory=DataclassChild)


def raise_error():
  raise ValueError('My fancy exception')


_T = TypeVar('_T')


@dataclasses.dataclass
class GenericClass(Generic[_T]):
  x: _T = 1


class ConfigTest(parameterized.TestCase):

  def test_config_for_classes(self):
    class_config = fdl.Config(SampleClass, 1, kwarg2='kwarg2')
    pytype_extensions.assert_type(class_config, fdl.Config[SampleClass])
    self.assertEqual(class_config.arg1, 1)
    self.assertEqual(class_config.kwarg2, 'kwarg2')
    class_config.arg1 = 'arg1'
    self.assertEqual(class_config.arg1, 'arg1')
    class_config.arg2 = 'arg2'
    class_config.kwarg1 = 'kwarg1'

    instance = fdl.build(class_config)
    pytype_extensions.assert_type(instance, SampleClass)
    self.assertEqual(instance.arg1, 'arg1')
    self.assertEqual(instance.arg2, 'arg2')
    self.assertEqual(instance.kwarg1, 'kwarg1')
    self.assertEqual(instance.kwarg2, 'kwarg2')

  def test_config_for_functions(self):
    fn_config = fdl.Config(basic_fn, 1, kwarg2='kwarg2')
    pytype_extensions.assert_type(
        fn_config, fdl.Config[Dict[str, Any]]
    )  # use-fiddle-overlay
    self.assertEqual(fn_config.arg1, 1)
    self.assertEqual(fn_config.kwarg2, 'kwarg2')
    fn_config.arg1 = 'arg1'
    self.assertEqual(fn_config.arg1, 'arg1')
    fn_config.arg2 = 'arg2'
    fn_config.kwarg1 = 'kwarg1'

    result = fdl.build(fn_config)
    pytype_extensions.assert_type(result, Dict[str, Any])  # use-fiddle-overlay
    self.assertEqual(result, {
        'arg1': 'arg1',
        'arg2': 'arg2',
        'kwarg1': 'kwarg1',
        'kwarg2': 'kwarg2'
    })

  def test_config_for_functions_with_var_args(self):
    fn_config = fdl.Config(fn_with_var_args, 'arg1', kwarg1='kwarg1')
    pytype_extensions.assert_type(fn_config, fdl.Config)
    fn_args = fdl.build(fn_config)
    self.assertEqual(fn_args['arg1'], 'arg1')
    self.assertEqual(fn_args['kwarg1'], 'kwarg1')

    fn_config.arg1 = 'new_arg1'
    fn_config.kwarg1 = 'new_kwarg1'
    fn_args = fdl.build(fn_config)
    pytype_extensions.assert_type(fn_args, Any)
    self.assertEqual(fn_args['arg1'], 'new_arg1')
    self.assertEqual(fn_args['kwarg1'], 'new_kwarg1')

  def test_config_for_functions_with_var_kwargs(self):
    fn_config = fdl.Config(
        fn_with_var_kwargs,
        'arg1',
        kwarg1='kwarg1',
        kwarg2='kwarg2',
        kwarg3='kwarg3')
    fn_args = fdl.build(fn_config)
    self.assertEqual(fn_args['arg1'], 'arg1')
    self.assertEqual(fn_args['kwarg1'], 'kwarg1')
    self.assertEqual(fn_args['kwargs'], {
        'kwarg2': 'kwarg2',
        'kwarg3': 'kwarg3'
    })

    fn_config.kwarg1 = 'new_kwarg1'
    fn_config.kwarg3 = 'new_kwarg3'
    fn_args = fdl.build(fn_config)
    self.assertEqual(fn_args['kwarg1'], 'new_kwarg1')
    self.assertEqual(fn_args['kwargs'], {
        'kwarg2': 'kwarg2',
        'kwarg3': 'new_kwarg3'
    })

  def test_config_for_functions_with_var_args_and_kwargs(self):
    fn_config = fdl.Config(fn_with_var_args_and_kwargs, arg1='arg1')
    fn_args = fdl.build(fn_config)
    self.assertEqual(fn_args['arg1'], 'arg1')

    fn_config.kwargs = 'kwarg_called_kwarg'
    fn_args = fdl.build(fn_config)
    self.assertEqual(fn_args['kwargs'], {
        'kwargs': 'kwarg_called_kwarg'
    })

  def test_positional_args_access(self):
    fn_config = fdl.Config(fn_with_position_args, 1, 2, 3, 4, 5)
    self.assertEqual(fn_config[0], 1)
    self.assertEqual(fn_config[-1], 5)
    self.assertSequenceEqual(fn_config[3:], [4, 5])
    self.assertSequenceEqual(fn_config[:], [1, 2, 3, 4, 5])

  def test_positional_args_modification(self):
    with self.subTest('with_var_positional'):
      fn_config = fdl.Config(fn_with_position_args, 1, 2, 3, 4, 5)
      fn_config[0] = 0
      self.assertSequenceEqual(fn_config[:], [0, 2, 3, 4, 5])
      fn_config[:3] = [5, 6, 7]
      self.assertSequenceEqual(fn_config[:], [5, 6, 7, 4, 5])
      fn_config[3:] = [8, 9, 10]
      self.assertSequenceEqual(fn_config[:], [5, 6, 7, 8, 9, 10])
      fn_config[:] = [1, 2, 3, 4, 5, 6]
      self.assertSequenceEqual(fn_config[:], [1, 2, 3, 4, 5, 6])

    with self.subTest('wo_var_positional'):

      def foo(a, b, /, c):
        return a, b, c

      fn_config = fdl.Config(foo, 1, 2, 3)
      fn_config[0] = 0
      self.assertSequenceEqual(fn_config[:], [0, 2, 3])
      fn_config[:2] = [4, 5]
      self.assertSequenceEqual(fn_config[:], [4, 5, 3])

  def test_positional_args_delete(self):
    fn_config = fdl.Config(fn_with_position_args, 1, 2, 3, 4, 5, 6)
    del fn_config[0]
    self.assertEqual(fn_config[:], [config_lib.NO_VALUE, 2, 3, 4, 5, 6])
    del fn_config[-1]
    self.assertEqual(fn_config[:], [config_lib.NO_VALUE, 2, 3, 4, 5])
    del fn_config[3:]
    self.assertEqual(fn_config[:], [config_lib.NO_VALUE, 2, 3])

  def test_negative_index_access(self):
    fn_config = fdl.Config(fn_with_position_args, 0, 1, 2, 3, 4, 5)
    del fn_config.c
    self.assertEqual(fn_config[:], [0, 1, 1, 3, 4, 5])
    self.assertEqual(fn_config[-1], 5)
    self.assertEqual(fn_config[-6], 0)

  def test_negative_index_modify(self):
    fn_config = fdl.Config(fn_with_position_args, 0, 1, 2, 3, 4, 5)
    del fn_config.c
    self.assertEqual(fn_config[:], [0, 1, 1, 3, 4, 5])
    fn_config[-1] = -5
    self.assertEqual(fn_config[:], [0, 1, 1, 3, 4, -5])
    fn_config[-6] = 6
    self.assertEqual(fn_config[:], [6, 1, 1, 3, 4, -5])

  def test_modify_positional_only_args(self):
    def foo(a, b, c, /):
      return a, b, c

    fn_config = fdl.Config(foo)
    fn_config[2] = 5
    self.assertEqual(
        fn_config[:], [config_lib.NO_VALUE, config_lib.NO_VALUE, 5]
    )

  def test_negative_index_delete(self):
    fn_config = fdl.Config(fn_with_position_args, 0, 1, 2, 3, 4, 5)
    del fn_config.c
    self.assertEqual(fn_config[:], [0, 1, 1, 3, 4, 5])
    del fn_config[-1]
    self.assertEqual(fn_config[:], [0, 1, 1, 3, 4])
    del fn_config[-5]
    self.assertEqual(
        fn_config[:],
        [config_lib.NO_VALUE, 1, 1, 3, 4],
    )

  def test_varargs_index_handle_access(self):
    fn_config = fdl.Config(fn_with_position_args, 1, 2, 3, 4, 5)
    self.assertEqual(fn_config[fdl.VARARGS], 4)
    self.assertSequenceEqual(fn_config[fdl.VARARGS :], [4, 5])

  def test_varargs_index_handle_modify(self):
    fn_config = fdl.Config(fn_with_position_args, 1, 2, 3, 4, 5)
    fn_config[fdl.VARARGS :] = []
    self.assertSequenceEqual(fn_config[:], [1, 2, 3])
    fn_config[fdl.VARARGS :] = [7, 8, 9]
    self.assertSequenceEqual(fn_config[:], [1, 2, 3, 7, 8, 9])
    fn_config[fdl.VARARGS] = 0
    self.assertSequenceEqual(fn_config[:], [1, 2, 3, 0, 8, 9])

  def test_varargs_index_handle_delete(self):
    fn_config = fdl.Config(fn_with_position_args, 1, 2, 3, 4, 5)
    del fn_config[fdl.VARARGS]
    self.assertSequenceEqual(fn_config[:], [1, 2, 3, 5])
    del fn_config[fdl.VARARGS :]
    self.assertSequenceEqual(fn_config[:], [1, 2, 3])

  def test_modification_when_var_args_are_empty(self):
    fn_config = fdl.Config(fn_with_position_args, 1, 2, 3)
    self.assertEmpty(fn_config[fdl.VARARGS :])
    fn_config[fdl.VARARGS :] = ['a', 'b', 'c']
    self.assertSequenceEqual(fn_config[:], [1, 2, 3, 'a', 'b', 'c'])

  def test_only_some_positional_args_are_defined(self):
    fn_cfg = config_lib.Config(fn_with_position_args, 0)
    self.assertEqual(fn_cfg[:], [0, 0, 1])
    fn_cfg[:2] = [5, 6]
    self.assertEqual(fn_cfg[:], [5, 6, 1])

  def test_positional_args_direct_access_is_forbidden(self):
    fn_config = fdl.Config(fn_with_position_args, 1, 2, 3, 4, 5)
    with self.assertRaisesRegex(
        AttributeError,
        'Cannot access positional-only or variadic positional arguments',
    ):
      _ = fn_config.args

    with self.assertRaisesRegex(
        AttributeError,
        'Cannot access positional-only or variadic positional arguments',
    ):
      _ = fn_config.a

  def test_positional_args_direct_modification_is_forbidden(self):
    fn_config = fdl.Config(fn_with_position_args, 1, 2, 3, 4, 5)
    with self.assertRaisesRegex(
        AttributeError, 'Cannot access VAR_POSITIONAL parameter'
    ):
      fn_config.args = [0]

    with self.assertRaisesRegex(
        AttributeError, 'Cannot access POSITIONAL_ONLY parameter'
    ):
      fn_config.a = 0

  def test_positional_or_keyword_args_have_consistent_values(self):
    fn_config = fdl.Config(fn_with_position_args, 1, 2, 3, 4, 5)
    fn_config[2] = 'arg-c'
    self.assertEqual(fn_config.c, 'arg-c')
    fn_config.c = 'index-2'
    self.assertEqual(fn_config[2], 'index-2')

  def test_set_out_of_range_index(self):
    fn_config = config_lib.Config(fn_with_position_args)
    arr = [1, 2, 3]
    added = [7, 8, 9]
    fn_config[:3] = arr
    fn_config[10:] = added
    arr[10:] = added
    self.assertSequenceEqual(fn_config[:], arr)

  def test_index_out_of_range(self):
    fn_config = fdl.Config(fn_with_var_args, 1, 2)
    self.assertLen(fn_config[:], 2)
    with self.assertRaisesRegex(IndexError, 'index out of range'):
      fn_config[2] = 'index-2'
    with self.assertRaisesRegex(IndexError, 'index out of range'):
      _ = fn_config[2]

  def test_args_config_shallow_copy(self):
    fn_config = fdl.Config(fn_with_var_args, 1, 2)
    self.assertLen(fn_config[:], 2)
    a_copy = fn_config[:]
    a_copy.append('3')
    self.assertLen(fn_config[:], 2)
    self.assertLen(a_copy, 3)

  def test_args_config_build(self):
    fn_config = fdl.Config(fn_with_position_args, 1, 2, 3, 4, 5)
    self.assertEqual(
        fdl.build(fn_config),
        {'a': 1, 'b': 2, 'c': 3, 'args': (4, 5), 'kwarg1': None, 'kwargs': {}},
    )

  def test_nested_positional_args_config_build(self):
    fn_config = fdl.Config(
        fn_with_position_args,
        1,
        2,
        3,
        4,
        5,
        kwarg1=fdl.Config(
            fn_with_position_args, 'a', 'b', kwarg1='kwarg1', kwarg2='kwarg2'
        ),
    )
    self.assertEqual(
        fdl.build(fn_config),
        {
            'a': 1,
            'b': 2,
            'c': 3,
            'args': (4, 5),
            'kwarg1': {
                'a': 'a',
                'b': 'b',
                'c': 1,
                'kwarg1': 'kwarg1',
                'args': tuple(),
                'kwargs': {'kwarg2': 'kwarg2'},
            },
            'kwargs': {},
        },
    )

  def test_config_for_dicts(self):
    dict_config = fdl.Config(dict, a=1, b=2)
    dict_config.c = 3
    instance = fdl.build(dict_config)
    self.assertEqual(instance, {'a': 1, 'b': 2, 'c': 3})

  def test_config_with_default_args(self):

    def my_func(x: int = 2, y: str = 'abc'):  # pylint: disable=unused-argument
      return locals()

    cfg = fdl.Config(my_func)
    self.assertEqual(2, cfg.x)
    self.assertEqual('abc', cfg.y)

    obj = fdl.build(cfg)
    self.assertEqual(2, obj['x'])
    self.assertEqual('abc', obj['y'])

    cfg.y = 'xyz'
    obj = fdl.build(cfg)
    self.assertEqual(2, obj['x'])
    self.assertEqual('xyz', obj['y'])

  def test_config_defaults_not_materialized(self):

    def my_func(x: int, y: str = 'abc', z: float = 2.0):  # pylint: disable=unused-argument
      return locals()

    cfg = fdl.Config(my_func)

    self.assertEqual('abc', cfg.y)  # Should return the default.
    self.assertEqual({}, cfg.__arguments__)  # but not materialized.

    cfg.x = 42
    output = fdl.build(cfg)

    self.assertEqual({'x': 42, 'y': 'abc', 'z': 2.0}, output)

  def test_nested_configs(self):
    fn_config1_args = {
        'arg1': 'innermost1',
        'arg2': 'innermost2',
        'kwarg1': 'kw1',
        'kwarg2': 'kw2'
    }
    fn_config1 = fdl.Config(basic_fn, *fn_config1_args.values())

    class_config = fdl.Config(
        SampleClass, arg1=fdl.cast(fdl.Partial, fn_config1), arg2=fn_config1)
    fn_config2 = fdl.Config(
        basic_fn, arg1=fdl.cast(fdl.Partial, class_config), arg2=class_config)

    fn_config2_args = fdl.build(fn_config2)

    test_class_partial = fn_config2_args['arg1']
    test_class_instance = test_class_partial()
    self.assertEqual(type(test_class_instance), SampleClass)
    self.assertEqual(test_class_instance.arg1(), fn_config1_args)
    self.assertEqual(test_class_instance.arg2, fn_config1_args)

    test_class_instance = fn_config2_args['arg2']
    self.assertEqual(type(test_class_instance), SampleClass)
    self.assertEqual(test_class_instance.arg1(), fn_config1_args)
    self.assertEqual(test_class_instance.arg2, fn_config1_args)

  def test_instance_sharing(self):
    class_config = fdl.Config(
        SampleClass, 'arg1', 'arg2', kwarg1='kwarg1', kwarg2='kwarg2')
    class_config_copy = copy.copy(class_config)
    class_config_copy.arg1 = 'separate_arg1'

    # Changing copied config parameters doesn't change the original fdl.
    self.assertEqual(class_config.arg1, 'arg1')

    fn_config = fdl.Config(basic_fn, class_config_copy, {
        'key1': [class_config, class_config],
        'key2': (class_config,)
    })

    fn_args = fdl.build(fn_config)
    separate_instance = fn_args['arg1']
    shared_instance = fn_args['arg2']['key1'][0]
    structure = fn_args['arg2']

    self.assertIsNot(shared_instance, separate_instance)
    self.assertIs(structure['key1'][0], shared_instance)
    self.assertIs(structure['key1'][1], shared_instance)
    self.assertIs(structure['key2'][0], shared_instance)

    self.assertEqual(shared_instance.arg1, 'arg1')
    self.assertEqual(separate_instance.arg1, 'separate_arg1')

  def test_instance_sharing_collections(self):
    child_configs = [fdl.Config(basic_fn, 1, 'a'), fdl.Config(basic_fn, 2, 'b')]
    cfg = fdl.Config(SampleClass)
    cfg.arg1 = child_configs
    cfg.arg2 = child_configs
    obj = fdl.build(cfg)

    self.assertIsInstance(obj, SampleClass)
    self.assertIs(obj.arg1, obj.arg2)
    self.assertIs(obj.arg1[0], obj.arg2[0])
    self.assertIs(obj.arg1[1], obj.arg2[1])

  def test_shallow_copy(self):
    class_config = fdl.Config(SampleClass, 'arg1', 'arg2')
    fn_config = fdl.Config(basic_fn, class_config, 'fn_arg2')
    fn_config_copy = copy.copy(fn_config)
    # Changing the copy doesn't change the original.
    fn_config_copy.arg2 = 'fn_arg2_copy'
    self.assertEqual(fn_config.arg2, 'fn_arg2')
    # But modifying a shared value is seen by both.
    fn_config_copy.arg1.arg2 = 'mutated'  # pytype: disable=not-writable  # use-fiddle-overlay
    self.assertEqual(fn_config.arg1.arg2, 'mutated')  # pytype: disable=attribute-error  # use-fiddle-overlay

  def test_buildable_subclass(self):

    class SampleClassConfig(fdl.Config):

      def __init__(self, *args, **kwargs):
        super().__init__(SampleClass, *args, **kwargs)

      @classmethod
      def __unflatten__(cls, values, metadata):
        return cls(**metadata.arguments(values))

    sub_cfg = SampleClassConfig(1, 2)
    cfg = SampleClassConfig('foo', sub_cfg)

    with self.subTest('copy'):
      cfg_copy = copy.copy(cfg)
      self.assertEqual(cfg, cfg_copy)
      self.assertIs(cfg_copy.arg2, sub_cfg)
      cfg_copy.arg1 = 'blah'
      self.assertNotEqual(cfg, cfg_copy)

    with self.subTest('deepcopy'):
      cfg_deepcopy = copy.deepcopy(cfg)
      self.assertEqual(cfg, cfg_deepcopy)
      self.assertIsNot(cfg_deepcopy.arg2, sub_cfg)
      cfg_copy.arg2.arg1 = 'blah'
      self.assertNotEqual(cfg, cfg_deepcopy)

    with self.subTest('traverse'):
      values_by_path = daglish_legacy.collect_value_by_path(
          cfg, memoizable_only=False)
      values_by_path_str = {
          daglish.path_str(path): value
          for path, value in values_by_path.items()
      }
      expected = {
          '': cfg,
          '.arg1': 'foo',
          '.arg2': sub_cfg,
          '.arg2.arg1': 'blah',
          '.arg2.arg2': 2
      }
      self.assertEqual(expected, values_by_path_str)

  def test_deep_copy(self):
    class_config = fdl.Config(SampleClass, 'arg1', 'arg2')
    fn_config = fdl.Config(basic_fn, class_config, 'fn_arg2')
    fn_config_copy = copy.deepcopy(fn_config)
    # Changing the copy doesn't change the original.
    fn_config_copy.arg2 = 'fn_arg2_copy'
    self.assertEqual(fn_config.arg2, 'fn_arg2')
    # With a deep copy, the value is no longer shared.
    fn_config_copy.arg1.arg2 = 'mutated'  # pytype: disable=not-writable  # use-fiddle-overlay
    self.assertEqual(fn_config.arg1.arg2, 'arg2')  # pytype: disable=attribute-error  # use-fiddle-overlay

  def test_deep_copy_preserves_instance_sharing(self):
    class_config = fdl.Config(SampleClass, 'arg1', 'arg2')
    fn_config = fdl.Config(basic_fn, arg1=class_config, arg2=class_config)
    self.assertIs(fn_config.arg1, fn_config.arg2)
    fn_config_copy = copy.deepcopy(fn_config)
    self.assertIsNot(fn_config.arg1, fn_config_copy.arg1)
    self.assertIs(fn_config_copy.arg1, fn_config_copy.arg2)

  def test_deep_copy_no_value(self):
    self.assertIs(fdl.NO_VALUE, copy.copy(fdl.NO_VALUE))
    self.assertIs(fdl.NO_VALUE, copy.deepcopy(fdl.NO_VALUE))

  def test_equality_arguments(self):
    cfg1 = fdl.Config(SampleClass, 'arg1')
    cfg2 = fdl.Config(SampleClass, 'arg1')
    self.assertEqual(cfg1, cfg2)
    cfg2.arg1 = 'arg2'
    self.assertNotEqual(cfg1, cfg2)

  def test_equality_arguments_nested(self):

    def make_nested_config():
      fn_config1 = fdl.Config(
          basic_fn,
          arg1='innermost1',
          arg2='innermost2',
          kwarg1='kw1',
          kwarg2='kw2')
      class_config = fdl.Config(
          SampleClass, arg1=fdl.cast(fdl.Partial, fn_config1), arg2=fn_config1)
      fn_config2 = fdl.Config(
          basic_fn, arg1=fdl.cast(fdl.Partial, class_config), arg2=class_config)
      return fn_config2

    cfg1 = make_nested_config()
    cfg2 = make_nested_config()
    self.assertEqual(cfg1, cfg2)
    cfg2.arg2.arg1.kwarg1 = 'another value'  # pytype: disable=attribute-error  # use-fiddle-overlay
    self.assertNotEqual(cfg1, cfg2)
    cfg1.arg2.arg1.kwarg1 = 'another value'  # pytype: disable=attribute-error  # use-fiddle-overlay
    self.assertEqual(cfg1, cfg2)

  def test_equality_fn_or_cls_mismatch(self):
    cls_cfg = fdl.Config(SampleClass, 'arg1')
    fn_cfg = fdl.Config(basic_fn, 'arg1')
    self.assertNotEqual(cls_cfg, fn_cfg)

  def test_equality_buildable_type_mismatch(self):
    cfg = fdl.Config(SampleClass, 'arg1')

    # Compare to something that isn't a `Buildable`.
    self.assertNotEqual(cfg, 5)

    # Compare to a `Partial`.
    partial = fdl.cast(fdl.Partial, cfg)
    self.assertNotEqual(cfg, partial)

    # Compare to a `Config` subclass.
    class ConfigSubClass(fdl.Config):
      pass

    cfg_subclass = fdl.cast(ConfigSubClass, cfg)
    self.assertNotEqual(cfg, cfg_subclass)
    # The logic governing how __eq__ is actually invoked from an == comparison
    # actually takes subclassing relationships into account and always calls
    # b.__eq__(a) if isinstance(b, a.__class__), so testing explicitly here.
    self.assertFalse(cfg.__eq__(cfg_subclass))
    self.assertFalse(cfg_subclass.__eq__(cfg))

  def test_equality_classmethods(self):
    cfg_a = fdl.Config(SampleClass.a_classmethod)  # pytype: disable=invalid-annotation  # use-fiddle-overlay
    cfg_b = fdl.Config(SampleClass.a_classmethod)  # pytype: disable=invalid-annotation  # use-fiddle-overlay
    self.assertEqual(cfg_a, cfg_b)

  def test_equality_positional_args_default_value(self):
    cfg_a = fdl.Config(fn_with_position_args, 'a')
    cfg_b = fdl.Config(fn_with_position_args, 'a', 0)
    cfg_c = fdl.Config(fn_with_position_args, 'a', c=1)
    cfg_d = fdl.Config(fn_with_position_args, 'a', 0, 1, 2, 3, 4, 5)
    self.assertEqual(cfg_a, cfg_b)
    self.assertEqual(cfg_a, cfg_c)
    self.assertNotEqual(cfg_a, cfg_d)

  def test_default_value_equality(self):
    cfg1 = fdl.Config(SampleClass, 1, 2)
    cfg2 = fdl.Config(SampleClass, 1, 2, None, kwarg2=None)
    self.assertEqual(cfg1, cfg2)

    cfg1 = fdl.Config(basic_fn, 1, 2)
    cfg2 = fdl.Config(basic_fn, 1, 2, None, kwarg2=None)
    self.assertEqual(cfg1, cfg2)

  def test_generic_classes(self):
    if sys.version_info >= (3, 9):
      cfg = fdl.Config(GenericClass, 1)
      self.assertEqual(fdl.build(cfg), GenericClass(1))

    self.assertEqual(fdl.Config(GenericClass).x, 1)
    self.assertEqual(fdl.Config(GenericClass[int]).x, 1)

  def test_config_with_non_comparable_values(self):
    # This test ensures that fdl.Config and fdl.build work properly with
    # argument defaults that don't support equality testing. Something related
    # can come up with NumPy arrays, which test for equality against scalars in
    # an elementwise fashion, and subsequently can't directly be coerced to a
    # bool (e.g., used as the condition in an if statement).

    @dataclasses.dataclass(frozen=True, eq=False)
    class ClassWithDisabledEquality:

      def __eq__(self, _):
        raise NotImplementedError()

    def fn_with_non_comparable_default(
        value1, value2=ClassWithDisabledEquality()
    ):
      return value1, value2

    cfg = fdl.Config(fn_with_non_comparable_default)
    cfg.value1 = ClassWithDisabledEquality()
    value1, value2 = fdl.build(cfg)
    self.assertIsInstance(value1, ClassWithDisabledEquality)
    self.assertIsInstance(value2, ClassWithDisabledEquality)

  def test_config_dag_structure_comparison(self):
    a = fdl.Config(SampleClass, 1, 2)
    b = fdl.Config(SampleClass, 1, 2)
    with self.subTest('python_list'):
      x = [a, a]
      y = [a, b]
      self.assertEqual(x, y)

    with self.subTest('node_sharing_detection'):
      x = fdl.Config(SampleClass, a, b)
      y = fdl.Config(SampleClass, a, a)
      self.assertNotEqual(x, y)

    with self.subTest('node_sharing_difference'):
      x = fdl.Config(SampleClass, a, b, b)
      y = fdl.Config(SampleClass, a, a, b)
      self.assertNotEqual(x, y)

  def test_config_internables_comparison(self):
    x, y = demo_configs.get_equal_but_not_object_identical_string_configs()
    self.assertEqual(x, y)

  def test_dict_with_different_order_comparison(self):
    x, y = demo_configs.get_equal_but_not_object_identical_string_configs()
    self.assertEqual(x, y)

  def test_unsetting_argument(self):
    fn_config = fdl.Config(basic_fn)
    fn_config.arg1 = 3
    fn_config.arg2 = 4

    del fn_config.arg1

    with self.assertRaisesRegex(AttributeError,
                                "No parameter 'arg1' has been set"):
      _ = fn_config.arg1

    with self.assertRaisesRegex(AttributeError, "'arg1'"):
      del fn_config.arg1

    with self.assertRaisesRegex(AttributeError, "'unknown_arg'"):
      del fn_config.unknown_arg

    self.assertEqual(4, fn_config.arg2)

  def test_flatten_unflatten_histories(self):
    cfg = fdl.Config(SampleClass)
    cfg.arg1 = 4
    cfg.arg1 = 5
    values, metadata = cfg.__flatten__()
    copied = fdl.Config.__unflatten__(values, metadata)
    self.assertEqual(
        copied.__argument_history__['arg1'][-1].location.line_number,
        cfg.__argument_history__['arg1'][-1].location.line_number)
    self.assertEqual(copied.__argument_history__['arg1'][0].new_value, 4)
    self.assertEqual(copied.__argument_history__['arg1'][1].new_value, 5)
    self.assertEqual(copied.__argument_history__['arg1'][-1].kind,
                     history.ChangeKind.NEW_VALUE)

  def test_dir_simple(self):
    fn_config = fdl.Config(basic_fn)
    self.assertEqual(['arg1', 'arg2', 'kwarg1', 'kwarg2'], dir(fn_config))

  def test_dir_cls(self):
    cfg = fdl.Config(SampleClass)
    self.assertEqual(['arg1', 'arg2', 'kwarg1', 'kwarg2'], dir(cfg))

  def test_dir_var_args_and_kwargs(self):
    varargs_config = fdl.Config(fn_with_var_args_and_kwargs)
    varargs_config.abc = '123'
    self.assertEqual(['abc', 'arg1', 'kwarg1'], dir(varargs_config))

  def test_typed_config(self):
    class_config = make_typed_config()
    pytype_extensions.assert_type(class_config, fdl.Config[SampleClass])
    instance = fdl.build(class_config)
    pytype_extensions.assert_type(instance, SampleClass)
    self.assertEqual(instance.arg1, 1)
    self.assertEqual(instance.arg2, 2)

  def test_untyped_config(self):
    class_config = make_untyped_config(SampleClass, arg1=2, arg2=3)
    pytype_extensions.assert_type(class_config, fdl.Config)
    instance = fdl.build(class_config)
    pytype_extensions.assert_type(instance, Any)
    self.assertEqual(instance.arg1, 2)
    self.assertEqual(instance.arg2, 3)

  def test_repr_class_config(self):
    class_config = fdl.Config(SampleClass, 1, 2, kwarg1='kwarg1')
    expected_repr = "<Config[SampleClass(arg1=1, arg2=2, kwarg1='kwarg1')]>"
    self.assertEqual(repr(class_config), expected_repr)

  def test_repr_nested_indentation(self):
    config = fdl.Config(basic_fn, 1,
                        fdl.Config(basic_fn, 'x' * 50, fdl.Config(basic_fn, 1)))
    expected_repr = """<Config[basic_fn(
  arg1=1,
  arg2=<Config[basic_fn(
    arg1='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
    arg2=<Config[basic_fn(arg1=1)]>)]>)]>"""
    self.assertEqual(repr(config), expected_repr)

  def test_repr_fn_config(self):
    fn_config = fdl.Config(basic_fn, 1, 2, kwarg1='kwarg1')
    expected_repr = "<Config[basic_fn(arg1=1, arg2=2, kwarg1='kwarg1')]>"
    self.assertEqual(repr(fn_config), expected_repr)

  def test_repr_varkwargs(self):
    # Note: in the repr, kwarg1 comes before x and y and z because kwarg1 is an
    # explicit keyword parameter, while x, y, and z are **kwargs parameters.
    cfg = fdl.Config(fn_with_var_kwargs, 1, x=2, z=3, y=4, kwarg1=5)
    expected_repr = (
        '<Config[fn_with_var_kwargs(arg1=1, kwarg1=5, x=2, z=3, y=4)]>')
    self.assertEqual(repr(cfg), expected_repr)

  def test_nonexistent_attribute_error(self):
    class_config = fdl.Config(SampleClass, 1)
    expected_msg = (r"No parameter 'nonexistent_arg' has been set on "
                    r'<Config\[SampleClass\(arg1=1\)\]>\.')
    with self.assertRaisesRegex(AttributeError, expected_msg):
      getattr(class_config, 'nonexistent_arg')

  def test_nonexistent_parameter_error(self):
    class_config = fdl.Config(SampleClass)
    expected_msg = (r"No parameter named 'nonexistent_arg' exists for "
                    r"<class '.*\.SampleClass'>; valid parameter names: "
                    r'arg1, arg2, kwarg1, kwarg2\.')
    with self.assertRaisesRegex(AttributeError, expected_msg):
      class_config.nonexistent_arg = 'error!'

  def test_build_inside_build(self):

    def inner_build(x: int) -> str:
      return str(x)

    def nest_call(x: int) -> str:
      cfg = fdl.Config(inner_build, x)
      return fdl.build(cfg)

    outer = fdl.Config(nest_call, 3)
    expected_msg = r'forbidden to call `fdl.build` inside another `fdl.build`'
    with self.assertRaisesRegex(Exception, expected_msg):
      fdl.build(outer)

  def test_history_tracking(self):
    cfg = fdl.Config(SampleClass, 'arg1_value')
    cfg.arg2 = 'arg2_value'
    del cfg.arg1

    self.assertEqual(
        set(['arg1', 'arg2', '__fn_or_cls__']),
        set(cfg.__argument_history__.keys()))
    self.assertLen(cfg.__argument_history__['arg1'], 2)
    self.assertLen(cfg.__argument_history__['arg2'], 1)
    self.assertEqual('arg1', cfg.__argument_history__['arg1'][0].param_name)
    self.assertEqual('arg1_value',
                     cfg.__argument_history__['arg1'][0].new_value)
    self.assertRegex(
        str(cfg.__argument_history__['arg1'][0].location),
        r'config_test.py:\d+:test_history_tracking')
    self.assertEqual('arg2', cfg.__argument_history__['arg2'][0].param_name)
    self.assertEqual('arg2_value',
                     cfg.__argument_history__['arg2'][0].new_value)
    self.assertRegex(
        str(cfg.__argument_history__['arg2'][0].location),
        r'config_test.py:\d+:test_history_tracking')
    self.assertEqual('arg1', cfg.__argument_history__['arg1'][1].param_name)
    self.assertEqual(history.DELETED,
                     cfg.__argument_history__['arg1'][1].new_value)
    self.assertRegex(
        str(cfg.__argument_history__['arg1'][1].location),
        r'config_test.py:\d+:test_history_tracking')

    self.assertEqual(cfg.__argument_history__['arg1'][0].sequence_id + 1,
                     cfg.__argument_history__['arg2'][0].sequence_id)
    self.assertEqual(cfg.__argument_history__['arg2'][0].sequence_id + 1,
                     cfg.__argument_history__['arg1'][1].sequence_id)

  def test_custom_location_history_tracking(self):
    with history.custom_location(lambda: 'abc:123'):
      cfg = fdl.Config(SampleClass, 'arg1')
    cfg.arg2 = 'arg2'
    self.assertEqual(
        set(['arg1', 'arg2', '__fn_or_cls__']),
        set(cfg.__argument_history__.keys()))
    self.assertLen(cfg.__argument_history__['arg1'], 1)
    self.assertLen(cfg.__argument_history__['arg2'], 1)
    self.assertEqual('arg1', cfg.__argument_history__['arg1'][0].param_name)
    self.assertRegex(
        str(cfg.__argument_history__['arg1'][0].location), 'abc:123')
    self.assertEqual('arg2', cfg.__argument_history__['arg2'][0].param_name)
    self.assertRegex(
        str(cfg.__argument_history__['arg2'][0].location),
        r'config_test.py:\d+:test_custom_location_history_tracking')
    self.assertEqual(cfg.__argument_history__['arg1'][0].sequence_id + 1,
                     cfg.__argument_history__['arg2'][0].sequence_id)

  def test_accessing_functions_on_config(self):
    """Test helpful error messages when users hold them wrong!

    Config objects should provide a helpful error message when users attempt
    to use them as if they were the actual built (underlying) objects.
    """

    cfg = fdl.Config(SampleClass)
    expected_msg = 'a_method.*Note: .*SampleClass has an attribute/method with '
    with self.assertRaisesRegex(AttributeError, expected_msg):
      cfg.a_method()
    with self.assertRaisesRegex(AttributeError, expected_msg):
      _ = cfg.a_method

  def test_unhashable(self):
    """All Buildable's should be unhashable: mutability and custom __eq__."""
    with self.assertRaisesRegex(TypeError, 'unhashable'):
      _ = fdl.Config(SampleClass) in {}
    with self.assertRaisesRegex(TypeError, 'unhashable'):
      _ = fdl.Partial(SampleClass) in {}

  def test_pickling_config(self):
    """Bulidable types should be pickle-able."""
    cfg = fdl.Config(SampleClass, 1, 'abc')
    self.assertEqual(cfg, pickle.loads(pickle.dumps(cfg)))
    reloaded = pickle.loads(pickle.dumps(cfg))
    reloaded.kwarg1 = 3  # mutate after unpickling.
    self.assertNotEqual(cfg, reloaded)

  def test_pickling_non_serializable_default(self):
    pickle.dumps(fdl.Config(_test_fn_unserializable_default))

  def test_build_nested_structure(self):
    class_config = fdl.Config(SampleClass, 'arg1', 'arg2')
    built = fdl.build([class_config, {'child': class_config}])
    self.assertIsInstance(built[0], SampleClass)
    self.assertEqual(built[0].arg1, 'arg1')
    self.assertEqual(built[0].arg2, 'arg2')
    self.assertIs(built[0], built[1]['child'])

  def test_build_raises_nice_error_too_few_args(self):
    cfg = fdl.Config(basic_fn, fdl.Config(SampleClass, 1), 2)
    with self.assertRaises(TypeError) as e:
      fdl.build(cfg)
    self.assertEqual(
        e.exception.proxy_message,  # pytype: disable=attribute-error
        '\n\nFiddle context: failed to construct or call SampleClass at '
        '<root>.arg1 with positional arguments: (), keyword arguments: '
        '(arg1=1).',
    )

  def test_build_raises_exception_on_call(self):
    cfg = fdl.Config(raise_error)
    msg = (
        'My fancy exception\n\nFiddle context: failed to construct or call '
        'raise_error at <root> with positional arguments: (), '
        'keyword arguments: ().'
    )
    with self.assertRaisesWithLiteralMatch(ValueError, msg):
      fdl.build(cfg)

  def test_build_error_path(self):
    # This will raise an error, because it doesn't have one arg populated.
    sub_cfg = fdl.Config(basic_fn, 1)
    sub_dict = {'a': 0, 'b': 2, 'c': sub_cfg, 'd': 10}
    cfg = fdl.Config(fn_with_var_kwargs, [1, sub_dict])

    with self.assertRaises(TypeError) as e:
      fdl.build(cfg)

    self.assertEqual(
        e.exception.proxy_message,  # pytype: disable=attribute-error
        '\n\nFiddle context: failed to construct or call basic_fn at <root>.'
        "arg1[1]['c'] with positional arguments: (), keyword arguments: "
        '(arg1=1).',
    )

  def test_multithreaded_build(self):
    """Two threads can each invoke build.build without interfering."""
    output = None
    background_entered_build = threading.Event()
    foreground_entered_build = threading.Event()

    def other_thread():
      nonlocal output

      def blocking_function(x):
        background_entered_build.set()
        foreground_entered_build.wait()
        return x

      cfg = fdl.Config(blocking_function, 3)
      output = fdl.build(cfg)

    def blocking_function(x):
      foreground_entered_build.set()
      background_entered_build.wait()
      return x

    cfg = fdl.Config(blocking_function, 1)
    thread = threading.Thread(target=other_thread)
    thread.start()
    obj = fdl.build(cfg)
    thread.join()
    self.assertEqual(1, obj)
    self.assertEqual(3, output)

  def test_copy_constructor_errors(self):
    cfg1 = fdl.Config(fn_with_var_kwargs, 1, 2)
    fdl.add_tag(cfg1, 'arg1', Tag1)
    with self.assertRaises(ValueError):
      fdl.Partial(cfg1)  # pytype: disable=invalid-annotation  # use-fiddle-overlay

  def test_copy_constructor_with_updates_errors(self):
    cfg1 = fdl.Config(fn_with_var_kwargs, 1, 2, c=[])
    fdl.add_tag(cfg1, 'arg1', Tag1)
    with self.assertRaises(ValueError):
      fdl.Partial(cfg1, 5, a='a', b='b')

  def test_dataclass_default_factory(self):

    cfg = fdl.Config(DataclassParent)

    with self.subTest('read_default_is_error'):
      expected_error = (
          r"Can't get default value for dataclass field DataclassParent\.child "
          r'since it uses a default_factory\.')
      with self.assertRaisesRegex(ValueError, expected_error):
        cfg.child.x = 5

    with self.subTest('read_ok_after_override'):
      cfg.child = fdl.Config(DataclassChild)  # override default w/ a value
      cfg.child.x = 5  # now it's ok to configure child.
      self.assertEqual(fdl.build(cfg), DataclassParent(DataclassChild(5)))


class OrderedArgumentsTest(parameterized.TestCase):

  def test_ordered_arguments(self):
    cfg = fdl.Config(fn_with_position_args, 0, 1, 2, 3, 4)
    cfg.var_kwarg1 = 5
    cfg.kwarg1 = 'hi'
    cfg.var_kwarg2 = 6
    cfg.c = -2
    self.assertEqual(
        list(cfg.__arguments__.items()),
        [
            ('c', -2),
            (0, 0),
            (1, 1),
            (3, 3),
            (4, 4),
            ('var_kwarg1', 5),
            ('kwarg1', 'hi'),
            ('var_kwarg2', 6),
        ],
    )
    self.assertEqual(
        list(config_lib.ordered_arguments(cfg).items()),
        [
            (0, 0),
            (1, 1),
            ('c', -2),
            (3, 3),
            (4, 4),
            ('kwarg1', 'hi'),
            ('var_kwarg1', 5),
            ('var_kwarg2', 6),
        ],
    )

    with self.subTest('path element and keys match up'):
      _, metadata = cfg.__flatten__()
      path_elements = cfg.__path_elements__()
      names = []
      for path_element in path_elements:
        if isinstance(path_element, daglish.Attr):
          names.append(path_element.name)
        elif isinstance(path_element, daglish.Index):
          names.append(path_element.index)
        else:
          raise ValueError(
              'No other path element should exists, got:', path_element
          )
      self.assertEqual(
          metadata.argument_names,
          tuple(names),
      )

  @parameterized.parameters([
      # incl_var_kw, incl_defaults, incl_unset, incl_eq_to_default, expected
      (False, False, False, True, [('arg2', 5), ('kwarg2', 99)]),
      (
          False,
          True,
          False,
          True,
          [('arg2', 5), ('kwarg1', None), ('kwarg2', 99)],
      ),
      (
          False,
          False,
          True,
          True,
          [('arg1', fdl.NO_VALUE), ('arg2', 5), ('kwarg2', 99)],
      ),
      (
          False,
          True,
          True,
          True,
          [
              ('arg1', fdl.NO_VALUE),
              ('arg2', 5),
              ('kwarg1', None),
              ('kwarg2', 99),
          ],
      ),
      (True, False, False, True, [('arg2', 5), ('kwarg2', 99), ('foo', 12)]),
      (
          True,
          True,
          False,
          True,
          [('arg2', 5), ('kwarg1', None), ('kwarg2', 99), ('foo', 12)],
      ),
      (
          True,
          False,
          True,
          True,
          [('arg1', fdl.NO_VALUE), ('arg2', 5), ('kwarg2', 99), ('foo', 12)],
      ),
      (
          True,
          True,
          True,
          True,
          [
              ('arg1', fdl.NO_VALUE),
              ('arg2', 5),
              ('kwarg1', None),
              ('kwarg2', 99),
              ('foo', 12),
          ],
      ),
      (False, False, False, False, [('arg2', 5)]),
      (True, False, False, False, [('arg2', 5), ('foo', 12)]),
  ])
  def test_ordered_arguments_options(
      self,
      include_var_keyword,
      include_defaults,
      include_unset,
      include_equal_to_default,
      expected,
  ):
    def fn(arg1, arg2, kwarg1=None, kwarg2=99, **kwargs):
      return (arg1, arg2, kwarg1, kwarg2, kwargs)

    # Arguments to the function:
    #   * arg1 has no value
    #   * arg2 has an explicit value
    #   * kwarg1 has no value, but has a default value
    #   * kwarg2 is explicitly set to its default value
    #   * foo is consumed by var_keyword arg.
    cfg = fdl.Config(fn, arg2=5, kwarg2=99, foo=12)

    args = fdl.ordered_arguments(
        cfg,
        include_var_keyword=include_var_keyword,
        include_defaults=include_defaults,
        include_unset=include_unset,
        include_equal_to_default=include_equal_to_default,
    )
    self.assertEqual(list(args.items()), expected)

  def test_include_positional(self):
    cfg = fdl.Config(fn_with_position_args, 0, 1, 2, 3, 4, 5, kwarg='hi')
    self.assertEqual(
        config_lib.ordered_arguments(cfg, include_positional=False),
        {'c': 2, 'kwarg': 'hi'},
    )


if __name__ == '__main__':
  absltest.main()
