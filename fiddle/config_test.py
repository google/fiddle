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

"""Tests for the `fiddle.config` module."""

import copy
import dataclasses
import functools
import pickle
import threading
from typing import Any, Callable, Dict

from absl.testing import absltest
import fiddle as fdl
from fiddle import config as config_lib
from fiddle import daglish
from fiddle import history
from fiddle.experimental import daglish_legacy

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


class FiddleInitStaticMethod:

  def __init__(self, x, y=3, **kwargs):
    self.x = x
    self.y = y
    for key, value in kwargs.items():
      setattr(self, key, value)

  @staticmethod
  def __fiddle_init__(cfg):
    cfg.z = 5


class FiddleInitStaticMethodChild(FiddleInitStaticMethod):

  @classmethod
  def __fiddle_init__(cls, cfg):
    super().__fiddle_init__(cfg)
    cfg.w = 42


class FiddleInitClassMethod:

  def __init__(self, x, y=3, **kwargs):
    self.x = x
    self.y = y
    for key, value in kwargs.items():
      setattr(self, key, value)

  @classmethod
  def __fiddle_init__(cls, cfg):
    cfg.z = 5


class FiddleInitClassMethodChild(FiddleInitClassMethod):

  @classmethod
  def __fiddle_init__(cls, cfg):
    super().__fiddle_init__(cfg)
    cfg.w = 10


class FiddleInitIncompatibleChild(FiddleInitClassMethod):

  def __init__(self, a, b):
    self.a = a
    self.b = b
    super().__init__(x=a + 1, y=b - 3, z=a + b)

  @classmethod
  def __fiddle_init__(cls, cfg):
    # Since FiddleInitClassMethod has different init arguments, we cannot call
    # super().__fiddle_init__(cfg).
    super_cfg = fdl.Config(FiddleInitClassMethod)
    cfg.b = super_cfg.y + 3  # Demonstrate copying over.


class FiddleInitIncompatibleChildBroken(FiddleInitClassMethod):

  def __init__(self, a, b):  # pylint: disable=super-init-not-called
    pass


def make_typed_config() -> fdl.Config[SampleClass]:
  """Helper function which returns a fdl.Config whose type is known."""
  return fdl.Config(SampleClass, arg1=1, arg2=2)


def make_untyped_config(arg_to_configure, **kwargs) -> fdl.Config:
  """Helper function which returns an untyped fdl.Config."""
  return fdl.Config(arg_to_configure, **kwargs)


def make_typed_partial() -> fdl.Partial[SampleClass]:
  """Helper function to create a typed Partial instance."""
  return fdl.Partial(SampleClass, arg1=1)


def make_untyped_partial(arg_to_configure, **kwargs) -> fdl.Partial:
  """Helper function which returns an untyped fdl.Config."""
  return fdl.Partial(arg_to_configure, **kwargs)


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


class ConfigTest(absltest.TestCase):

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
    pytype_extensions.assert_type(fn_config, fdl.Config[Dict[str, Any]])
    self.assertEqual(fn_config.arg1, 1)
    self.assertEqual(fn_config.kwarg2, 'kwarg2')
    fn_config.arg1 = 'arg1'
    self.assertEqual(fn_config.arg1, 'arg1')
    fn_config.arg2 = 'arg2'
    fn_config.kwarg1 = 'kwarg1'

    result = fdl.build(fn_config)
    pytype_extensions.assert_type(result, Dict[str, Any])
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

    fn_config.args = 'kwarg_called_arg'
    fn_config.kwargs = 'kwarg_called_kwarg'
    fn_args = fdl.build(fn_config)
    self.assertEqual(fn_args['kwargs'], {
        'args': 'kwarg_called_arg',
        'kwargs': 'kwarg_called_kwarg'
    })

  def test_fiddle_init_config_static(self):
    cfg = fdl.Config(FiddleInitStaticMethod)
    self.assertEqual(5, cfg.z)
    cfg.x = 1
    obj = fdl.build(cfg)
    self.assertIsInstance(obj, FiddleInitStaticMethod)
    self.assertEqual(1, obj.x)
    self.assertEqual(3, obj.y)
    self.assertEqual(5, obj.z)  # pytype: disable=attribute-error

  def test_fiddle_init_partial_static(self):
    cfg = fdl.Partial(FiddleInitStaticMethod)
    self.assertEqual(5, cfg.z)
    partial = fdl.build(cfg)
    obj = partial(x=1)
    self.assertIsInstance(obj, FiddleInitStaticMethod)
    self.assertEqual(1, obj.x)
    self.assertEqual(3, obj.y)
    self.assertEqual(5, obj.z)  # pytype: disable=attribute-error

  def test_fiddle_init_config_static_subclass(self):
    cfg = fdl.Config(FiddleInitStaticMethodChild)
    self.assertEqual(3, cfg.y)
    self.assertEqual(5, cfg.z)
    self.assertEqual(42, cfg.w)
    cfg.x = 1
    obj = fdl.build(cfg)
    self.assertIsInstance(obj, FiddleInitStaticMethodChild)
    self.assertEqual(42, obj.w)  # pytype: disable=attribute-error
    self.assertEqual(1, obj.x)
    self.assertEqual(3, obj.y)
    self.assertEqual(5, cfg.z)  # pytype: disable=attribute-error

  def test_fiddle_init_config_class(self):
    cfg = fdl.Config(FiddleInitClassMethod, 1)
    self.assertEqual(5, cfg.z)
    obj = fdl.build(cfg)
    self.assertIsInstance(obj, FiddleInitClassMethod)
    self.assertEqual(1, obj.x)
    self.assertEqual(3, obj.y)
    self.assertEqual(5, obj.z)  # pytype: disable=attribute-error

  def test_fiddle_init_partial_class(self):
    cfg = fdl.Partial(FiddleInitClassMethod, 1)
    self.assertEqual(5, cfg.z)
    partial = fdl.build(cfg)
    obj = partial(x=1)
    self.assertIsInstance(obj, FiddleInitClassMethod)
    self.assertEqual(1, obj.x)
    self.assertEqual(3, obj.y)
    self.assertEqual(5, obj.z)  # pytype: disable=attribute-error

  def test_fiddle_init_config_subclass(self):
    cfg = fdl.Config(FiddleInitClassMethodChild, 0)
    self.assertEqual(5, cfg.z)
    self.assertEqual(10, cfg.w)
    obj = fdl.build(cfg)
    self.assertIsInstance(obj, FiddleInitClassMethodChild)
    self.assertEqual(0, obj.x)
    self.assertEqual(3, obj.y)
    self.assertEqual(5, obj.z)  # pytype: disable=attribute-error
    self.assertEqual(10, obj.w)  # pytype: disable=attribute-error

  def test_fiddle_init_incompatible_subclass(self):
    cfg = fdl.Config(FiddleInitIncompatibleChild)
    self.assertEqual(6, cfg.b)
    cfg.a = 8
    obj = fdl.build(cfg)
    self.assertIsInstance(obj, FiddleInitIncompatibleChild)
    self.assertEqual(8, obj.a)
    self.assertEqual(9, obj.x)
    self.assertEqual(6, obj.b)
    self.assertEqual(3, obj.y)
    self.assertEqual(14, obj.z)  # pytype: disable=attribute-error

  def test_fiddle_init_incompatible_broken(self):
    with self.assertRaisesRegex(
        TypeError, 'No parameter named.*; valid parameter names: a, b'):
      _ = fdl.Config(FiddleInitIncompatibleChildBroken)

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
        SampleClass, arg1=fdl.Partial(fn_config1), arg2=fn_config1)
    fn_config2 = fdl.Config(
        basic_fn, arg1=fdl.Partial(class_config), arg2=class_config)

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
    fn_config_copy.arg1.arg2 = 'mutated'
    self.assertEqual(fn_config.arg1.arg2, 'mutated')

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
    fn_config_copy.arg1.arg2 = 'mutated'
    self.assertEqual(fn_config.arg1.arg2, 'arg2')

  def test_deep_copy_preserves_instance_sharing(self):
    class_config = fdl.Config(SampleClass, 'arg1', 'arg2')
    fn_config = fdl.Config(basic_fn, arg1=class_config, arg2=class_config)
    self.assertIs(fn_config.arg1, fn_config.arg2)
    fn_config_copy = copy.deepcopy(fn_config)
    self.assertIsNot(fn_config.arg1, fn_config_copy.arg1)
    self.assertIs(fn_config_copy.arg1, fn_config_copy.arg2)

  def test_deep_copy_partials(self):
    class_partial = fdl.Partial(SampleClass, 'arg1', 'arg2')
    fn_config = fdl.Config(
        basic_fn,
        arg1=fdl.Config(class_partial),
        arg2=fdl.Config(class_partial))
    self.assertIsNot(fn_config.arg1, fn_config.arg2)
    fn_config_copy = copy.deepcopy(fn_config)
    self.assertIsNot(fn_config_copy.arg1, fn_config_copy.arg2)
    self.assertIsNot(fn_config.arg1, fn_config_copy.arg1)

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
          SampleClass, arg1=fdl.Partial(fn_config1), arg2=fn_config1)
      fn_config2 = fdl.Config(
          basic_fn, arg1=fdl.Partial(class_config), arg2=class_config)
      return fn_config2

    cfg1 = make_nested_config()
    cfg2 = make_nested_config()
    self.assertEqual(cfg1, cfg2)
    cfg2.arg2.arg1.kwarg1 = 'another value'
    self.assertNotEqual(cfg1, cfg2)
    cfg1.arg2.arg1.kwarg1 = 'another value'
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
    partial = fdl.Partial(cfg)
    self.assertNotEqual(cfg, partial)

    # Compare to a `Config` subclass.
    class ConfigSubClass(fdl.Config):
      pass

    cfg_subclass = ConfigSubClass(cfg)
    self.assertNotEqual(cfg, cfg_subclass)
    # The logic governing how __eq__ is actually invoked from an == comparison
    # actually takes subclassing relationships into account and always calls
    # b.__eq__(a) if isinstance(b, a.__class__), so testing explicitly here.
    self.assertFalse(cfg.__eq__(cfg_subclass))
    self.assertFalse(cfg_subclass.__eq__(cfg))

  def test_equality_classmethods(self):
    cfg_a = fdl.Config(SampleClass.a_classmethod)
    cfg_b = fdl.Config(SampleClass.a_classmethod)
    self.assertEqual(cfg_a, cfg_b)

  def test_default_value_equality(self):
    cfg1 = fdl.Config(SampleClass, 1, 2)
    cfg2 = fdl.Config(SampleClass, 1, 2, None, kwarg2=None)
    self.assertEqual(cfg1, cfg2)

    cfg1 = fdl.Config(basic_fn, 1, 2)
    cfg2 = fdl.Config(basic_fn, 1, 2, None, kwarg2=None)
    self.assertEqual(cfg1, cfg2)

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

  def test_tagging(self):
    cfg = fdl.Config(SampleClass)
    fdl.add_tag(cfg, 'arg1', Tag1)

    self.assertEqual(frozenset([Tag1]), fdl.get_tags(cfg, 'arg1'))
    self.assertEqual(frozenset([]), fdl.get_tags(cfg, 'arg2'))

    fdl.add_tag(cfg, 'arg1', Tag2)
    self.assertEqual(frozenset([Tag1, Tag2]), fdl.get_tags(cfg, 'arg1'))

    fdl.remove_tag(cfg, 'arg1', Tag2)
    self.assertEqual(frozenset([Tag1]), fdl.get_tags(cfg, 'arg1'))

    with self.assertRaisesRegex(ValueError, '.*not set.*'):
      fdl.remove_tag(cfg, 'arg1', Tag2)

  def test_clear_tags(self):
    cfg = fdl.Config(SampleClass)
    fdl.add_tag(cfg, 'arg1', Tag1)
    fdl.add_tag(cfg, 'arg1', Tag2)
    fdl.clear_tags(cfg, 'arg1')
    self.assertEqual(frozenset([]), fdl.get_tags(cfg, 'arg1'))

  def test_set_tags(self):
    cfg = fdl.Config(SampleClass)
    fdl.add_tag(cfg, 'arg1', Tag1)
    fdl.add_tag(cfg, 'arg1', Tag2)
    fdl.set_tags(cfg, 'arg1', {Tag2})
    self.assertEqual(frozenset([Tag2]), fdl.get_tags(cfg, 'arg1'))

  def test_flatten_unflatten_tags(self):
    cfg = fdl.Config(SampleClass)
    fdl.add_tag(cfg, 'arg1', Tag1)
    values, metadata = cfg.__flatten__()
    copied = fdl.Config.__unflatten__(values, metadata)
    fdl.add_tag(copied, 'arg1', Tag2)
    self.assertEqual(frozenset([Tag1]), fdl.get_tags(cfg, 'arg1'))
    self.assertEqual(frozenset([Tag1, Tag2]), fdl.get_tags(copied, 'arg1'))

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

  def test_partial_for_classes(self):
    class_partial = fdl.Partial(SampleClass, 'arg1', 'arg2')
    pytype_extensions.assert_type(class_partial, fdl.Partial[SampleClass])
    partial = fdl.build(class_partial)
    pytype_extensions.assert_type(partial, Callable[..., SampleClass])
    instance = partial()
    pytype_extensions.assert_type(instance, SampleClass)
    self.assertIsInstance(partial, functools.partial)
    self.assertIsInstance(instance, SampleClass)
    self.assertEqual(instance.arg1, 'arg1')
    self.assertEqual(instance.arg2, 'arg2')

    # We can override parameters at the call site.
    instance = partial(arg1='new_arg1', arg2='new_arg2', kwarg1='new_kwarg1')
    self.assertEqual(instance.arg1, 'new_arg1')
    self.assertEqual(instance.arg2, 'new_arg2')
    self.assertEqual(instance.kwarg1, 'new_kwarg1')

  def test_partial_for_functions(self):
    fn_partial = fdl.Partial(basic_fn, 1, kwarg2='kwarg2')
    pytype_extensions.assert_type(fn_partial, fdl.Partial[Dict[str, Any]])
    self.assertEqual(fn_partial.arg1, 1)
    self.assertEqual(fn_partial.kwarg2, 'kwarg2')
    fn_partial.arg1 = 'arg1'
    self.assertEqual(fn_partial.arg1, 'arg1')
    fn_partial.arg2 = 'arg2'
    fn_partial.kwarg1 = 'kwarg1'

    partial = fdl.build(fn_partial)
    pytype_extensions.assert_type(partial, Callable[..., Dict[str, Any]])
    value = partial()
    pytype_extensions.assert_type(value, Dict[str, Any])
    self.assertEqual(value, {
        'arg1': 'arg1',
        'arg2': 'arg2',
        'kwarg1': 'kwarg1',
        'kwarg2': 'kwarg2'
    })

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

  def test_typed_partial(self):
    class_partial = make_typed_partial()
    pytype_extensions.assert_type(class_partial, fdl.Partial[SampleClass])
    partial = fdl.build(class_partial)
    pytype_extensions.assert_type(partial, Callable[..., SampleClass])
    instance = partial(arg2=4)
    self.assertEqual(instance.arg1, 1)
    self.assertEqual(instance.arg2, 4)

  def test_untyped_partial(self):
    class_partial = make_untyped_partial(SampleClass, arg1=2)
    pytype_extensions.assert_type(class_partial, fdl.Partial)
    partial = fdl.build(class_partial)
    pytype_extensions.assert_type(partial, Callable[..., Any])
    instance = partial(arg2=4)
    self.assertEqual(instance.arg1, 2)
    self.assertEqual(instance.arg2, 4)

  def test_convert_partial(self):
    class_partial = fdl.Partial(SampleClass, 'arg1', 'arg2')
    class_config = fdl.Config(class_partial, 'new_arg1', kwarg1='new_kwarg1')
    class_config.arg2 = 'new_arg2'
    self.assertEqual(class_partial.arg2, 'arg2')
    instance = fdl.build(class_config)
    self.assertEqual(instance.arg1, 'new_arg1')
    self.assertEqual(instance.arg2, 'new_arg2')
    self.assertEqual(instance.kwarg1, 'new_kwarg1')

  def test_convert_partial_nested(self):
    class_partial = fdl.Partial(SampleClass, 'arg1', 'arg2')
    class_config = fdl.Config(SampleClass, fdl.Config(class_partial),
                              fdl.Config(class_partial))
    instance = fdl.build(class_config)
    self.assertEqual(instance.arg1.arg1, 'arg1')
    self.assertEqual(instance.arg1.arg2, 'arg2')
    self.assertEqual(instance.arg2.arg1, 'arg1')
    self.assertEqual(instance.arg2.arg2, 'arg2')
    self.assertIsNot(instance.arg1, instance.arg2)

  def test_repr_class_config(self):
    class_config = fdl.Config(SampleClass, 1, 2, kwarg1='kwarg1')
    expected_repr = "<Config[SampleClass(arg1=1, arg2=2, kwarg1='kwarg1')]>"
    self.assertEqual(repr(class_config), expected_repr)

  def test_repr_fn_config(self):
    fn_config = fdl.Config(basic_fn, 1, 2, kwarg1='kwarg1')
    expected_repr = "<Config[basic_fn(arg1=1, arg2=2, kwarg1='kwarg1')]>"
    self.assertEqual(repr(fn_config), expected_repr)

  def test_repr_class_partial(self):
    class_partial = fdl.Partial(SampleClass, 1, 2, kwarg1='kwarg1')
    expected_repr = "<Partial[SampleClass(arg1=1, arg2=2, kwarg1='kwarg1')]>"
    self.assertEqual(repr(class_partial), expected_repr)

  def test_repr_fn_partial(self):
    fn_partial = fdl.Partial(basic_fn, 1, 2, kwarg1='kwarg1')
    expected_repr = "<Partial[basic_fn(arg1=1, arg2=2, kwarg1='kwarg1')]>"
    self.assertEqual(repr(fn_partial), expected_repr)

  def test_repr_varkwargs(self):
    # Note: in the repr, kwarg1 comes before x and y and z because kwarg1 is an
    # explicit keyword parameter, while x, y, and z are **kwargs parameters.
    cfg = fdl.Config(fn_with_var_kwargs, 1, x=2, z=3, y=4, kwarg1=5)
    expected_repr = (
        '<Config[fn_with_var_kwargs(arg1=1, kwarg1=5, x=2, z=3, y=4)]>')
    self.assertEqual(repr(cfg), expected_repr)

  def test_repr_class_tags(self):
    config = fdl.Config(
        SampleClass,
        1,
        kwarg1='kwarg1',
        kwarg2=fdl.Config(
            SampleClass, 'nested value might be large so ' +
            'put tag next to param, not after value.'))
    fdl.add_tag(config, 'arg1', Tag1)
    fdl.add_tag(config, 'arg2', Tag2)
    fdl.add_tag(config, 'kwarg2', Tag1)
    fdl.add_tag(config, 'kwarg2', Tag2)
    expected_repr = (
        '<Config[SampleClass(arg1[#__main__.Tag1]=1, arg2[#__main__.Tag2], ' +
        "kwarg1='kwarg1', kwarg2[#__main__.Tag1, #__main__.Tag2]=" +
        "<Config[SampleClass(arg1='nested value might be large so put tag " +
        "next to param, not after value.')]>)]>")
    self.assertEqual(repr(config), expected_repr)

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
    with self.assertRaisesRegex(TypeError, expected_msg):
      class_config.nonexistent_arg = 'error!'

  def test_nonexistent_var_args_parameter_error(self):
    fn_config = fdl.Config(fn_with_var_args)
    expected_msg = (r'Variadic arguments \(e.g. \*args\) are not supported\.')
    with self.assertRaisesRegex(TypeError, expected_msg):
      fn_config.args = (1, 2, 3)

  def test_unsupported_var_args_error(self):
    expected_msg = (r'Variable positional arguments \(aka `\*args`\) not '
                    r'supported\.')
    with self.assertRaisesRegex(NotImplementedError, expected_msg):
      fdl.Config(fn_with_var_args, 1, 2, 3)

  def test_build_inside_build(self):

    def inner_build(x: int) -> str:
      return str(x)

    def nest_call(x: int) -> str:
      cfg = fdl.Config(inner_build, x)
      return fdl.build(cfg)

    outer = fdl.Config(nest_call, 3)
    expected_msg = r'test_build_inside_build.<locals>.nest_call \(at <root>\)'
    with self.assertRaisesRegex(fdl.BuildError, expected_msg) as e:
      fdl.build(outer)
    expected_msg = r'forbidden to call `fdl.build` inside another `fdl.build`'
    self.assertRegex(str(e.exception.__cause__), expected_msg)

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

  def test_pickling_partial(self):
    cfg = fdl.Partial(SampleClass)
    cfg.arg1 = 'something'
    self.assertEqual(cfg, pickle.loads(pickle.dumps(cfg)))

  def test_pickling_composition(self):
    cfg = fdl.Config(SampleClass, 1, 'abc')
    cfg.kwarg2 = fdl.Partial(SampleClass)
    cfg.kwarg2.arg1 = 'something'
    self.assertEqual(cfg, pickle.loads(pickle.dumps(cfg)))

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
    with self.assertRaisesRegex(fdl.BuildError, '.*SampleClass.*') as e:
      fdl.build(cfg)
    self.assertIs(e.exception.buildable, cfg.arg1)
    self.assertEqual(e.exception.path_from_config_root, '<root>.arg1')
    self.assertIsInstance(e.exception.original_error, TypeError)
    self.assertEqual(e.exception.args, ())
    self.assertEqual(e.exception.kwargs, {'arg1': 1})

  def test_build_raises_exception_on_call(self):

    def raise_error():
      raise ValueError('My fancy exception')

    cfg = fdl.Config(raise_error)
    with self.assertRaisesWithLiteralMatch(
        fdl.BuildError, 'Failed to construct or call ConfigTest.'
        'test_build_raises_exception_on_call.<locals>.raise_error '
        '(at <root>) with arguments\n    args: ()\n    kwargs: {}'):
      fdl.build(cfg)

  def test_build_error_path(self):
    # This will raise an error, because it doesn't have one arg populated.
    sub_cfg = fdl.Config(basic_fn, 1)
    sub_dict = {'a': 0, 'b': 2, 'c': sub_cfg, 'd': 10}
    cfg = fdl.Config(fn_with_var_kwargs, [1, sub_dict])

    with self.assertRaises(fdl.BuildError) as e:
      fdl.build(cfg)

    self.assertEqual(e.exception.path_from_config_root, "<root>.arg1[1]['c']")

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

  def test_update_callable(self):
    cfg = fdl.Config(basic_fn, 1, 'xyz', kwarg1='abc')
    fdl.update_callable(cfg, SampleClass)
    cfg.kwarg2 = '123'
    obj = fdl.build(cfg)
    self.assertIsInstance(obj, SampleClass)
    self.assertEqual(1, obj.arg1)
    self.assertEqual('xyz', obj.arg2)
    self.assertEqual('abc', obj.kwarg1)
    self.assertEqual('123', obj.kwarg2)

  def test_update_callable_invalid_arg(self):
    cfg = fdl.Config(fn_with_var_kwargs, abc='123', xyz='321')
    with self.assertRaisesRegex(TypeError,
                                r"have invalid arguments \['abc', 'xyz'\]"):
      fdl.update_callable(cfg, SampleClass)

  def test_update_callable_new_kwargs(self):
    cfg = fdl.Config(SampleClass)
    cfg.arg1 = 1
    fdl.update_callable(cfg, fn_with_var_kwargs)
    cfg.abc = '123'  # A **kwargs value should now be allowed.
    self.assertEqual({
        'arg1': 1,
        'kwarg1': None,
        'kwargs': {
            'abc': '123'
        }
    }, fdl.build(cfg))

  def test_update_callable_varargs(self):
    cfg = fdl.Config(fn_with_var_kwargs, 1, 2)
    with self.assertRaisesRegex(NotImplementedError,
                                'Variable positional arguments'):
      fdl.update_callable(cfg, fn_with_var_args_and_kwargs)

  def test_assign(self):
    cfg = fdl.Config(fn_with_var_kwargs, 1, 2)
    fdl.assign(cfg, a='a', b='b')
    self.assertEqual({
        'arg1': 1,
        'kwarg1': 2,
        'kwargs': {
            'a': 'a',
            'b': 'b'
        }
    }, fdl.build(cfg))

  def test_assign_wrong_argument(self):
    cfg = fdl.Config(basic_fn)
    with self.assertRaisesRegex(TypeError, 'not_there'):
      fdl.assign(cfg, arg1=1, not_there=2)

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

  def test_config_for_fn_with_special_arg_names(self):
    # The reason that these tests pass is that we use positional-only
    # parameters for self, etc. in functions such as Config.__build__.
    # If we used keyword-or-positional parameters instead, then these
    # tests would fail with a "multiple values for argument" TypeError.

    def f(self, fn_or_cls, buildable=0):
      return self + fn_or_cls + buildable

    cfg = fdl.Config(f)
    cfg.self = 100
    cfg.fn_or_cls = 200
    self.assertEqual(fdl.build(cfg), 300)

    fdl.assign(cfg, buildable=10)  # pytype: disable=wrong-arg-types
    self.assertEqual(fdl.build(cfg), 310)

    cfg2 = fdl.Config(f, self=5, fn_or_cls=1)  # pytype: disable=wrong-arg-types
    self.assertEqual(fdl.build(cfg2), 6)


class OrderedArgumentsTest(absltest.TestCase):

  def test_ordered_arguments(self):
    cfg = fdl.Config(fn_with_var_kwargs)
    cfg.var_kwarg1 = 3
    cfg.kwarg1 = 'hi'
    cfg.arg1 = 4
    cfg.var_kwarg2 = 0
    self.assertEqual(
        list(cfg.__arguments__.items()), [
            ('var_kwarg1', 3),
            ('kwarg1', 'hi'),
            ('arg1', 4),
            ('var_kwarg2', 0),
        ])
    self.assertEqual(
        list(config_lib.ordered_arguments(cfg).items()), [
            ('arg1', 4),
            ('kwarg1', 'hi'),
            ('var_kwarg1', 3),
            ('var_kwarg2', 0),
        ])

    with self.subTest('path element and keys match up'):
      _, metadata = cfg.__flatten__()
      path_elements = cfg.__path_elements__()
      self.assertEqual(
          metadata.argument_names,
          tuple(path_element.name for path_element in path_elements))


if __name__ == '__main__':
  absltest.main()
