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
import pickle
import threading
from typing import Any

from absl.testing import absltest
from fiddle import config
from fiddle import history

import tree


class TestClass:
  arg1: Any
  arg2: Any
  kwarg1: Any
  kwarg2: Any

  def __init__(self, arg1, arg2, kwarg1=None, kwarg2=None):  # pylint: disable=unused-argument
    self.__dict__.update(locals())

  def a_method(self):
    return 4  # A random number (https://xkcd.com/221/)


def test_fn(arg1, arg2, kwarg1=None, kwarg2=None):  # pylint: disable=unused-argument
  return locals()


def test_fn_with_var_args(arg1, *args, kwarg1=None):  # pylint: disable=unused-argument
  return locals()


def test_fn_with_var_kwargs(arg1, kwarg1=None, **kwargs):  # pylint: disable=unused-argument
  return locals()


def test_fn_with_var_args_and_kwargs(arg1, *args, kwarg1=None, **kwargs):  # pylint: disable=unused-argument
  return locals()


def make_typed_config() -> config.Config[TestClass]:
  """Helper function which returns a config.Config whose type is known."""
  return config.Config(TestClass, arg1=1, arg2=2)


def make_untyped_config(arg_to_configure, **kwargs) -> config.Config:
  """Helper function which returns an untyped config.Config."""
  return config.Config(arg_to_configure, **kwargs)


def make_typed_partial() -> config.Partial[TestClass]:
  """Helper function to create a typed Partial instance."""
  return config.Partial(TestClass, arg1=1)


def make_untyped_partial(arg_to_configure, **kwargs) -> config.Partial:
  """Helper function which returns an untyped config.Config."""
  return config.Partial(arg_to_configure, **kwargs)


class ConfigTest(absltest.TestCase):

  def test_config_for_classes(self):
    class_config: config.Config[TestClass] = config.Config(
        TestClass, 1, kwarg2='kwarg2')
    self.assertEqual(class_config.arg1, 1)
    self.assertEqual(class_config.kwarg2, 'kwarg2')
    class_config.arg1 = 'arg1'
    self.assertEqual(class_config.arg1, 'arg1')
    class_config.arg2 = 'arg2'
    class_config.kwarg1 = 'kwarg1'

    instance = config.build(class_config)
    self.assertEqual(instance.arg1, 'arg1')
    self.assertEqual(instance.arg2, 'arg2')
    self.assertEqual(instance.kwarg1, 'kwarg1')
    self.assertEqual(instance.kwarg2, 'kwarg2')

  def test_config_for_functions(self):
    fn_config = config.Config(test_fn, 1, kwarg2='kwarg2')
    self.assertEqual(fn_config.arg1, 1)
    self.assertEqual(fn_config.kwarg2, 'kwarg2')
    fn_config.arg1 = 'arg1'
    self.assertEqual(fn_config.arg1, 'arg1')
    fn_config.arg2 = 'arg2'
    fn_config.kwarg1 = 'kwarg1'

    result = config.build(fn_config)
    self.assertEqual(result, {
        'arg1': 'arg1',
        'arg2': 'arg2',
        'kwarg1': 'kwarg1',
        'kwarg2': 'kwarg2'
    })

  def test_config_for_functions_with_var_args(self):
    fn_config = config.Config(test_fn_with_var_args, 'arg1', kwarg1='kwarg1')
    fn_args = config.build(fn_config)
    self.assertEqual(fn_args['arg1'], 'arg1')
    self.assertEqual(fn_args['kwarg1'], 'kwarg1')

    fn_config.arg1 = 'new_arg1'
    fn_config.kwarg1 = 'new_kwarg1'
    fn_args = config.build(fn_config)
    self.assertEqual(fn_args['arg1'], 'new_arg1')
    self.assertEqual(fn_args['kwarg1'], 'new_kwarg1')

  def test_config_for_functions_with_var_kwargs(self):
    fn_config = config.Config(
        test_fn_with_var_kwargs,
        'arg1',
        kwarg1='kwarg1',
        kwarg2='kwarg2',
        kwarg3='kwarg3')
    fn_args = config.build(fn_config)
    self.assertEqual(fn_args['arg1'], 'arg1')
    self.assertEqual(fn_args['kwarg1'], 'kwarg1')
    self.assertEqual(fn_args['kwargs'], {
        'kwarg2': 'kwarg2',
        'kwarg3': 'kwarg3'
    })

    fn_config.kwarg1 = 'new_kwarg1'
    fn_config.kwarg3 = 'new_kwarg3'
    fn_args = config.build(fn_config)
    self.assertEqual(fn_args['kwarg1'], 'new_kwarg1')
    self.assertEqual(fn_args['kwargs'], {
        'kwarg2': 'kwarg2',
        'kwarg3': 'new_kwarg3'
    })

  def test_config_for_functions_with_var_args_and_kwargs(self):
    fn_config = config.Config(test_fn_with_var_args_and_kwargs, arg1='arg1')
    fn_args = config.build(fn_config)
    self.assertEqual(fn_args['arg1'], 'arg1')

    fn_config.args = 'kwarg_called_arg'
    fn_config.kwargs = 'kwarg_called_kwarg'
    fn_args = config.build(fn_config)
    self.assertEqual(fn_args['kwargs'], {
        'args': 'kwarg_called_arg',
        'kwargs': 'kwarg_called_kwarg'
    })

  def test_config_with_default_args(self):

    def my_func(x: int = 2, y: str = 'abc'):  # pylint: disable=unused-argument
      return locals()

    cfg = config.Config(my_func)
    self.assertEqual(2, cfg.x)
    self.assertEqual('abc', cfg.y)

    obj = config.build(cfg)
    self.assertEqual(2, obj['x'])
    self.assertEqual('abc', obj['y'])

    cfg.y = 'xyz'
    obj = config.build(cfg)
    self.assertEqual(2, obj['x'])
    self.assertEqual('xyz', obj['y'])

  def test_nested_configs(self):
    fn_config1_args = {
        'arg1': 'innermost1',
        'arg2': 'innermost2',
        'kwarg1': 'kw1',
        'kwarg2': 'kw2'
    }
    fn_config1 = config.Config(test_fn, *fn_config1_args.values())

    class_config = config.Config(
        TestClass, arg1=config.Partial(fn_config1), arg2=fn_config1)
    fn_config2 = config.Config(
        test_fn, arg1=config.Partial(class_config), arg2=class_config)

    fn_config2_args = config.build(fn_config2)

    test_class_partialclass = fn_config2_args['arg1']
    self.assertTrue(issubclass(test_class_partialclass, TestClass))

    test_class_instance = test_class_partialclass()
    self.assertEqual(type(test_class_instance), TestClass)
    self.assertEqual(test_class_instance.arg1(), fn_config1_args)
    self.assertEqual(test_class_instance.arg2, fn_config1_args)

    test_class_instance = fn_config2_args['arg2']
    self.assertEqual(type(test_class_instance), TestClass)
    self.assertEqual(test_class_instance.arg1(), fn_config1_args)
    self.assertEqual(test_class_instance.arg2, fn_config1_args)

  def test_instance_sharing(self):
    class_config = config.Config(
        TestClass, 'arg1', 'arg2', kwarg1='kwarg1', kwarg2='kwarg2')
    class_config_copy = copy.copy(class_config)
    class_config_copy.arg1 = 'separate_arg1'

    # Changing copied config parameters doesn't change the original config.
    self.assertEqual(class_config.arg1, 'arg1')

    fn_config = config.Config(test_fn, class_config_copy, {
        'key1': [class_config, class_config],
        'key2': (class_config,)
    })

    memo = {}
    fn_args = config.build(fn_config, memo=memo)
    separate_instance = fn_args['arg1']
    shared_instance = memo[id(class_config)]
    structure = fn_args['arg2']

    self.assertIsNot(shared_instance, separate_instance)
    for leaf in tree.flatten(structure):
      self.assertIs(leaf, shared_instance)

    self.assertEqual(shared_instance.arg1, 'arg1')
    self.assertEqual(separate_instance.arg1, 'separate_arg1')

  def test_shallow_copy(self):
    class_config = config.Config(TestClass, 'arg1', 'arg2')
    fn_config = config.Config(test_fn, class_config, 'fn_arg2')
    fn_config_copy = copy.copy(fn_config)
    # Changing the copy doesn't change the original.
    fn_config_copy.arg2 = 'fn_arg2_copy'
    self.assertEqual(fn_config.arg2, 'fn_arg2')
    # But modifying a shared value is seen by both.
    fn_config_copy.arg1.arg2 = 'mutated'
    self.assertEqual(fn_config.arg1.arg2, 'mutated')

  def test_deep_copy(self):
    class_config = config.Config(TestClass, 'arg1', 'arg2')
    fn_config = config.Config(test_fn, class_config, 'fn_arg2')
    fn_config_copy = copy.deepcopy(fn_config)
    # Changing the copy doesn't change the original.
    fn_config_copy.arg2 = 'fn_arg2_copy'
    self.assertEqual(fn_config.arg2, 'fn_arg2')
    # With a deep copy, the value is no longer shared.
    fn_config_copy.arg1.arg2 = 'mutated'
    self.assertEqual(fn_config.arg1.arg2, 'arg2')

  def test_deep_copy_preserves_instance_sharing(self):
    class_config = config.Config(TestClass, 'arg1', 'arg2')
    fn_config = config.Config(test_fn, arg1=class_config, arg2=class_config)
    self.assertIs(fn_config.arg1, fn_config.arg2)
    fn_config_copy = copy.deepcopy(fn_config)
    self.assertIsNot(fn_config.arg1, fn_config_copy.arg1)
    self.assertIs(fn_config_copy.arg1, fn_config_copy.arg2)

  def test_deep_copy_partials(self):
    class_partial = config.Partial(TestClass, 'arg1', 'arg2')
    fn_config = config.Config(
        test_fn, arg1=class_partial(), arg2=class_partial())
    self.assertIsNot(fn_config.arg1, fn_config.arg2)
    fn_config_copy = copy.deepcopy(fn_config)
    self.assertIsNot(fn_config_copy.arg1, fn_config_copy.arg2)
    self.assertIsNot(fn_config.arg1, fn_config_copy.arg1)

  def test_equality(self):
    cfg1 = config.Config(TestClass, 'arg1')
    cfg2 = config.Config(TestClass, 'arg1')
    self.assertEqual(cfg1, cfg2)
    cfg1.arg1 = 'arg2'
    self.assertNotEqual(cfg1, cfg2)

    partial = config.Partial(TestClass, 'arg1')
    self.assertNotEqual(cfg2, partial)

  def test_memo_override(self):
    class_config = config.Config(
        TestClass, 'arg1', 'arg2', kwarg1='kwarg1', kwarg2='kwarg2')
    # Passing `class_config` as the first arg (`arg1`) to `test_fn`, and then as
    # all the leaves of the dict passed as the second arg (`arg2`). Below, we
    # make sure all the resulting instances really are the same, and are
    # properly overridden via the `memo` dictionary.
    fn_config = config.Config(test_fn, class_config, {
        'key1': [class_config, class_config],
        'key2': (class_config,)
    })

    overridden_instance_value = object()
    memo = {id(class_config): overridden_instance_value}
    fn_args = config.build(fn_config, memo=memo)
    instance = fn_args['arg1']
    structure = fn_args['arg2']

    self.assertIs(instance, overridden_instance_value)
    for leaf in tree.flatten(structure):
      self.assertIs(leaf, overridden_instance_value)

  def test_unsetting_argument(self):
    fn_config = config.Config(test_fn)
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

  def test_dir_simple(self):
    fn_config = config.Config(test_fn)
    self.assertEqual(['arg1', 'arg2', 'kwarg1', 'kwarg2'], dir(fn_config))

  def test_dir_cls(self):
    cfg = config.Config(TestClass)
    self.assertEqual(['arg1', 'arg2', 'kwarg1', 'kwarg2'], dir(cfg))

  def test_dir_var_args_and_kwargs(self):
    varargs_config = config.Config(test_fn_with_var_args_and_kwargs)
    varargs_config.abc = '123'
    self.assertEqual(['abc', 'arg1', 'kwarg1'], dir(varargs_config))

  def test_partial(self):
    class_partial = config.Partial(TestClass, 'arg1', 'arg2')
    partialclass = config.build(class_partial)
    self.assertIsInstance(partialclass, type)
    self.assertTrue(issubclass(partialclass, TestClass))

    instance = partialclass()
    self.assertIsInstance(instance, TestClass)
    self.assertEqual(instance.arg1, 'arg1')  # pytype: disable=attribute-error  # kwargs-checking
    self.assertEqual(instance.arg2, 'arg2')  # pytype: disable=attribute-error  # kwargs-checking

    # We can override parameters at the call site.
    instance = partialclass(
        arg1='new_arg1', arg2='new_arg2', kwarg1='new_kwarg1')
    self.assertEqual(instance.arg1, 'new_arg1')
    self.assertEqual(instance.arg2, 'new_arg2')
    self.assertEqual(instance.kwarg1, 'new_kwarg1')

  def test_typed_config(self):
    class_config = make_typed_config()
    instance = config.build(class_config)
    self.assertEqual(instance.arg1, 1)
    self.assertEqual(instance.arg2, 2)

  def test_untyped_config(self):
    class_config = make_untyped_config(TestClass, arg1=2, arg2=3)
    instance = config.build(class_config)
    self.assertEqual(instance.arg1, 2)
    self.assertEqual(instance.arg2, 3)

  def test_typed_partial(self):
    class_config = make_typed_partial()
    subclass = config.build(class_config)
    instance = subclass(arg2=4)
    self.assertEqual(instance.arg1, 1)
    self.assertEqual(instance.arg2, 4)

  def test_untyped_partial(self):
    class_config = make_untyped_partial(TestClass, arg1=2)
    subclass = config.build(class_config)
    instance = subclass(arg2=4)
    self.assertEqual(instance.arg1, 2)
    self.assertEqual(instance.arg2, 4)

  def test_call_partial(self):
    class_partial = config.Partial(TestClass, 'arg1', 'arg2')
    class_config = class_partial('new_arg1', kwarg1='new_kwarg1')
    class_config.arg2 = 'new_arg2'
    self.assertEqual(class_partial.arg2, 'arg2')
    instance = config.build(class_config)
    self.assertEqual(instance.arg1, 'new_arg1')
    self.assertEqual(instance.arg2, 'new_arg2')
    self.assertEqual(instance.kwarg1, 'new_kwarg1')

  def test_call_partial_nested(self):
    class_partial = config.Partial(TestClass, 'arg1', 'arg2')
    class_config = config.Config(TestClass, class_partial(), class_partial())
    instance = config.build(class_config)
    self.assertEqual(instance.arg1.arg1, 'arg1')
    self.assertEqual(instance.arg1.arg2, 'arg2')
    self.assertEqual(instance.arg2.arg1, 'arg1')
    self.assertEqual(instance.arg2.arg2, 'arg2')
    self.assertIsNot(instance.arg1, instance.arg2)

  def test_repr_class_config(self):
    class_config = config.Config(TestClass, 1, 2, kwarg1='kwarg1')
    expected_repr = "<Config[TestClass(arg1=1, arg2=2, kwarg1='kwarg1')]>"
    self.assertEqual(repr(class_config), expected_repr)

  def test_repr_fn_config(self):
    fn_config = config.Config(test_fn, 1, 2, kwarg1='kwarg1')
    expected_repr = "<Config[test_fn(arg1=1, arg2=2, kwarg1='kwarg1')]>"
    self.assertEqual(repr(fn_config), expected_repr)

  def test_repr_class_partial(self):
    class_partial = config.Partial(TestClass, 1, 2, kwarg1='kwarg1')
    expected_repr = "<Partial[TestClass(arg1=1, arg2=2, kwarg1='kwarg1')]>"
    self.assertEqual(repr(class_partial), expected_repr)

  def test_repr_fn_partial(self):
    fn_partial = config.Partial(test_fn, 1, 2, kwarg1='kwarg1')
    expected_repr = "<Partial[test_fn(arg1=1, arg2=2, kwarg1='kwarg1')]>"
    self.assertEqual(repr(fn_partial), expected_repr)

  def test_nonexistent_attribute_error(self):
    class_config = config.Config(TestClass, 1)
    expected_msg = (r"No parameter 'nonexistent_arg' has been set on "
                    r'<Config\[TestClass\(arg1=1\)\]>\.')
    with self.assertRaisesRegex(AttributeError, expected_msg):
      getattr(class_config, 'nonexistent_arg')

  def test_nonexistent_parameter_error(self):
    class_config = config.Config(TestClass)
    expected_msg = (r"No parameter named 'nonexistent_arg' exists for "
                    r"<class '__main__.TestClass'>; valid parameter names: "
                    r'arg1, arg2, kwarg1, kwarg2\.')
    with self.assertRaisesRegex(TypeError, expected_msg):
      class_config.nonexistent_arg = 'error!'

  def test_nonexistent_var_args_parameter_error(self):
    fn_config = config.Config(test_fn_with_var_args)
    expected_msg = (r'Variadic arguments \(e.g. \*args\) are not supported\.')
    with self.assertRaisesRegex(TypeError, expected_msg):
      fn_config.args = (1, 2, 3)

  def test_unsupported_var_args_error(self):
    expected_msg = (r'Variable positional arguments \(aka `\*args`\) not '
                    r'supported\.')
    with self.assertRaisesRegex(NotImplementedError, expected_msg):
      config.Config(test_fn_with_var_args, 1, 2, 3)

  def test_build_inside_build(self):

    def inner_build(x: int) -> str:
      return str(x)

    def nest_call(x: int) -> str:
      cfg = config.Config(inner_build, x)
      return config.build(cfg)

    outer = config.Config(nest_call, 3)
    expected_msg = r'test_build_inside_build.<locals>.nest_call \(at <root>\)'
    with self.assertRaisesRegex(config.BuildError, expected_msg) as e:
      config.build(outer)
    expected_msg = r'forbidden to call `fdl.build` inside another `fdl.build`'
    self.assertRegex(str(e.exception.__cause__), expected_msg)

  def test_history_tracking(self):
    cfg = config.Config(TestClass, 'arg1_value')
    cfg.arg2 = 'arg2_value'
    del cfg.arg1

    self.assertEqual(
        set(['arg1', 'arg2']), set(cfg.__argument_history__.keys()))
    self.assertLen(cfg.__argument_history__['arg1'], 2)
    self.assertLen(cfg.__argument_history__['arg2'], 1)
    self.assertEqual('arg1', cfg.__argument_history__['arg1'][0].param_name)
    self.assertEqual('arg1_value', cfg.__argument_history__['arg1'][0].value)
    self.assertRegex(
        str(cfg.__argument_history__['arg1'][0].location),
        r'config_test.py:\d+:test_history_tracking')
    self.assertEqual('arg2', cfg.__argument_history__['arg2'][0].param_name)
    self.assertEqual('arg2_value', cfg.__argument_history__['arg2'][0].value)
    self.assertRegex(
        str(cfg.__argument_history__['arg2'][0].location),
        r'config_test.py:\d+:test_history_tracking')
    self.assertEqual('arg1', cfg.__argument_history__['arg1'][1].param_name)
    self.assertEqual(history.DELETED, cfg.__argument_history__['arg1'][1].value)
    self.assertRegex(
        str(cfg.__argument_history__['arg1'][1].location),
        r'config_test.py:\d+:test_history_tracking')

    self.assertEqual(cfg.__argument_history__['arg1'][0].sequence_id + 1,
                     cfg.__argument_history__['arg2'][0].sequence_id)
    self.assertEqual(cfg.__argument_history__['arg2'][0].sequence_id + 1,
                     cfg.__argument_history__['arg1'][1].sequence_id)

  def test_custom_location_history_tracking(self):
    with history.custom_location(lambda: 'abc:123'):
      cfg = config.Config(TestClass, 'arg1')
    cfg.arg2 = 'arg2'
    self.assertEqual(
        set(['arg1', 'arg2']), set(cfg.__argument_history__.keys()))
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

    cfg = config.Config(TestClass)
    expected_msg = 'a_method.*Note: .*TestClass has an attribute/method with '
    with self.assertRaisesRegex(AttributeError, expected_msg):
      cfg.a_method()
    with self.assertRaisesRegex(AttributeError, expected_msg):
      _ = cfg.a_method

  def test_unhashable(self):
    """All Buildable's should be unhashable: mutability and custom __eq__."""
    with self.assertRaisesRegex(TypeError, 'unhashable'):
      _ = config.Config(TestClass) in {}
    with self.assertRaisesRegex(TypeError, 'unhashable'):
      _ = config.Partial(TestClass) in {}

  def test_pickling_config(self):
    """Bulidable types should be pickle-able."""
    cfg = config.Config(TestClass, 1, 'abc')
    self.assertEqual(cfg, pickle.loads(pickle.dumps(cfg)))
    reloaded = pickle.loads(pickle.dumps(cfg))
    reloaded.kwarg1 = 3  # mutate after unpickling.
    self.assertNotEqual(cfg, reloaded)

  def test_pickling_partial(self):
    cfg = config.Partial(TestClass)
    cfg.arg1 = 'something'
    self.assertEqual(cfg, pickle.loads(pickle.dumps(cfg)))

  def test_pickling_composition(self):
    cfg = config.Config(TestClass, 1, 'abc')
    cfg.kwarg2 = config.Partial(TestClass)
    cfg.kwarg2.arg1 = 'something'
    self.assertEqual(cfg, pickle.loads(pickle.dumps(cfg)))

  def test_build_raises_nice_error_too_few_args(self):
    cfg = config.Config(test_fn, config.Config(TestClass, 1), 2)
    with self.assertRaisesRegex(config.BuildError, '.*TestClass.*') as e:
      config.build(cfg)
    self.assertIs(e.exception.buildable, cfg.arg1)
    self.assertEqual(e.exception.path_from_config_root, '<root>.arg1')
    self.assertIsInstance(e.exception.original_error, TypeError)
    self.assertEqual(e.exception.args, ())
    self.assertEqual(e.exception.kwargs, {'arg1': 1})

  def test_build_raises_exception_on_call(self):

    def raise_error():
      raise ValueError('My fancy exception')

    cfg = config.Config(raise_error)
    with self.assertRaisesWithLiteralMatch(
        config.BuildError, 'Failed to construct or call ConfigTest.'
        'test_build_raises_exception_on_call.<locals>.raise_error '
        '(at <root>) with arguments\n    args: ()\n    kwargs: {}'):
      config.build(cfg)

  def test_build_error_path(self):
    # This will raise an error, because it doesn't have one arg populated.
    sub_cfg = config.Config(test_fn, 1)
    sub_dict = {'a': 0, 'b': 2, 'c': sub_cfg, 'd': 10}
    cfg = config.Config(test_fn_with_var_kwargs, [1, sub_dict])

    with self.assertRaises(config.BuildError) as e:
      config.build(cfg)

    self.assertEqual(e.exception.path_from_config_root, "<root>.arg1[1]['c']")

  def test_multithreaded_build(self):
    """Two threads can each invoke config.build without interfering."""
    output = None
    background_entered_build = threading.Event()
    foreground_entered_build = threading.Event()

    def other_thread():
      nonlocal output

      def blocking_function(x):
        background_entered_build.set()
        foreground_entered_build.wait()
        return x

      cfg = config.Config(blocking_function, 3)
      output = config.build(cfg)

    def blocking_function(x):
      foreground_entered_build.set()
      background_entered_build.wait()
      return x

    cfg = config.Config(blocking_function, 1)
    thread = threading.Thread(target=other_thread)
    thread.start()
    obj = config.build(cfg)
    thread.join()
    self.assertEqual(1, obj)
    self.assertEqual(3, output)


if __name__ == '__main__':
  absltest.main()
