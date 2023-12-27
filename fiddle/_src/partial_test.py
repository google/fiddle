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
import logging
import pickle
from typing import Any, Callable, Dict

from absl.testing import absltest
from absl.testing import parameterized
import fiddle as fdl
from fiddle import arg_factory
import pytype_extensions


class Tag1(fdl.Tag):
  """One tag."""


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


def make_typed_partial() -> fdl.Partial[SampleClass]:
  """Helper function to create a typed Partial instance."""
  return fdl.Partial(SampleClass, arg1=1)


def make_untyped_partial(arg_to_configure, **kwargs) -> fdl.Partial:
  """Helper function which returns an untyped fdl.Config."""
  return fdl.Partial(arg_to_configure, **kwargs)


@dataclasses.dataclass
class DataclassChild:
  x: int = 0


class PartialTest(parameterized.TestCase):

  def test_deep_copy_partials(self):
    class_partial = fdl.Partial(SampleClass, 'arg1', 'arg2')
    fn_config = fdl.Config(
        basic_fn,
        arg1=fdl.cast(fdl.Config, class_partial),
        arg2=fdl.cast(fdl.Config, class_partial),
    )
    self.assertIsNot(fn_config.arg1, fn_config.arg2)
    fn_config_copy = copy.deepcopy(fn_config)
    self.assertIsNot(fn_config_copy.arg1, fn_config_copy.arg2)
    self.assertIsNot(fn_config.arg1, fn_config_copy.arg1)

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
    pytype_extensions.assert_type(
        fn_partial, fdl.Partial[Dict[str, Any]]
    )  # use-fiddle-overlay
    self.assertEqual(fn_partial.arg1, 1)
    self.assertEqual(fn_partial.kwarg2, 'kwarg2')
    fn_partial.arg1 = 'arg1'
    self.assertEqual(fn_partial.arg1, 'arg1')
    fn_partial.arg2 = 'arg2'
    fn_partial.kwarg1 = 'kwarg1'

    partial = fdl.build(fn_partial)
    pytype_extensions.assert_type(
        partial, Callable[..., Dict[str, Any]]
    )  # use-fiddle-overlay
    value = partial()
    pytype_extensions.assert_type(value, Dict[str, Any])  # use-fiddle-overlay
    self.assertEqual(
        value,
        {
            'arg1': 'arg1',
            'arg2': 'arg2',
            'kwarg1': 'kwarg1',
            'kwarg2': 'kwarg2',
        },
    )

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
    class_config = fdl.cast(fdl.Config, class_partial)
    class_config.arg1 = 'new_arg1'
    class_config.arg2 = 'new_arg2'
    class_config.kwarg1 = 'new_kwarg1'
    self.assertEqual(class_partial.arg2, 'arg2')
    instance = fdl.build(class_config)
    self.assertEqual(instance.arg1, 'new_arg1')
    self.assertEqual(instance.arg2, 'new_arg2')
    self.assertEqual(instance.kwarg1, 'new_kwarg1')

  def test_convert_partial_nested(self):
    class_partial = fdl.Partial(SampleClass, 'arg1', 'arg2')
    class_config = fdl.Config(
        SampleClass,
        fdl.cast(fdl.Config, class_partial),
        fdl.cast(fdl.Config, class_partial),
    )
    instance = fdl.build(class_config)
    self.assertEqual(instance.arg1.arg1, 'arg1')
    self.assertEqual(instance.arg1.arg2, 'arg2')
    self.assertEqual(instance.arg2.arg1, 'arg1')
    self.assertEqual(instance.arg2.arg2, 'arg2')
    self.assertIsNot(instance.arg1, instance.arg2)

  def test_repr_class_partial(self):
    class_partial = fdl.Partial(SampleClass, 1, 2, kwarg1='kwarg1')
    expected_repr = "<Partial[SampleClass(arg1=1, arg2=2, kwarg1='kwarg1')]>"
    self.assertEqual(repr(class_partial), expected_repr)

  def test_repr_fn_partial(self):
    fn_partial = fdl.Partial(basic_fn, 1, 2, kwarg1='kwarg1')
    expected_repr = "<Partial[basic_fn(arg1=1, arg2=2, kwarg1='kwarg1')]>"
    self.assertEqual(repr(fn_partial), expected_repr)

  def test_pickling_partial(self):
    cfg = fdl.Partial(SampleClass)
    cfg.arg1 = 'something'
    self.assertEqual(cfg, pickle.loads(pickle.dumps(cfg)))

  def test_pickling_composition(self):
    cfg = fdl.Config(SampleClass, 1, 'abc')
    cfg.kwarg2 = fdl.Partial(SampleClass)
    cfg.kwarg2.arg1 = 'something'
    self.assertEqual(cfg, pickle.loads(pickle.dumps(cfg)))

  @parameterized.product(
      from_type=[fdl.Config, fdl.Partial, fdl.ArgFactory],
      to_type=[fdl.Config, fdl.Partial, fdl.ArgFactory],
  )
  def test_cast(self, from_type, to_type):
    cfg1 = from_type(fn_with_var_kwargs, 1, 2)
    fdl.add_tag(cfg1, 'arg1', Tag1)
    cfg2 = fdl.cast(to_type, cfg1)
    self.assertIsInstance(cfg2, to_type)
    self.assertEqual(fdl.get_callable(cfg1), fdl.get_callable(cfg2))
    self.assertEqual(cfg1.__arguments__, cfg2.__arguments__)
    self.assertEqual(cfg1.__argument_tags__, cfg2.__argument_tags__)

  def test_no_warnings_on_casting_partial_to_config(self):
    cfg = fdl.Partial(SampleClass)
    # Assert that no logs of WARNING or higher are triggered when casting a
    # Partial to a Config.
    # NOTE: `self.assertNoLogs` only exists for Python>=3.10, and OSS
    # supports >=3.8. When PY3.10 is available, use `self.assertNoLogs`.
    with self.assertRaisesRegex(
        AssertionError, 'no logs of level WARNING or higher triggered'
    ):
      with self.assertLogs(level=logging.WARNING) as cm:
        fdl.cast(fdl.Config, cfg)
        self.assertEqual(cm.output, [])


class ArgFactoryTest(absltest.TestCase):

  def test_build_argfactory(self):
    """Build an ArgFactory(...)."""
    cfg = fdl.ArgFactory(SampleClass, arg1=5)
    value = fdl.build(cfg)
    self.assertIsInstance(value.factory, functools.partial)
    self.assertIs(value.factory.func, SampleClass)
    self.assertEqual(value.factory.keywords, dict(arg1=5))
    self.assertEqual(value.factory.args, ())

  def test_build_partial_argfactory(self):
    """Build a Partial(..., ArgFactory(...))."""
    cfg = fdl.Partial(basic_fn, fdl.ArgFactory(DataclassChild))
    partial_fn = fdl.build(cfg)
    self.assertTrue(arg_factory.is_arg_factory_partial(partial_fn))
    self.assertPartialsEqual(
        partial_fn, arg_factory.partial(basic_fn, arg1=DataclassChild)
    )

    # Run the partial twice, and check for shared values.
    v1 = partial_fn(arg2=3)
    v2 = partial_fn(arg2=3)
    self.assertEqual(v1, v2)
    self.assertEqual(
        v1, dict(arg1=DataclassChild(), arg2=3, kwarg1=None, kwarg2=None)
    )
    self.assertIsNot(v1['arg1'], v2['arg1'])

  def test_build_partial_argfactory_argfactory(self):
    """Build a Partial(..., ArgFactory(..., ArgFactory(...)))."""
    cfg = fdl.Partial(
        basic_fn, fdl.ArgFactory(basic_fn, fdl.ArgFactory(DataclassChild))
    )
    cfg.arg1.arg2 = 2
    cfg.arg2 = 3
    partial_fn = fdl.build(cfg)
    self.assertTrue(arg_factory.is_arg_factory_partial(partial_fn))
    self.assertPartialsEqual(
        partial_fn,
        functools.partial(
            arg_factory.partial(
                basic_fn,
                arg1=functools.partial(
                    arg_factory.partial(basic_fn, arg1=DataclassChild), arg2=2
                ),
            ),
            arg2=3,
        ),
    )

    # Run the partial twice, and check for shared values.
    v1 = partial_fn()
    v2 = partial_fn()
    self.assertEqual(v1, v2)
    self.assertEqual(
        v1,
        dict(
            arg1=dict(arg1=DataclassChild(), arg2=2, kwarg1=None, kwarg2=None),
            arg2=3,
            kwarg1=None,
            kwarg2=None,
        ),
    )
    self.assertIsInstance(v1['arg1'], dict)
    self.assertIsInstance(v1['arg1']['arg1'], DataclassChild)

    # Neither the dict nor the DataclassChild is shared.
    self.assertIsNot(v1['arg1'], v2['arg1'])
    self.assertIsNot(v1['arg1']['arg1'], v2['arg1']['arg1'])

  def test_build_partial_argfactory_config(self):
    """Build a Partial(..., ArgFactory(..., Config(...)))."""
    cfg = fdl.Partial(
        basic_fn, fdl.ArgFactory(basic_fn, fdl.Config(DataclassChild))
    )
    cfg.arg1.arg2 = 2
    cfg.arg2 = 3
    partial_fn = fdl.build(cfg)
    self.assertTrue(arg_factory.is_arg_factory_partial(partial_fn))
    self.assertPartialsEqual(
        partial_fn,
        functools.partial(
            arg_factory.partial(
                basic_fn,
                arg1=functools.partial(basic_fn, arg1=DataclassChild(), arg2=2),
            ),
            arg2=3,
        ),
    )

    # Run the partial twice, and check for shared values.
    v1 = partial_fn()
    v2 = partial_fn()
    self.assertEqual(v1, v2)
    self.assertEqual(
        v1,
        dict(
            arg1=dict(arg1=DataclassChild(), arg2=2, kwarg1=None, kwarg2=None),
            arg2=3,
            kwarg1=None,
            kwarg2=None,
        ),
    )
    self.assertIsInstance(v1['arg1'], dict)
    self.assertIsInstance(v1['arg1']['arg1'], DataclassChild)

    # The dict is not shared, but the DataclassChild *is* shared.
    self.assertIsNot(v1['arg1'], v2['arg1'])
    self.assertIs(v1['arg1']['arg1'], v2['arg1']['arg1'])

  def test_build_partial_tree_argfactory(self):
    """Build a Partial(..., Tree(..., ArgFactory(...)))."""
    cfg = fdl.Partial(
        basic_fn,
        {
            'x': [[fdl.Config(object)], fdl.ArgFactory(DataclassChild)],
            'y': [1, 2, 3],
        },
    )
    partial_fn = fdl.build(cfg)
    self.assertTrue(arg_factory.is_arg_factory_partial(partial_fn))

    # Run the partial twice, and check for shared values.
    v1 = partial_fn(arg2=3)
    v2 = partial_fn(arg2=3)
    self.assertEqual(v1, v2)
    obj = v1['arg1']['x'][0][0]  # created from fdl.Config(object)
    self.assertEqual(
        v1,
        dict(
            arg1={'x': [[obj], DataclassChild()], 'y': [1, 2, 3]},
            arg2=3,
            kwarg1=None,
            kwarg2=None,
        ),
    )
    # The DataclassChild and its ancestors are not shared:
    self.assertIsNot(v1['arg1'], v2['arg1'])
    self.assertIsNot(v1['arg1']['x'], v2['arg1']['x'])
    self.assertIsNot(v1['arg1']['x'][1], v2['arg1']['x'][1])
    # But other parts of the tree (including the object created
    # by fdl.Config) are shared:
    self.assertIs(v1['arg1']['x'][0], v2['arg1']['x'][0])
    self.assertIs(v1['arg1']['x'][0][0], obj)
    self.assertIs(v2['arg1']['x'][0][0], obj)
    self.assertIs(v1['arg1']['y'], v2['arg1']['y'])

  def test_build_varargs(self):
    config = fdl.Partial(fn_with_var_args)
    list_factory = fdl.build(fdl.ArgFactory(list))

    # Call __build__ with positional args.  (Note: fdl.build only calls
    # __build__ with keyword args; so we need to use __call__ directly to
    # test this.)
    partial_fn = config.__build__(list_factory, 2, [], list_factory, 5)

    # Since we called __build__ with positional arguments that alternate back
    # and forth between values and factories, it will need to
    # construct a nested partial object with the following structure.  (Note:
    # functools.partial automatically merges when its `func` is a partial, so
    # this ends up being 2 partial objects, not 4.)
    expected = functools.partial(
        arg_factory.partial(
            functools.partial(
                arg_factory.partial(fn_with_var_args, list), 2, []
            ),
            list,
        ),
        5,
    )
    self.assertPartialsEqual(expected, partial_fn)

    v1 = partial_fn()
    v2 = partial_fn()
    self.assertEqual(v1, {'arg1': [], 'args': (2, [], [], 5), 'kwarg1': None})
    self.assertEqual(v2, {'arg1': [], 'args': (2, [], [], 5), 'kwarg1': None})
    self.assertIsNot(v1['arg1'], v2['arg1'])  # from list_factory
    self.assertIs(v1['args'][1], v2['args'][1])  # literal list
    self.assertIsNot(v1['args'][2], v2['args'][2])  # from list_factory

    v3 = partial_fn(10, kwarg1=20)
    self.assertEqual(v3, {'arg1': [], 'args': (2, [], [], 5, 10), 'kwarg1': 20})

  def assertPartialsEqual(self, x, y):
    # Compare using reprs, since `==` will consider different instances of
    # functools.partial to be different even if they have the same function
    # and args.
    self.assertEqual(repr(x), repr(y))


if __name__ == '__main__':
  absltest.main()
