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

from fiddle import config
from fiddle.experimental import auto_config
from fiddle.experimental import autobuilders as ab


def test_fn(arg, kwarg='test'):
  return {'arg': arg, 'kwarg': kwarg}


def test_fn_with_kwargs(**kwargs):
  return {'kwargs': kwargs}


@dataclasses.dataclass(frozen=True)
class TestClass:
  arg1: Any
  arg2: Any

  def method(self):
    pass


def explicit_config_building_fn(x: int) -> config.Config:
  return config.Config(test_fn, 5, kwarg=x)


def pass_through(arg):
  return arg


def _line_number():
  return sys._getframe(1).f_lineno


class AutoConfigTest(parameterized.TestCase):

  def test_create_basic_config(self):
    expected_config = config.Config(TestClass, 1, arg2=2)

    @auto_config.auto_config
    def test_class_config():
      return TestClass(1, 2)

    self.assertEqual(expected_config, test_class_config.as_buildable())
    self.assertEqual(TestClass(1, 2), test_class_config())

  def test_create_basic_config_parents(self):
    expected_config = config.Config(TestClass, 1, arg2=2)

    @auto_config.auto_config()  # Note the parenthesis!
    def test_class_config():
      return TestClass(1, 2)

    self.assertEqual(expected_config, test_class_config.as_buildable())
    self.assertEqual(TestClass(1, 2), test_class_config())

  def test_create_basic_partial(self):
    expected_config = config.Partial(test_fn, 1, kwarg='kwarg')

    @auto_config.auto_config
    def test_fn_config():
      return functools.partial(test_fn, 1, 'kwarg')

    self.assertEqual(expected_config, test_fn_config.as_buildable())

  def test_create_config_with_args(self):
    expected_config = config.Config(TestClass, 'positional', arg2='default')

    @auto_config.auto_config
    def test_class_config(arg1, arg2='default'):
      return TestClass(arg1, arg2)

    self.assertEqual(expected_config,
                     test_class_config.as_buildable('positional'))
    self.assertEqual(
        TestClass('positional', 'default'), test_class_config('positional'))

  def test_create_config_with_kwonly_args(self):
    expected_config = config.Config(TestClass, 'positional', arg2='default')

    @auto_config.auto_config
    def test_class_config(arg1, *, arg2='default'):
      return TestClass(arg1, arg2)

    self.assertEqual(expected_config,
                     test_class_config.as_buildable('positional'))
    self.assertEqual(
        TestClass('positional', 'default'), test_class_config('positional'))

  def test_calling_auto_config(self):
    expected_config = config.Config(
        test_fn, 1, kwarg=config.Config(TestClass, 1, 2))

    @auto_config.auto_config
    def test_class_config():
      return TestClass(1, arg2=2)

    @auto_config.auto_config
    def test_fn_config():
      return test_fn(1, test_class_config())

    self.assertEqual(expected_config, test_fn_config.as_buildable())
    self.assertEqual({
        'arg': 1,
        'kwarg': TestClass(1, arg2=2)
    }, test_fn_config())

  def test_nested_calls(self):
    expected_config = config.Config(
        TestClass, 1, arg2=config.Config(test_fn, 2, 'kwarg'))

    @auto_config.auto_config
    def test_class_config():
      return TestClass(1, test_fn(2, 'kwarg'))

    self.assertEqual(expected_config, test_class_config.as_buildable())
    self.assertEqual(
        TestClass(1, {
            'arg': 2,
            'kwarg': 'kwarg'
        }), test_class_config())

  def test_calling_explicit_function(self):
    expected_config = config.Config(
        TestClass, 1, arg2=config.Config(test_fn, 5, 10))

    @auto_config.auto_config
    def test_nested_call():
      return TestClass(1, explicit_config_building_fn(10))

    self.assertEqual(expected_config, test_nested_call.as_buildable())

  def test_reference_nonlocal(self):
    nonlocal_var = 3
    expected_config = config.Config(test_fn, 1, kwarg=nonlocal_var)

    @auto_config.auto_config
    def test_fn_config():
      return test_fn(1, nonlocal_var)

    self.assertEqual(expected_config, test_fn_config.as_buildable())

  def test_calling_builtins(self):
    expected_config = config.Config(TestClass, [0, 1, 2], ['a', 'b'])

    @auto_config.auto_config
    def test_config():
      return TestClass(list(range(3)), list({'a': 0, 'b': 1}.keys()))

    self.assertEqual(expected_config, test_config.as_buildable())

  def test_auto_config_eligibility(self):
    # Some common types that have no signature.
    for builtin_type in (range, dict):
      self.assertFalse(auto_config._is_auto_config_eligible(builtin_type))
    # Some common builtins.
    for builtin in (list, tuple, sum, any, all):
      self.assertFalse(auto_config._is_auto_config_eligible(builtin))
    # A method.
    test_class = TestClass(1, 2)
    self.assertFalse(auto_config._is_auto_config_eligible(test_class.method))
    # Buildable subclasses.
    self.assertFalse(auto_config._is_auto_config_eligible(config.Config))
    self.assertFalse(auto_config._is_auto_config_eligible(config.Partial))
    # Auto-config annotations are not eligible, and this case is tested in
    # `test_calling_auto_config` above.

  def test_autobuilders_in_auto_config(self):
    expected_config = config.Config(
        test_fn, arg=ab.config(TestClass, require_skeleton=False))

    @auto_config.auto_config
    def autobuilder_using_fn():
      x = ab.config(TestClass, require_skeleton=False)
      return test_fn(x)

    self.assertEqual(expected_config, autobuilder_using_fn.as_buildable())

  def test_auto_configuring_non_function(self):
    with self.assertRaisesRegex(ValueError, 'only compatible with functions'):
      auto_config.auto_config(3)

  def test_control_flow_if(self):

    def test_config(condition):
      if condition:
        return pass_through('true branch')
      else:
        return pass_through('false branch')

    error_line_number = _line_number() - 5

    exception_type = auto_config.UnsupportedControlFlowError
    msg = (r'Control flow \(If\) is unsupported by auto_config\. '
           r'\(auto_config_test\.py, line \d+\)')
    with self.assertRaisesRegex(exception_type, msg) as context:
      auto_config.auto_config(test_config)
    self.assertEqual(error_line_number, context.exception.lineno)
    # The text and offset here don't quite match with the indentation level of
    # test_config above, due to the `dedent` performed inside `auto_config`.
    # This slight discrepancy is unlikely to cause any confusion.
    self.assertEqual('  if condition:', context.exception.text)
    self.assertEqual(2, context.exception.offset)

    build_config_fn = auto_config.auto_config(
        test_config, experimental_allow_control_flow=True).as_buildable
    self.assertEqual('true branch', config.build(build_config_fn(True)))
    self.assertEqual('false branch', config.build(build_config_fn(False)))

  def test_control_flow_for(self):

    def test_config():
      layers = []
      for i in range(3):
        layers.append(TestClass(i, i))
      return pass_through(layers)

    msg = (r'Control flow \(For\) is unsupported by auto_config\. '
           r'\(auto_config_test\.py, line \d+\)')
    with self.assertRaisesRegex(auto_config.UnsupportedControlFlowError, msg):
      auto_config.auto_config(test_config)

    expected_config = config.Config(
        pass_through, [config.Config(TestClass, i, i) for i in range(3)])
    actual_config = auto_config.auto_config(
        test_config, experimental_allow_control_flow=True).as_buildable()
    self.assertEqual(expected_config, actual_config)

  def test_control_flow_while(self):

    def test_config():
      i = 10
      while i > 0:
        i -= 1
      return pass_through(i)

    msg = (r'Control flow \(While\) is unsupported by auto_config\. '
           r'\(auto_config_test\.py, line \d+\)')
    with self.assertRaisesRegex(auto_config.UnsupportedControlFlowError, msg):
      auto_config.auto_config(test_config)

    pass_through_config = auto_config.auto_config(
        test_config, experimental_allow_control_flow=True).as_buildable()
    self.assertEqual(0, config.build(pass_through_config))

  @parameterized.parameters(
      (lambda: pass_through([i + 1 for i in [0, 1, 2]]), 'ListComp'),
      (lambda: pass_through({i + 1 for i in [0, 1, 2]}), 'SetComp'),
      (lambda: pass_through({i + 1: None for i in [0, 1, 2]}), 'DictComp'),
      (lambda: pass_through((i + 1 for i in [0, 1, 2])), 'GeneratorExp'),
      (lambda: pass_through(None if list() else [1, 2, 3]), 'IfExp'),
  )
  def test_control_flow_expression(self, test_config, node_name):
    msg = (rf'Control flow \({node_name}\) is unsupported by auto_config\. '
           r'\(auto_config_test\.py, line \d+\)')
    with self.assertRaisesRegex(auto_config.UnsupportedControlFlowError, msg):
      auto_config.auto_config(test_config)

    pass_through_config = auto_config.auto_config(
        test_config, experimental_allow_control_flow=True).as_buildable()
    self.assertCountEqual([1, 2, 3], config.build(pass_through_config))

  def test_allow_control_flow_decorator(self):

    @auto_config.auto_config(experimental_allow_control_flow=True)
    def test_config():
      return pass_through([i + 1 for i in range(3)])

    self.assertEqual([1, 2, 3], config.build(test_config.as_buildable()))


if __name__ == '__main__':
  absltest.main()
