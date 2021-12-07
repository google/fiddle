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
from typing import Any

from absl.testing import absltest

from fiddle import config
from fiddle.experimental import auto_config


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


class AutoConfigTest(absltest.TestCase):

  def test_create_basic_config(self):
    expected_config = config.Config(TestClass, 1, arg2=2)

    @auto_config.auto_config
    def test_class_config():
      return TestClass(1, 2)

    self.assertEqual(expected_config, test_class_config())

  def test_create_basic_partial(self):
    expected_config = config.Partial(test_fn, 1, kwarg='kwarg')

    @auto_config.auto_config
    def test_fn_config():
      return functools.partial(test_fn, 1, 'kwarg')

    self.assertEqual(expected_config, test_fn_config())

  def test_create_config_with_args(self):
    expected_config = config.Config(TestClass, 'positional', arg2='default')

    @auto_config.auto_config
    def test_class_config(arg1, arg2='default'):
      return TestClass(arg1, arg2)

    self.assertEqual(expected_config, test_class_config('positional'))

  def test_create_config_with_kwonly_args(self):
    expected_config = config.Config(TestClass, 'positional', arg2='default')

    @auto_config.auto_config
    def test_class_config(arg1, *, arg2='default'):
      return TestClass(arg1, arg2)

    self.assertEqual(expected_config, test_class_config('positional'))

  def test_calling_auto_config(self):
    expected_config = config.Config(
        test_fn, 1, kwarg=config.Config(TestClass, 1, 2))

    @auto_config.auto_config
    def test_class_config():
      return TestClass(1, arg2=2)

    @auto_config.auto_config
    def test_fn_config():
      return test_fn(1, test_class_config())

    self.assertEqual(expected_config, test_fn_config())

  def test_nested_calls(self):
    expected_config = config.Config(
        TestClass, 1, arg2=config.Config(test_fn, 2, 'kwarg'))

    @auto_config.auto_config
    def test_class_config():
      return TestClass(1, test_fn(2, 'kwarg'))

    return self.assertEqual(expected_config, test_class_config())

  def test_reference_nonlocal(self):
    nonlocal_var = 3
    expected_config = config.Config(test_fn, 1, kwarg=nonlocal_var)

    @auto_config.auto_config
    def test_fn_config():
      return test_fn(1, nonlocal_var)

    self.assertEqual(expected_config, test_fn_config())

  def test_calling_builtins(self):
    expected_config = [
        config.Config(test_fn_with_kwargs, value=i) for i in range(3)
    ]

    @auto_config.auto_config
    def test_config():
      mapping = {'a': 0, 'b': 1, 'c': 2}
      return [test_fn_with_kwargs(value=value) for value in mapping.values()]

    return self.assertEqual(expected_config, test_config())

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


if __name__ == '__main__':
  absltest.main()
