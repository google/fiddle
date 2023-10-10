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

"""Tests for mutate_buildable."""

import dataclasses
from typing import Any

from absl.testing import absltest
import fiddle as fdl
from fiddle._src import building
from fiddle._src import config as config_lib
from fiddle._src import config_test
from fiddle._src import mutate_buildable


@dataclasses.dataclass
class SampleClass:
  arg1: Any


class MutateBuildableTest(absltest.TestCase):

  def test_move_buildable_internals_history(self):
    source = fdl.Config(SampleClass)
    source.arg1 = 4
    source.arg1 = 5
    destination = fdl.Config(SampleClass)
    mutate_buildable.move_buildable_internals(
        source=source, destination=destination)
    self.assertEqual(destination.arg1, 5)
    self.assertEqual(destination.__argument_history__['arg1'][0].new_value, 4)
    self.assertEqual(destination.__argument_history__['arg1'][1].new_value, 5)

  def test_no_unexpected_attributes(self):
    sample_config = fdl.Config(SampleClass)
    expected = mutate_buildable._buildable_internals_keys
    self.assertEqual(set(expected), set(sample_config.__dict__.keys()))


class CallableApisTest(absltest.TestCase):

  def test_update_callable(self):
    cfg = config_lib.Config(config_test.basic_fn, 1, 'xyz', kwarg1='abc')
    mutate_buildable.update_callable(cfg, config_test.SampleClass)
    cfg.kwarg2 = '123'
    obj = building.build(cfg)
    self.assertIsInstance(obj, config_test.SampleClass)
    self.assertEqual(1, obj.arg1)
    self.assertEqual('xyz', obj.arg2)
    self.assertEqual('abc', obj.kwarg1)
    self.assertEqual('123', obj.kwarg2)

  def test_update_callable_invalid_arg(self):
    cfg = config_lib.Config(
        config_test.fn_with_var_kwargs, abc='123', xyz='321'
    )
    with self.assertRaisesRegex(
        TypeError, r"have invalid arguments \['abc', 'xyz'\]"
    ):
      mutate_buildable.update_callable(cfg, config_test.SampleClass)

  def test_update_callable_drop_invalid_arg(self):
    cfg = config_lib.Config(
        config_test.fn_with_var_kwargs, arg1='123', xyz='321'
    )
    mutate_buildable.update_callable(
        cfg, config_test.SampleClass, drop_invalid_args=True
    )
    self.assertEqual(
        cfg, config_lib.Config(config_test.SampleClass, arg1='123')
    )

  def test_update_callable_new_kwargs(self):
    cfg = config_lib.Config(config_test.SampleClass)
    cfg.arg1 = 1
    mutate_buildable.update_callable(cfg, config_test.fn_with_var_kwargs)
    cfg.abc = '123'  # A **kwargs value should now be allowed.
    self.assertEqual(
        {'arg1': 1, 'kwarg1': None, 'kwargs': {'abc': '123'}},
        building.build(cfg),
    )

  def test_get_callable(self):
    cfg = config_lib.Config(config_test.basic_fn)
    self.assertIs(fdl.get_callable(cfg), config_test.basic_fn)

  def test_positional_args(self):
    cfg = config_lib.Config(config_test.fn_with_position_args, 1, 2)
    with self.assertRaisesRegex(
        NotImplementedError, 'positional arguments is not supported'
    ):
      mutate_buildable.update_callable(
          cfg, config_test.fn_with_args_and_kwargs_only
      )


class AssignTest(absltest.TestCase):

  def test_assign(self):
    cfg = config_lib.Config(config_test.fn_with_var_kwargs, 1, 2)
    mutate_buildable.assign(cfg, a='a', b='b')
    self.assertEqual(
        {'arg1': 1, 'kwarg1': 2, 'kwargs': {'a': 'a', 'b': 'b'}},
        building.build(cfg),
    )

  def test_assign_wrong_argument(self):
    cfg = config_lib.Config(config_test.basic_fn)
    with self.assertRaisesRegex(AttributeError, 'not_there'):
      mutate_buildable.assign(cfg, arg1=1, not_there=2)

  @absltest.skip('Enable this after dropping pyhon 3.7 support')
  def test_config_for_fn_with_special_arg_names(self):
    # The reason that these tests pass is that we use positional-only
    # parameters for self, etc. in functions such as Config.__build__.
    # If we used keyword-or-positional parameters instead, then these
    # tests would fail with a "multiple values for argument" TypeError.

    def f(self, fn_or_cls, buildable=0):
      return self + fn_or_cls + buildable

    cfg = config_lib.Config(f)
    cfg.self = 100
    cfg.fn_or_cls = 200
    self.assertEqual(building.build(cfg), 300)

    mutate_buildable.assign(cfg, buildable=10)  # pytype: disable=duplicate-keyword-argument
    self.assertEqual(building.build(cfg), 310)

    cfg2 = config_lib.Config(f, self=5, fn_or_cls=1)  # pytype: disable=duplicate-keyword-argument
    self.assertEqual(building.build(cfg2), 6)


if __name__ == '__main__':
  absltest.main()
