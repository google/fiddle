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

"""Tests for module_reflection."""

import types

from absl.testing import absltest
from fiddle import module_reflection
from fiddle import module_reflection_test_module as test_module
from fiddle.experimental import auto_config


class ModuleReflectionTest(absltest.TestCase):

  def test_find_fiddler_like_things(self):
    ns = types.SimpleNamespace()
    ns.simple = lambda x: None
    ns.not_one = lambda x, y: None
    ns.another = lambda y: None
    ns.zero_arity = lambda: None

    def kwarg_fn(cfg, **kwargs):  # pylint: disable=unused-argument
      pass

    ns.kwarg_fn = kwarg_fn

    def with_defaults(cfg, defaulted_arg=1):  # pylint: disable=unused-argument
      pass

    ns.with_defaults = with_defaults

    expected = sorted(['simple', 'another', 'kwarg_fn', 'with_defaults'])
    self.assertEqual(expected, module_reflection.find_fiddler_like_things(ns))

  def test_find_base_config_like_things(self):
    ns = types.SimpleNamespace()
    ns.base = lambda: None
    ns.another = lambda: None
    ns.fiddler = lambda x: None
    ns.not_me = lambda x, y: None

    expected = sorted(['base', 'another'])
    self.assertEqual(expected,
                     module_reflection.find_base_config_like_things(ns))

  def test_find_base_config_like_things_module(self):
    expected = sorted(['simple_base', 'alternate_base', 'base_with_defaults'])
    self.assertEqual(
        expected, module_reflection.find_base_config_like_things(test_module))

  def test_find_fiddler_like_things_module(self):
    expected = sorted(['fiddler1', 'fiddler2', 'another_fiddler'])
    self.assertEqual(expected,
                     module_reflection.find_fiddler_like_things(test_module))

  def test_auto_config_functions(self):

    def my_fn(x, y=3):
      return (x, y)

    @auto_config.auto_config
    def base_config():
      return my_fn(1, 3)

    ns = types.SimpleNamespace()
    ns.base_config = base_config

    expected = ['base_config']
    self.assertIn('base_config', dir(ns))
    self.assertEqual(expected,
                     module_reflection.find_base_config_like_things(ns))


if __name__ == '__main__':
  absltest.main()
