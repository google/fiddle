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

"""Tests for namespace_config."""

import copy
import types

from absl.testing import absltest
import fiddle as fdl
from fiddle.experimental import namespace_config


class NamespaceConfigTest(absltest.TestCase):

  def test_simple(self):
    cfg = namespace_config.NamespaceConfig()
    cfg.x = 1
    cfg.y = 'abc'
    obj = fdl.build(cfg)
    self.assertIsInstance(obj, types.SimpleNamespace)
    self.assertEqual(1, obj.x)
    self.assertEqual('abc', obj.y)

  def test_initialize_from_dict(self):
    test_dict = {'a': 1, 'b': 'xyz'}
    cfg = namespace_config.NamespaceConfig(**test_dict)
    obj = fdl.build(cfg)
    self.assertIsInstance(obj, types.SimpleNamespace)
    self.assertEqual(1, obj.a)
    self.assertEqual('xyz', obj.b)

  def test_nesting(self):

    def dummy_fn(**kwargs):
      return (dummy_fn, kwargs)

    cfg = namespace_config.NamespaceConfig()
    cfg.nested = fdl.Config(dummy_fn, x='a')
    cfg.nested.y = 123
    cfg.nested.other_dictconfig = namespace_config.NamespaceConfig(
        very_inside='here')
    cfg.other = 321

    obj = fdl.build(cfg)
    self.assertIsInstance(obj, types.SimpleNamespace)
    self.assertIs(dummy_fn, obj.nested[0])
    self.assertEqual(321, obj.other)
    self.assertIsInstance(obj.nested[1]['other_dictconfig'],
                          types.SimpleNamespace)
    self.assertEqual('here', obj.nested[1]['other_dictconfig'].very_inside)

  def test_copy(self):
    cfg1 = namespace_config.NamespaceConfig()
    cfg1.x = 1
    cfg1.y = 'abc'
    cfg2 = copy.copy(cfg1)
    self.assertEqual(cfg1, cfg2)


if __name__ == '__main__':
  absltest.main()
