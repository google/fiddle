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

"""Tests for dict_config."""

import copy

from absl.testing import absltest
import fiddle as fdl
from fiddle.experimental import dict_config


class DictConfigTest(absltest.TestCase):

  def test_simple(self):
    cfg = dict_config.DictConfig()
    cfg.x = 1
    cfg.y = 'abc'
    obj = fdl.build(cfg)
    self.assertIsInstance(obj, dict)
    self.assertEqual(1, obj['x'])
    self.assertEqual('abc', obj['y'])

  def test_initialize_from_dict(self):
    test_dict = {'a': 1, 'b': 'xyz'}
    cfg = dict_config.DictConfig(**test_dict)
    obj = fdl.build(cfg)
    self.assertIsInstance(obj, dict)
    self.assertEqual(1, obj['a'])
    self.assertEqual('xyz', obj['b'])

  def test_nesting(self):

    def dummy_fn(**kwargs):
      return (dummy_fn, kwargs)

    cfg = dict_config.DictConfig()
    cfg.nested = fdl.Config(dummy_fn, x='a')
    cfg.nested.y = 123
    cfg.nested.other_dictconfig = dict_config.DictConfig(very_inside='here')
    cfg.other = 321

    obj = fdl.build(cfg)
    self.assertIsInstance(obj, dict)
    self.assertIs(dummy_fn, obj['nested'][0])
    expected_inner = {
        'x': 'a',
        'y': 123,
        'other_dictconfig': {
            'very_inside': 'here'
        }
    }
    self.assertEqual(expected_inner, obj['nested'][1])
    self.assertEqual(321, obj['other'])
    self.assertIsInstance(obj['nested'][1]['other_dictconfig'], dict)
    self.assertEqual('here',
                     obj['nested'][1]['other_dictconfig']['very_inside'])

  def test_copy(self):
    cfg1 = dict_config.DictConfig()
    cfg1.x = 1
    cfg1.y = 'abc'
    cfg2 = copy.copy(cfg1)
    self.assertEqual(cfg1, cfg2)


if __name__ == '__main__':
  absltest.main()
