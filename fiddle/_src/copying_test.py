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

"""Tests for the `fiddle._src.copy` module."""

from absl.testing import absltest
from fiddle._src import building
from fiddle._src import config as config_lib
from fiddle._src import config_test
from fiddle._src import copying
from fiddle._src import tagging


class CopyTest(absltest.TestCase):

  def test_copy_with(self):
    cfg1 = config_lib.Config(config_test.fn_with_var_kwargs, 1, 2, c=[])
    tagging.add_tag(cfg1, 'arg1', config_test.Tag1)

    expected_cfg1 = dict(arg1=1, kwarg1=2, kwargs=dict(c=[]))
    expected_cfg2 = dict(arg1=5, kwarg1=2, kwargs=dict(a='a', b='b', c=[]))

    with self.subTest('cfg1_value'):
      self.assertEqual(expected_cfg1, building.build(cfg1))

    with self.subTest('shallow_copy'):
      cfg2 = copying.copy_with(cfg1, arg1=5, a='a', b='b')
      self.assertIsNot(cfg1, cfg2)
      self.assertIsNot(cfg1.__arguments__, cfg2.__arguments__)
      self.assertIsNot(cfg1.__argument_tags__, cfg2.__argument_tags__)
      self.assertEqual(cfg1.__argument_tags__, cfg2.__argument_tags__)
      self.assertIs(cfg1.c, cfg2.c)  # Shallow copy.
      self.assertEqual(expected_cfg2, building.build(cfg2))

    with self.subTest('deep_copy'):
      cfg2 = copying.deepcopy_with(cfg1, arg1=5, a='a', b='b')
      self.assertIsNot(cfg1, cfg2)
      self.assertIsNot(cfg1.__arguments__, cfg2.__arguments__)
      self.assertIsNot(cfg1.__argument_tags__, cfg2.__argument_tags__)
      self.assertEqual(cfg1.__argument_tags__, cfg2.__argument_tags__)
      self.assertIsNot(cfg1.c, cfg2.c)  # Deep copy.
      self.assertEqual(expected_cfg2, building.build(cfg2))


if __name__ == '__main__':
  absltest.main()
