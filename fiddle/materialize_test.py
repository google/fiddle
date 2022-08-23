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

"""Tests for materialize."""

from absl.testing import absltest
import fiddle as fdl
from fiddle import materialize


class MaterializeTest(absltest.TestCase):

  def test_materialize_defaults(self):

    def test_defaulting_fn(x: int, y=3, z: str = 'abc'):
      del x, y, z  # Unused.

    cfg = fdl.Config(test_defaulting_fn)

    self.assertEqual({}, cfg.__arguments__)
    materialize.materialize_defaults(cfg)
    self.assertEqual({'y': 3, 'z': 'abc'}, cfg.__arguments__)

  def test_materialize_defaults_recursive(self):

    class Inner:

      def __init__(self, a, b: str = 'b'):
        self.a = a
        self.b = b

    def test_defaulting_fn(x, y, z: str = 'abc'):
      del x, y, z  # Unused.

    cfg = fdl.Config(test_defaulting_fn)
    cfg.y = [
        fdl.Config(Inner, 'a'),
        fdl.Config(Inner),
    ]
    self.assertEqual({'a': 'a'}, cfg.y[0].__arguments__)
    self.assertEqual({}, cfg.y[1].__arguments__)

    materialize.materialize_defaults(cfg)
    self.assertEqual({'a': 'a', 'b': 'b'}, cfg.y[0].__arguments__)
    self.assertEqual({'b': 'b'}, cfg.y[1].__arguments__)
    self.assertEqual('abc', cfg.__arguments__['z'])


if __name__ == '__main__':
  absltest.main()
