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

"""Tests for with_tags."""

from absl.testing import absltest
import fiddle as fdl
from fiddle._src import tagging
from fiddle._src.experimental import auto_config
from fiddle._src.experimental import with_tags


class FakeTag1(fdl.Tag):
  """Sample tag for testing."""


class FakeTag2(fdl.Tag):
  """Sample tag for testing."""


class WithTagTest(absltest.TestCase):

  def test_multiple_tags(self):
    def add(x, y, z):
      return x + y + z

    @auto_config.auto_config
    def fn():
      return add(
          x=with_tags.with_tags(100, [FakeTag1, FakeTag2]),
          y=with_tags.with_tags(20, FakeTag1),
          z=3,
      )

    with self.subTest("dunder_call"):
      self.assertEqual(fn(), 123)

    with self.subTest("as_buildable"):
      cfg = fn.as_buildable()
      tagging.set_tagged(root=cfg, tag=FakeTag1, value=0)
      tagging.set_tagged(root=cfg, tag=FakeTag2, value=200)
      self.assertEqual(fdl.build(cfg), 203)

  def test_nested_call(self):
    def add(x, y):
      return x + y

    @auto_config.auto_config
    def outer_fn():
      a = add(x=with_tags.with_tags(1, FakeTag1), y=2)
      b = add(x=with_tags.with_tags(3, FakeTag1), y=4)
      return add(a, b)

    with self.subTest("dunder_call"):
      self.assertEqual(outer_fn(), 10)

    with self.subTest("as_buildable"):
      cfg = outer_fn.as_buildable()
      tagging.set_tagged(root=cfg.x, tag=FakeTag1, value=0)
      tagging.set_tagged(root=cfg.y, tag=FakeTag1, value=100)
      self.assertEqual(fdl.build(cfg), 106)


if __name__ == "__main__":
  absltest.main()
