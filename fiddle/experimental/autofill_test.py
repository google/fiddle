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

"""Tests for autofill."""

import dataclasses

from absl.testing import absltest
from fiddle import building
from fiddle import config
from fiddle.experimental import autofill


class Child:

  def __init__(self, x: str):
    self.x = x

  @staticmethod
  def __fiddle_init__(cfg):
    cfg.x = 42


@dataclasses.dataclass
class ChildDC:
  a: int
  b: str


@dataclasses.dataclass
class ParentDC:
  child1: Child
  child2: ChildDC


class AutoFillTest(absltest.TestCase):

  def test_nothing_to_fill(self):
    cfg = config.Config(Child)
    autofill.autofill(cfg)  # No changes expected.

    self.assertEqual(cfg, config.Config(Child))

  def test_simple_autofill(self):
    cfg = config.Config(ParentDC)

    self.assertFalse(hasattr(cfg, 'child1'))
    self.assertFalse(hasattr(cfg, 'child2'))

    autofill.autofill(cfg)

    self.assertIsInstance(cfg.child1, config.Config)
    self.assertEqual(cfg.child1.__fn_or_cls__, Child)
    self.assertIsInstance(cfg.child2, config.Config)
    self.assertEqual(cfg.child2.__fn_or_cls__, ChildDC)

  def test_recursive_autofill(self):

    class Grandparent:

      def __init__(self, parent: ParentDC):
        self.parent = parent

    cfg = config.Config(Grandparent)
    self.assertFalse(hasattr(cfg, 'parent'))

    autofill.autofill(cfg)

    self.assertIsInstance(cfg.parent, config.Config)
    self.assertEqual(cfg.parent.__fn_or_cls__, ParentDC)
    self.assertIsInstance(cfg.parent.child1, config.Config)
    self.assertEqual(cfg.parent.child1.__fn_or_cls__, Child)
    self.assertIsInstance(cfg.parent.child2, config.Config)
    self.assertEqual(cfg.parent.child2.__fn_or_cls__, ChildDC)

  def test_skip_if_no_annotation(self):

    class Parent:

      def __init__(self, child: ChildDC, other_thing):
        self.child = child
        self.other_thing = other_thing

    cfg = config.Config(Parent)

    self.assertFalse(hasattr(cfg, 'child'))
    self.assertFalse(hasattr(cfg, 'other_thing'))

    autofill.autofill(cfg)

    self.assertIsInstance(cfg.child, config.Config)
    self.assertFalse(hasattr(cfg, 'other_thing'))

  def test_skip_if_set(self):
    cfg = config.Config(ParentDC)
    cfg.child1 = config.Config(ChildDC)  # Switch type.
    autofill.autofill(cfg)

    self.assertIsInstance(cfg.child1, config.Config)
    self.assertEqual(cfg.child1.__fn_or_cls__, ChildDC)
    self.assertIsInstance(cfg.child2, config.Config)
    self.assertEqual(cfg.child2.__fn_or_cls__, ChildDC)

  def test_skip_if_default(self):

    @dataclasses.dataclass
    class Defaulted:
      child1: Child
      child2: ChildDC = ChildDC(a=-1, b='static')

    cfg = config.Config(Defaulted)
    autofill.autofill(cfg)

    self.assertIsInstance(cfg.child1, config.Config)
    self.assertIsInstance(cfg.child2, ChildDC)  # Not a config instance!

    cfg.child1.x = 42
    obj = building.build(cfg)

    self.assertEqual(-1, obj.child2.a)
    self.assertEqual('static', obj.child2.b)

  def test_handle_args_and_kwargs(self):

    def test_fn(a: int, b: Child, c: ParentDC, *args: ChildDC, **kwargs: Child):  # pylint: disable=unused-argument
      return locals()

    cfg = config.Config(test_fn)

    self.assertFalse(hasattr(cfg, 'a'))
    self.assertFalse(hasattr(cfg, 'b'))
    self.assertFalse(hasattr(cfg, 'c'))
    self.assertFalse(hasattr(cfg, 'args'))
    self.assertFalse(hasattr(cfg, 'kwargs'))

    autofill.autofill(cfg)

    self.assertFalse(hasattr(cfg, 'a'))
    self.assertTrue(hasattr(cfg, 'b'))
    self.assertTrue(hasattr(cfg, 'c'))
    self.assertFalse(hasattr(cfg, 'args'))
    self.assertFalse(hasattr(cfg, 'kwargs'))

    cfg.a = 42
    cfg.c.child2.a = 'a'
    cfg.c.child2.b = 'b'
    obj = building.build(cfg)

    self.assertEqual(42, obj['a'])
    self.assertIsInstance(obj['c'], ParentDC)


if __name__ == '__main__':
  absltest.main()
