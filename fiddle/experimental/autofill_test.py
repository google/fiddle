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
import fiddle as fdl
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


@dataclasses.dataclass
class StringTypeAnnotations:
  """Type annotations can be strings instead of the types themselves.

  See also `from future import annotations`.
  """
  x: 'Child'
  y: 'ChildDC'


class AutoFillTest(absltest.TestCase):

  def test_nothing_to_fill(self):
    cfg = fdl.Config(Child)
    autofill.autofill(cfg)  # No changes expected.

    self.assertEqual(cfg, fdl.Config(Child))

  def test_simple_autofill(self):
    cfg = fdl.Config(ParentDC)

    self.assertFalse(hasattr(cfg, 'child1'))
    self.assertFalse(hasattr(cfg, 'child2'))

    autofill.autofill(cfg)

    self.assertIsInstance(cfg.child1, fdl.Config)
    self.assertEqual(cfg.child1.__fn_or_cls__, Child)
    self.assertIsInstance(cfg.child2, fdl.Config)
    self.assertEqual(cfg.child2.__fn_or_cls__, ChildDC)

  def test_autofill_with_string_annotations(self):
    cfg = fdl.Config(StringTypeAnnotations)
    self.assertFalse(hasattr(cfg, 'x'))
    self.assertFalse(hasattr(cfg, 'y'))

    autofill.autofill(cfg)

    self.assertIsInstance(cfg.x, fdl.Config)
    self.assertEqual(cfg.x.__fn_or_cls__, Child)
    self.assertIsInstance(cfg.y, fdl.Config)
    self.assertEqual(cfg.y.__fn_or_cls__, ChildDC)

  def test_recursive_autofill(self):

    class Grandparent:

      def __init__(self, parent: ParentDC):
        self.parent = parent

    cfg = fdl.Config(Grandparent)
    self.assertFalse(hasattr(cfg, 'parent'))

    autofill.autofill(cfg)

    self.assertIsInstance(cfg.parent, fdl.Config)
    self.assertEqual(cfg.parent.__fn_or_cls__, ParentDC)
    self.assertIsInstance(cfg.parent.child1, fdl.Config)
    self.assertEqual(cfg.parent.child1.__fn_or_cls__, Child)
    self.assertIsInstance(cfg.parent.child2, fdl.Config)
    self.assertEqual(cfg.parent.child2.__fn_or_cls__, ChildDC)

  def test_skip_if_no_annotation(self):

    class Parent:

      def __init__(self, child: ChildDC, other_thing):
        self.child = child
        self.other_thing = other_thing

    cfg = fdl.Config(Parent)

    self.assertFalse(hasattr(cfg, 'child'))
    self.assertFalse(hasattr(cfg, 'other_thing'))

    autofill.autofill(cfg)

    self.assertIsInstance(cfg.child, fdl.Config)
    self.assertFalse(hasattr(cfg, 'other_thing'))

  def test_skip_if_set(self):
    cfg = fdl.Config(ParentDC)
    cfg.child1 = fdl.Config(ChildDC)  # Switch type.
    autofill.autofill(cfg)

    self.assertIsInstance(cfg.child1, fdl.Config)
    self.assertEqual(cfg.child1.__fn_or_cls__, ChildDC)
    self.assertIsInstance(cfg.child2, fdl.Config)
    self.assertEqual(cfg.child2.__fn_or_cls__, ChildDC)

  def test_skip_if_default(self):

    @dataclasses.dataclass
    class Defaulted:
      child1: Child
      child2: ChildDC = ChildDC(a=-1, b='static')

    cfg = fdl.Config(Defaulted)
    autofill.autofill(cfg)

    self.assertIsInstance(cfg.child1, fdl.Config)
    self.assertIsInstance(cfg.child2, ChildDC)  # Not a config instance!

    cfg.child1.x = 42
    obj = fdl.build(cfg)

    self.assertEqual(-1, obj.child2.a)
    self.assertEqual('static', obj.child2.b)

  def test_handle_args_and_kwargs(self):

    def test_fn(a: int, b: Child, c: ParentDC, *args: ChildDC, **kwargs: Child):  # pylint: disable=unused-argument
      return locals()

    cfg = fdl.Config(test_fn)

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
    obj = fdl.build(cfg)

    self.assertEqual(42, obj['a'])
    self.assertIsInstance(obj['c'], ParentDC)


if __name__ == '__main__':
  absltest.main()
