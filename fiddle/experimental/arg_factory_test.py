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

"""Tests for partial_with_arg_factory."""

import dataclasses
import functools
import inspect
import itertools
from typing import List
from absl.testing import absltest
from fiddle.experimental import arg_factory


@dataclasses.dataclass
class Child:
  x: int = 0


@dataclasses.dataclass
class Parent:
  children: List[Child] = dataclasses.field(default_factory=list)

  def add_child(self, child):
    self.children.append(child)

  new_child = arg_factory.partialmethod(add_child,
                                        arg_factory.ArgFactory(Child))


def test_fn(arg1, arg2, kwarg1=0, kwarg2=None):
  return (arg1, arg2, kwarg1, kwarg2)


class Counter:

  def __init__(self, name='counter'):
    self.name = name
    self.count = 0

  def __call__(self):
    self.count += 1
    return f'{self.name}_{self.count}'


class ArgFactoryTest(absltest.TestCase):

  def test_factory_arg(self):
    p = arg_factory.partial(test_fn, arg2=arg_factory.ArgFactory(Counter()))
    self.assertEqual(p(arg1=5), (5, 'counter_1', 0, None))
    self.assertEqual(p(arg1=5), (5, 'counter_2', 0, None))
    self.assertEqual(p(arg1=2, kwarg2=3), (2, 'counter_3', 0, 3))
    self.assertEqual(p(arg1=1, arg2=2), (1, 2, 0, None))
    self.assertEqual(p(arg1=5), (5, 'counter_4', 0, None))

  def test_factory_kwarg(self):
    p = arg_factory.partial(test_fn, kwarg1=arg_factory.ArgFactory(Counter()))
    self.assertEqual(p(1, 2), (1, 2, 'counter_1', None))
    self.assertEqual(p(1, 2), (1, 2, 'counter_2', None))
    self.assertEqual(p(1, 2, kwarg2='x'), (1, 2, 'counter_3', 'x'))

  def test_factory_is_not_run_when_overridden(self):
    p = arg_factory.partial(test_fn, arg2=arg_factory.ArgFactory(Counter()))
    self.assertEqual(p(arg1=5), (5, 'counter_1', 0, None))

    # On the following line, arg2 is overriden, so the factory does not
    # get called:
    self.assertEqual(p(arg1=1, arg2=2), (1, 2, 0, None))
    self.assertEqual(p(arg1=5), (5, 'counter_2', 0, None))

  def test_multiple_factories(self):
    p = arg_factory.partial(
        test_fn,
        arg1=arg_factory.ArgFactory(Counter('arg1')),
        arg2=arg_factory.ArgFactory(Counter('arg2')),
        kwarg2=arg_factory.ArgFactory(Counter('kwarg2')))
    self.assertEqual(p(), ('arg1_1', 'arg2_1', 0, 'kwarg2_1'))
    self.assertEqual(p(arg1=5), (5, 'arg2_2', 0, 'kwarg2_2'))
    self.assertEqual(p(arg1=5, arg2=6), (5, 6, 0, 'kwarg2_3'))
    self.assertEqual(p(arg1=5, arg2=6, kwarg1=7, kwarg2=8), (5, 6, 7, 8))
    self.assertEqual(p(), ('arg1_2', 'arg2_3', 0, 'kwarg2_4'))

  def test_reused_factory(self):
    factory = arg_factory.ArgFactory(Counter())
    p = arg_factory.partial(test_fn, arg1=factory, arg2=factory)
    self.assertEqual(p(), ('counter_1', 'counter_2', 0, None))
    self.assertEqual(p(), ('counter_3', 'counter_4', 0, None))

  def test_list_factory(self):

    def f(name, children):
      return {'name': name, 'children': children}

    p = arg_factory.partial(f, children=arg_factory.ArgFactory(list))
    joe = p('joe')
    sue = p('sue')

    self.assertEqual(joe, dict(name='joe', children=[]))
    self.assertEqual(sue, dict(name='sue', children=[]))
    self.assertIsNot(joe['children'], sue['children'])  # children not shared

  def test_itertools_count_factory(self):
    double = lambda x: x * 2
    p = arg_factory.partial(double,
                            arg_factory.ArgFactory(itertools.count().__next__))
    self.assertEqual(p(), 0)
    self.assertEqual(p(), 2)
    self.assertEqual(p(), 4)

  def test_customized_dataclass_default_factory(self):
    parent_factory = arg_factory.partial(
        Parent, arg_factory.ArgFactory(lambda: [Child(3)]))
    a = parent_factory()
    b = parent_factory()
    self.assertEqual(a, Parent([Child(3)]))
    self.assertEqual(b, Parent([Child(3)]))
    self.assertIsNot(a.children, b.children)  # child list is not shared
    self.assertIsNot(a.children[0], b.children[0])  # child not shared

  def test_param_named_self(self):
    # Note: this test succeeds because _InvokeArgFactoryWrapper.__call__
    # declares `self` as a positional-only parameter.  Otherwise, __call__ would
    # fail with "multiple values for argument 'self'".
    def f(self, other):
      return self + other

    p = arg_factory.partial(f, other=10)
    self.assertEqual(p(self=5), 15)

  def test_param_named_func(self):
    # Note: this test succeeds because arg_factory.partial declares `func` as a
    # positional-only parameter.  Otherwise, it would fail with "multiple values
    # for argument 'func'".
    def f(func, args, kwargs):
      return (func, args, kwargs)

    p = arg_factory.partial(f, func=1, args=2, kwargs=3)
    self.assertEqual(p(), (1, 2, 3))

  def test_partialmethod(self):
    parent = Parent()
    parent.new_child()
    parent.new_child()
    self.assertEqual(parent, Parent([Child(0), Child(0)]))
    self.assertIsNot(parent.children[0], parent.children[1])

  def test_is_a_factory_partial(self):
    # True:
    self.assertTrue(
        arg_factory.is_arg_factory_partial(arg_factory.partial(Child, x=0)))
    self.assertTrue(
        arg_factory.is_arg_factory_partial(
            arg_factory.partial(lambda x: x + 1, x=0)))
    self.assertTrue(
        arg_factory.is_arg_factory_partial(
            arg_factory.partialmethod(Child(0).__eq__)))

    # False:
    self.assertFalse(
        arg_factory.is_arg_factory_partial(functools.partial(Child, x=0)))
    self.assertFalse(
        arg_factory.is_arg_factory_partial(
            functools.partial(lambda x: x + 1, x=0)))
    self.assertFalse(
        arg_factory.is_arg_factory_partial(
            functools.partialmethod(Child(0).__eq__)))
    self.assertFalse(arg_factory.is_arg_factory_partial(None))
    self.assertFalse(arg_factory.is_arg_factory_partial('xyz'))

  def test_partial_func_signature(self):
    p1 = arg_factory.partial(test_fn, arg1=5)
    p2 = functools.partial(test_fn, arg1=5)
    self.assertEqual(inspect.signature(p1), inspect.signature(p2))


if __name__ == '__main__':
  absltest.main()
