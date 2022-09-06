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

"""Tests for default_factory."""

import dataclasses
import functools
import inspect
import itertools
from typing import List, Any
from absl.testing import absltest

from fiddle.experimental import default_factory


@dataclasses.dataclass
class Child:
  x: int = 0


@dataclasses.dataclass
class Parent:
  children: List[Child] = dataclasses.field(default_factory=list)

  def add_child(self, child):
    self.children.append(child)

  new_child = functools.partialmethod(
      default_factory.CallableWithDefaultFactory(add_child, child=Child))


def test_fn(arg1, arg2, kwarg1=0, kwarg2=None):
  return (arg1, arg2, kwarg1, kwarg2)


def posonly_fn(a, b, c, /):
  return (a, b, c)


class Counter:

  def __init__(self, name='counter'):
    self.name = name
    self.count = 0

  def __call__(self):
    self.count += 1
    return f'{self.name}_{self.count}'


class DefaultFactoryTest(absltest.TestCase):

  def test_factory_arg(self):
    p = default_factory.CallableWithDefaultFactory(test_fn, arg2=Counter())
    self.assertEqual(p(arg1=5), (5, 'counter_1', 0, None))
    self.assertEqual(p(arg1=5), (5, 'counter_2', 0, None))
    self.assertEqual(p(arg1=2, kwarg2=3), (2, 'counter_3', 0, 3))
    self.assertEqual(p(arg1=1, arg2=2), (1, 2, 0, None))
    self.assertEqual(p(arg1=5), (5, 'counter_4', 0, None))

  def test_factory_kwarg(self):
    p = default_factory.CallableWithDefaultFactory(test_fn, kwarg1=Counter())
    self.assertEqual(p(1, 2), (1, 2, 'counter_1', None))
    self.assertEqual(p(1, 2), (1, 2, 'counter_2', None))
    self.assertEqual(p(1, 2, kwarg2='x'), (1, 2, 'counter_3', 'x'))

  def test_factory_is_not_run_when_overridden(self):
    p = default_factory.CallableWithDefaultFactory(test_fn, arg2=Counter())
    self.assertEqual(p(arg1=5), (5, 'counter_1', 0, None))

    # On the following line, arg2 is overriden, so the factory does not
    # get called:
    self.assertEqual(p(arg1=1, arg2=2), (1, 2, 0, None))
    self.assertEqual(p(arg1=5), (5, 'counter_2', 0, None))

  def test_multiple_factories(self):
    p = default_factory.CallableWithDefaultFactory(
        test_fn,
        arg1=Counter('arg1'),
        arg2=Counter('arg2'),
        kwarg2=Counter('kwarg2'))
    self.assertEqual(p(), ('arg1_1', 'arg2_1', 0, 'kwarg2_1'))
    self.assertEqual(p(arg1=5), (5, 'arg2_2', 0, 'kwarg2_2'))
    self.assertEqual(p(arg1=5, arg2=6), (5, 6, 0, 'kwarg2_3'))
    self.assertEqual(p(arg1=5, arg2=6, kwarg1=7, kwarg2=8), (5, 6, 7, 8))
    self.assertEqual(p(), ('arg1_2', 'arg2_3', 0, 'kwarg2_4'))

  def test_reused_factory(self):
    factory = Counter()
    p = default_factory.CallableWithDefaultFactory(
        test_fn, arg1=factory, arg2=factory)
    self.assertEqual(p(), ('counter_1', 'counter_2', 0, None))
    self.assertEqual(p(), ('counter_3', 'counter_4', 0, None))

  def test_list_factory(self):

    def f(name, children):
      return {'name': name, 'children': children}

    p = default_factory.CallableWithDefaultFactory(f, children=list)
    joe = p('joe')
    sue = p('sue')

    self.assertEqual(joe, dict(name='joe', children=[]))
    self.assertEqual(sue, dict(name='sue', children=[]))
    self.assertIsNot(joe['children'], sue['children'])  # children not shared

  def test_itertools_count_factory(self):
    double = lambda x: x * 2
    p = default_factory.CallableWithDefaultFactory(
        double, x=itertools.count().__next__)
    self.assertEqual(p(), 0)
    self.assertEqual(p(), 2)
    self.assertEqual(p(), 4)

  def test_customized_dataclass_default_factory(self):
    parent_factory = default_factory.CallableWithDefaultFactory(
        Parent, children=lambda: [Child(3)])
    a = parent_factory()
    b = parent_factory()
    self.assertEqual(a, Parent([Child(3)]))
    self.assertEqual(b, Parent([Child(3)]))
    self.assertIsNot(a.children, b.children)  # child list is not shared
    self.assertIsNot(a.children[0], b.children[0])  # child not shared

  def test_no_multiple_value_errors(self):
    # Check that all parameter names are supported, including names such
    # as `self` and `func` that are used in the signature of the constructor
    # or the __call__ method.  The reason that these tests pass is that the
    # parameters in question are declared as positional-only or
    # vararg/varkwarg.
    def f(self, func, args, kwargs):
      return self + func + args + kwargs

    p = default_factory.CallableWithDefaultFactory(
        f, self=lambda: 1, func=lambda: 10)
    self.assertEqual(p(args=100, kwargs=1000), 1111)
    self.assertEqual(p(self=4, func=30, args=200, kwargs=1000), 1234)

  def test_param_named_func(self):
    # Note: this test succeeds because CallableWithDefaultFactory declares
    # `func` as a positional-only parameter.  Otherwise, it would fail with
    # "multiple values for argument 'func'".
    def f(func):
      return func

    p = default_factory.CallableWithDefaultFactory(f, func=lambda: 1)
    self.assertEqual(p(), 1)

  def test_partialmethod(self):
    parent = Parent()
    parent.new_child()
    parent.new_child()
    self.assertEqual(parent, Parent([Child(0), Child(0)]))
    self.assertIsNot(parent.children[0], parent.children[1])

  def test_posonly(self):
    p = default_factory.CallableWithDefaultFactory(posonly_fn, c=list)
    x = p(1, 2)
    y = p(1, 2)
    self.assertEqual(x, (1, 2, []))
    self.assertEqual(x, y)
    self.assertIsNot(x[2], y[2])  # child list not shared

  def test_signature_arg_without_default(self):
    p = default_factory.CallableWithDefaultFactory(test_fn, arg2=Child)
    self.assertEqual(
        str(inspect.signature(p)),
        '(arg1, arg2=<built by: Child()>, kwarg1=0, kwarg2=None)')

  def test_signature_arg_with_default(self):
    p = default_factory.CallableWithDefaultFactory(
        test_fn, arg2=Parent, kwarg2=list)
    self.assertEqual(
        str(inspect.signature(p)), '(arg1, arg2=<built by: Parent()>, '
        'kwarg1=0, kwarg2=<built by: list()>)')

  def test_signature_arg_before_arg_without_default(self):
    # All arguments after arg1 get converted to KEYWORD_ONLY.
    p = default_factory.CallableWithDefaultFactory(test_fn, arg1=Child)
    self.assertEqual(
        str(inspect.signature(p)),
        '(arg1=<built by: Child()>, *, arg2, kwarg1=0, kwarg2=None)')

    p = default_factory.CallableWithDefaultFactory(
        test_fn, arg1=Child, arg2=Child)
    self.assertEqual(
        str(inspect.signature(p)),
        '(arg1=<built by: Child()>, arg2=<built by: Child()>, ' +
        'kwarg1=0, kwarg2=None)')

  def test_signature_arg_before_pos_only_arg_without_default(self):
    with self.assertRaisesRegex(
        ValueError, "non-default argument 'c' follows default argument"):
      default_factory.CallableWithDefaultFactory(posonly_fn, b=list)

    p = default_factory.CallableWithDefaultFactory(posonly_fn, b=list, c=list)
    self.assertEqual(
        str(inspect.signature(p)),
        '(a, b=<built by: list()>, c=<built by: list()>, /)')

  def test_decorator(self):

    @default_factory.for_parameter(seq=list)
    def append(elt, seq: List[Any] = None):
      seq.append(elt)
      return seq

    x = append(3)
    y = append(4)
    z = append(5, x)
    self.assertEqual(x, [3, 5])
    self.assertEqual(y, [4])
    self.assertIs(x, z)
    self.assertIsNot(x, y)

  def test_repr(self):
    p1 = default_factory.CallableWithDefaultFactory(test_fn, arg2=Child)
    self.assertEqual(
        repr(p1), '<CallableWithDefaultFactory test_fn(arg1, '
        'arg2=<built by: Child()>, kwarg1=0, kwarg2=None)>')

  def test_default_for_keyword_only_param(self):

    @default_factory.for_parameter(y=list)
    def f(x, *, y=None):
      return (x, y)

    self.assertEqual(f(1), (1, []))
    self.assertEqual(f(1, y=2), (1, 2))

  def test_default_for_positional_only_param(self):

    @default_factory.for_parameter(y=list)
    def f(x, y=None, /):
      return (x, y)

    self.assertEqual(f(1), (1, []))
    self.assertEqual(f(1, 2), (1, 2))


if __name__ == '__main__':
  absltest.main()
