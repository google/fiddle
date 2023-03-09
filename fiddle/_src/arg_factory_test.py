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

import copy
import dataclasses
import functools
import inspect
import itertools
import random
from typing import List, Any
from absl.testing import absltest
from absl.testing import parameterized
from fiddle import arg_factory


@dataclasses.dataclass
class Child:
  x: int = 0


@dataclasses.dataclass
class Parent:
  children: List[Child] = dataclasses.field(default_factory=list)

  def add_child(self, child):
    self.children.append(child)

  new_child = arg_factory.partialmethod(add_child, Child)

  @arg_factory.supply_defaults
  def new_child_2(self, child=arg_factory.default_factory(Child)):
    self.add_child(child)


def sample_fn(arg1, arg2, kwarg1=0, kwarg2=None):
  return (arg1, arg2, kwarg1, kwarg2)


def posonly_fn(a, b, c, /):
  return (a, b, c)


def varargs_fn(*args, **kwargs):
  return (args, kwargs)


class Counter:

  def __init__(self, name='counter'):
    self.name = name
    self.count = 0

  def __call__(self):
    self.count += 1
    return f'{self.name}_{self.count}'


class ArgFactoryPartialTest(parameterized.TestCase):

  def test_factory_positional_arg(self):
    p = arg_factory.partial(sample_fn, Counter())
    self.assertEqual(p(5), ('counter_1', 5, 0, None))
    self.assertEqual(p(5), ('counter_2', 5, 0, None))
    self.assertEqual(p(2, kwarg2=3), ('counter_3', 2, 0, 3))
    self.assertEqual(p(arg2=5), ('counter_4', 5, 0, None))

  def test_factory_keyword_arg(self):
    p = arg_factory.partial(sample_fn, kwarg1=Counter())
    self.assertEqual(p(1, 2), (1, 2, 'counter_1', None))
    self.assertEqual(p(1, 2), (1, 2, 'counter_2', None))
    self.assertEqual(p(1, 2, kwarg2='x'), (1, 2, 'counter_3', 'x'))

  def test_factory_is_not_run_when_overridden(self):
    p = arg_factory.partial(sample_fn, arg2=Counter())
    self.assertEqual(p(arg1=5), (5, 'counter_1', 0, None))

    # On the following line, arg2 is overriden, so the factory does not
    # get called:
    self.assertEqual(p(arg1=1, arg2=2), (1, 2, 0, None))
    self.assertEqual(p(arg1=5), (5, 'counter_2', 0, None))

  @absltest.skip('Enable this after dropping pyhon 3.7 support')
  def test_positional_only_args(self):

    def f(x, **kwargs):
      # Use (x, /, **kwargs) when test is enabled.
      return x, kwargs

    # Note: `x` is passed in kwargs, and doesn't go to the positional-only
    # argument named `x`.
    p = arg_factory.partial(f, lambda: 'positional_x', x=lambda: 'keyword_x')
    self.assertEqual(p(), ('positional_x', {'x': 'keyword_x'}))

  def test_multiple_factories(self):
    p = arg_factory.partial(
        sample_fn,
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
    p = arg_factory.partial(sample_fn, arg1=factory, arg2=factory)
    self.assertEqual(p(), ('counter_1', 'counter_2', 0, None))
    self.assertEqual(p(), ('counter_3', 'counter_4', 0, None))

  def test_list_factory(self):

    def f(name, children):
      return {'name': name, 'children': children}

    p = arg_factory.partial(f, children=list)
    joe = p('joe')
    sue = p('sue')

    self.assertEqual(joe, dict(name='joe', children=[]))
    self.assertEqual(sue, dict(name='sue', children=[]))
    self.assertIsNot(joe['children'], sue['children'])  # children not shared

  def test_itertools_count_factory(self):
    double = lambda x: x * 2
    p = arg_factory.partial(double, itertools.count().__next__)
    self.assertEqual(p(), 0)
    self.assertEqual(p(), 2)
    self.assertEqual(p(), 4)

  def test_customized_dataclass_default_factory(self):
    parent_factory = arg_factory.partial(Parent, lambda: [Child(3)])
    a = parent_factory()
    b = parent_factory()
    self.assertEqual(a, Parent([Child(3)]))
    self.assertEqual(b, Parent([Child(3)]))
    self.assertIsNot(a.children, b.children)  # child list is not shared
    self.assertIsNot(a.children[0], b.children[0])  # child not shared

  @absltest.skip('Enable this after dropping pyhon 3.7 support')
  def test_param_named_self(self):
    # Note: this test succeeds because _InvokeArgFactoryWrapper.__call__
    # declares `self` as a positional-only parameter.  Otherwise, __call__
    # would fail with "multiple values for argument 'self'".
    def f(self, other):
      return self + other

    p = arg_factory.partial(f, other=lambda: 10)
    self.assertEqual(p(self=5), 15)

  def test_param_named_func(self):
    # Note: this test succeeds because arg_factory.partial declares `func` as
    # a positional-only parameter.  Otherwise, it would fail with "multiple
    # values for argument 'func'".
    def f(func, args, kwargs):
      return (func, args, kwargs)

    p = arg_factory.partial(f, func=list, args=list, kwargs=list)
    self.assertEqual(p(), ([], [], []))

  def test_partialmethod(self):
    parent = Parent()
    parent.new_child()
    parent.new_child()
    self.assertEqual(parent, Parent([Child(0), Child(0)]))
    self.assertIsNot(parent.children[0], parent.children[1])

  def test_is_a_factory_partial(self):
    # True:
    self.assertTrue(
        arg_factory.is_arg_factory_partial(arg_factory.partial(Child, x=list)))
    self.assertTrue(
        arg_factory.is_arg_factory_partial(
            arg_factory.partial(lambda x: x + 1, x=list)))
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

  @parameterized.parameters([
      (arg_factory.partial(sample_fn, arg1=list),
       '(*, arg1=<built by: list()>, arg2, kwarg1=0, kwarg2=None)'),
      (arg_factory.partial(sample_fn, kwarg1=list),
       '(arg1, arg2, *, kwarg1=<built by: list()>, kwarg2=None)'),
  ])
  def test_partial_func_signature(self, p, signature):
    self.assertEqual(str(inspect.signature(p)), signature)

  def test_arg_factory_partial_wrap_functools_partial(self):
    p1 = functools.partial(sample_fn, arg1=5)
    p2 = arg_factory.partial(p1, arg2=list)
    self.assertEqual(p2(kwarg1=10), (5, [], 10, None))

  def test_functools_partial_wrap_arg_factory_partial(self):
    p1 = arg_factory.partial(sample_fn, arg2=list)
    p2 = functools.partial(p1, arg1=5)
    self.assertEqual(p2(kwarg1=10), (5, [], 10, None))

  def test_arg_factories_must_be_callable_error(self):
    with self.assertRaisesRegex(TypeError, 'Expected .* to be callable'):
      arg_factory.partial(sample_fn, 5)
    with self.assertRaisesRegex(TypeError, 'Expected .* to be callable'):
      arg_factory.partial(sample_fn, kwarg1=None)

  def test_partial_cant_bind_positional_only_params_as_keywords(self):
    p = arg_factory.partial(posonly_fn, c=list)
    with self.assertRaisesRegex(
        TypeError,
        'got some positional-only arguments ' + 'passed as keyword arguments'):
      p(1, 2)


class DefaultFactoryTest(parameterized.TestCase):

  def test_factory_arg(self):

    @arg_factory.supply_defaults
    def p(
        arg1, arg2=arg_factory.default_factory(Counter()), kwarg1=0, kwarg2=None
    ):
      return (arg1, arg2, kwarg1, kwarg2)

    self.assertEqual(p(arg1=5), (5, 'counter_1', 0, None))
    self.assertEqual(p(arg1=5), (5, 'counter_2', 0, None))
    self.assertEqual(p(arg1=2, kwarg2=3), (2, 'counter_3', 0, 3))
    self.assertEqual(p(arg1=1, arg2=2), (1, 2, 0, None))
    self.assertEqual(p(arg1=5), (5, 'counter_4', 0, None))

  def test_factory_kwarg(self):

    @arg_factory.supply_defaults
    def p(
        arg1, arg2, kwarg1=arg_factory.default_factory(Counter()), kwarg2=None
    ):
      return (arg1, arg2, kwarg1, kwarg2)

    self.assertEqual(p(1, 2), (1, 2, 'counter_1', None))
    self.assertEqual(p(1, 2), (1, 2, 'counter_2', None))
    self.assertEqual(p(1, 2, kwarg2='x'), (1, 2, 'counter_3', 'x'))

  def test_factory_is_not_run_when_overridden(self):

    @arg_factory.supply_defaults
    def p(
        arg1, arg2=arg_factory.default_factory(Counter()), kwarg1=0, kwarg2=None
    ):
      return (arg1, arg2, kwarg1, kwarg2)

    self.assertEqual(p(arg1=5), (5, 'counter_1', 0, None))

    # On the following line, arg2 is overriden, so the factory does not
    # get called:
    self.assertEqual(p(arg1=1, arg2=2), (1, 2, 0, None))
    self.assertEqual(p(arg1=5), (5, 'counter_2', 0, None))

  def test_multiple_factories(self):

    @arg_factory.supply_defaults
    def p(
        arg1=arg_factory.default_factory(Counter('arg1')),
        arg2=arg_factory.default_factory(Counter('arg2')),
        kwarg1=0,
        kwarg2=arg_factory.default_factory(Counter('kwarg2')),
    ):
      return (arg1, arg2, kwarg1, kwarg2)

    self.assertEqual(p(), ('arg1_1', 'arg2_1', 0, 'kwarg2_1'))
    self.assertEqual(p(arg1=5), (5, 'arg2_2', 0, 'kwarg2_2'))
    self.assertEqual(p(arg1=5, arg2=6), (5, 6, 0, 'kwarg2_3'))
    self.assertEqual(p(arg1=5, arg2=6, kwarg1=7, kwarg2=8), (5, 6, 7, 8))
    self.assertEqual(p(), ('arg1_2', 'arg2_3', 0, 'kwarg2_4'))

  def test_reused_factory(self):
    factory = Counter()

    @arg_factory.supply_defaults
    def p(
        arg1=arg_factory.default_factory(factory),
        arg2=arg_factory.default_factory(factory),
        kwarg1=0,
        kwarg2=None,
    ):
      return (arg1, arg2, kwarg1, kwarg2)

    self.assertEqual(p(), ('counter_1', 'counter_2', 0, None))
    self.assertEqual(p(), ('counter_3', 'counter_4', 0, None))

  def test_list_factory(self):

    @arg_factory.supply_defaults
    def f(name, children=arg_factory.default_factory(list)):
      return {'name': name, 'children': children}

    joe = f('joe')
    sue = f('sue')

    self.assertEqual(joe, dict(name='joe', children=[]))
    self.assertEqual(sue, dict(name='sue', children=[]))
    self.assertIsNot(joe['children'], sue['children'])  # children not shared

  def test_itertools_count_factory(self):

    @arg_factory.supply_defaults
    def p(x=arg_factory.default_factory(itertools.count().__next__)):
      return x * 2

    self.assertEqual(p(), 0)
    self.assertEqual(p(), 2)
    self.assertEqual(p(), 4)

  def test_customized_dataclass_default_factory(self):

    @arg_factory.supply_defaults
    def parent_factory(
        children=arg_factory.default_factory(lambda: [Child(3)]),
    ):
      return Parent(children)

    a = parent_factory()
    b = parent_factory()
    self.assertEqual(a, Parent([Child(3)]))
    self.assertEqual(b, Parent([Child(3)]))
    self.assertIsNot(a.children, b.children)  # child list is not shared
    self.assertIsNot(a.children[0], b.children[0])  # child not shared

  def test_partialmethod(self):
    parent = Parent()
    parent.new_child_2()
    parent.new_child_2()
    self.assertEqual(parent, Parent([Child(0), Child(0)]))
    self.assertIsNot(parent.children[0], parent.children[1])

  def test_posonly(self):

    @arg_factory.supply_defaults
    def p(a, b, c=arg_factory.default_factory(list), /):
      return (a, b, c)

    x = p(1, 2)
    y = p(1, 2)
    self.assertEqual(x, (1, 2, []))
    self.assertEqual(x, y)
    self.assertIsNot(x[2], y[2])  # child list not shared

  def test_supply_defaults(self):
    @arg_factory.supply_defaults
    def append(elt, seq: List[Any] = arg_factory.default_factory(list)):
      """Docstring."""
      seq.append(elt)
      return seq

    x = append(3)  # pylint: disable=no-value-for-parameter
    y = append(4)  # pylint: disable=no-value-for-parameter
    z = append(5, x)
    self.assertEqual(x, [3, 5])
    self.assertEqual(y, [4])
    self.assertIs(x, z)
    self.assertIsNot(x, y)

    self.assertEqual(append.__doc__, 'Docstring.')
    self.assertEqual(append.__name__, 'append')
    self.assertTrue(hasattr(append, '__wrapped__'))
    self.assertEqual(append.__qualname__, append.__wrapped__.__qualname__)  # pytype: disable=attribute-error
    self.assertEqual(append.__module__, append.__wrapped__.__module__)  # pytype: disable=attribute-error

  def test_default_for_keyword_only_param(self):

    @arg_factory.supply_defaults
    def f(x, *, y=arg_factory.default_factory(list)):
      return (x, y)

    self.assertEqual(f(1), (1, []))
    self.assertEqual(f(1, y=2), (1, 2))

  def test_default_for_positional_only_param(self):

    @arg_factory.supply_defaults
    def f(x, y=arg_factory.default_factory(list), /):
      return (x, y)

    self.assertEqual(f(1), (1, []))
    self.assertEqual(f(1, 2), (1, 2))

  def test_supply_defaults_with_no_default_factory(self):
    with self.assertRaisesRegex(
        ValueError, 'expected at least one argument to have'
    ):
      arg_factory.supply_defaults(lambda x, y: (x, y))

  def test_missing_decorator_error(self):
    expected_error = (
        r'arg_factory\.default_factory\(.*\) does not support.*\n.*\nDid '
        r'you forget to apply the `@arg_factory\.supply_defaults` decorator\?'
    )

    def f(x, lst=arg_factory.default_factory(list)):
      lst.append(x)
      return lst

    def g(x, noise=arg_factory.default_factory(random.random)):
      return x * x + noise

    with self.assertRaisesRegex(AttributeError, expected_error):
      f(5)
    with self.assertRaisesRegex(ValueError, expected_error):
      g(5)

  def test_default_factory_type_override(self):
    # Make sure we don't get a pytype annotation-type-mismatch error.
    @arg_factory.supply_defaults
    def f(x: List[Any] = arg_factory.default_factory(list)):
      return x

    self.assertEqual(f(), [])

  def test_arg_factory_type_override(self):
    # Make sure we don't get a pytype annotation-type-mismatch error.
    @arg_factory.supply_defaults
    def f(x: List[Any] = arg_factory.ArgFactory(list)):
      return x

    self.assertEqual(f(), [])

  def test_arg_factory_equal(self):
    list_factory = arg_factory.ArgFactory(list)
    dict_factory = arg_factory.ArgFactory(dict)
    self.assertEqual(list_factory, arg_factory.ArgFactory(list))
    self.assertNotEqual(list_factory, dict_factory)

  def test_arg_factory_copy_and_deepcopy(self):
    factory = arg_factory.ArgFactory(list)
    self.assertEqual(factory, copy.copy(factory))
    self.assertEqual(factory, copy.deepcopy(factory))


if __name__ == '__main__':
  absltest.main()
