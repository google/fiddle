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

"""Tests for transform."""

import dataclasses
from typing import Callable, Optional, Tuple

from absl.testing import absltest
import fiddle as fdl
from fiddle.experimental import transform


@dataclasses.dataclass
class Foo:
  a_tuple: Tuple[int]
  another_tuple: Tuple[int]
  a_callable: Optional[Callable[[], None]] = None
  a_field_with_default: Optional[int] = 123


class TransformTest(absltest.TestCase):

  def test_python_interns_tuples(self):
    """Tests whether Python interns tuples.

    This test is unique in that it does not test our code's functionality but
    rather our assumptions about how Python works. Currently, we assume (and
    tested) that Python interns (i.e. share the same instance) tuples that have
    the same value but are declared in different locations in the code. If this
    test ever fails, then it likely means Python is no longer interning tuples,
    so there's no need for unintern_tuples_of_literals as each tuple will have a
    different ID.
    """
    foo_cfg = fdl.Config(Foo, a_tuple=(1, 2, 3), another_tuple=(1, 2, 3))
    self.assertIs(
        foo_cfg.a_tuple, foo_cfg.another_tuple,
        "This likely means Python is no longer interning tuples. We can remove "
        "unintern_tuples_of_literals in that case.")

  def test_unintern_tuple_of_literals(self):
    foo_cfg = fdl.Config(Foo, a_tuple=(1, 2, 3), another_tuple=(1, 2, 3))
    uninterned_foo_cfg = transform.unintern_tuples_of_literals(foo_cfg)

    self.assertNotEqual(
        id(uninterned_foo_cfg.a_tuple), id(uninterned_foo_cfg.another_tuple),
        "a_tuple and another_tuple have the same id indicating they're the "
        "same instance, expected them to be different instances after "
        "uninterning.")

  def test_unintern_tuple_of_literals_in_list(self):
    foo_cfg = fdl.Config(Foo, a_tuple=(1, 2, 3), another_tuple=(1, 2, 3))

    uninterned_foo_cfg = transform.unintern_tuples_of_literals(foo_cfg)

    self.assertNotEqual(
        id(uninterned_foo_cfg.a_tuple), id(uninterned_foo_cfg.another_tuple),
        "a_tuple and another_tuple have the same id indicating they're the "
        "same instance, expected them to be different instances after "
        "uninterning.")

  def test_unintern_tuple_of_non_literals(self):
    not_tuple_of_literals = ([1, 2], 3, 4)
    foo_cfg = fdl.Config(
        Foo, a_tuple=not_tuple_of_literals, another_tuple=not_tuple_of_literals)

    uninterned_foo_cfg = transform.unintern_tuples_of_literals(foo_cfg)

    self.assertEqual(
        id(uninterned_foo_cfg.a_tuple), id(uninterned_foo_cfg.another_tuple),
        "a_tuple and another_tuple have different id indicating they're the "
        "different instances, expected them to be the same instance after "
        "uninterning as the these are not tuples of literals.")

  def test_replace_unconfigured_partials_with_callables_replace_top_level(self):
    self.assertIs(
        Foo,
        transform.replace_unconfigured_partials_with_callables(
            fdl.Partial(Foo)))

  def test_replace_unconfigured_partials_with_callables_replace_nested(self):
    cfg = fdl.Config(Foo, a_callable=fdl.Partial(Foo))
    expected_cfg = fdl.Config(Foo, a_callable=Foo)
    self.assertEqual(
        expected_cfg,
        transform.replace_unconfigured_partials_with_callables(cfg))

  def test_replace_unconfigured_partials_with_callables_replace_if_only_defaults(
      self):
    # If an argument is passed to the Partial but it has the same value as the
    # default argument, then still consider it as having no arguments and
    # replace with the original Callable.
    self.assertIs(
        Foo,
        transform.replace_unconfigured_partials_with_callables(
            fdl.Partial(Foo, a_field_with_default=123)))


if __name__ == "__main__":
  absltest.main()
