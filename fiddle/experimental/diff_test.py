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

"""Tests for fiddle.diff."""

import collections
from absl.testing import absltest
import fiddle as fdl
from fiddle.experimental import diff

# Functions and classes that can be used to build Configs.
SimpleClass = collections.namedtuple('SimpleClass', 'x y z'.split())


def make_pair(first, second):
  return (first, second)


def make_triple(first, second, third):
  return (first, second, third)


def basic_fn(arg1, arg2, kwarg1=0, kwarg2=None):
  return {'a': arg1 + arg2, 'b': arg2 + kwarg1, 'c': kwarg2}


class DiffAlignmentTest(absltest.TestCase):

  def test_constructor(self):
    old = fdl.Config(make_pair, fdl.Config(SimpleClass, 1, 2, 3),
                     fdl.Config(basic_fn, 4, 5, 6))
    new = fdl.Config(make_pair, fdl.Config(basic_fn, 1, 2, 3, 4),
                     fdl.Partial(SimpleClass, z=12))

    empty_alignment = diff.DiffAlignment(old, new)

    # No values should be aligned (including the root objects `old` and `new`).
    self.assertEmpty(empty_alignment.aligned_values())
    self.assertEmpty(empty_alignment.aligned_value_ids())
    self.assertFalse(empty_alignment.is_old_value_aligned(old))
    self.assertFalse(empty_alignment.is_new_value_aligned(new))
    self.assertEqual(empty_alignment.old_name, 'old')
    self.assertEqual(empty_alignment.new_name, 'new')

    self.assertEqual(
        repr(empty_alignment),
        "<DiffAlignment from 'old' to 'new': 0 object(s) aligned>")
    self.assertEqual(
        str(empty_alignment), 'DiffAlignment:\n    (no objects aligned)')

  def test_align(self):
    old = fdl.Config(make_pair, fdl.Config(SimpleClass, 1, 2, [3, 4]),
                     fdl.Config(basic_fn, 5, 6, 7))
    new = fdl.Config(make_pair, fdl.Config(basic_fn, 1, 2, 3, 4),
                     fdl.Partial(SimpleClass, z=[12, 13]))
    alignment = diff.DiffAlignment(old, new)
    alignment.align(old, new)  # Same type, same __fn_or_cls__.
    alignment.align(old.first, new.first)  # Different __fn_or_cls__.
    alignment.align(old.first.z, new.second.z)  # Aligned lists.

    self.assertIs(alignment.old_to_new(old), new)
    self.assertIs(alignment.new_to_old(new), old)
    self.assertIs(alignment.old_to_new(old.first), new.first)
    self.assertIs(alignment.new_to_old(new.first), old.first)
    self.assertIs(alignment.old_to_new(old.first.z), new.second.z)
    self.assertIs(alignment.new_to_old(new.second.z), old.first.z)

    with self.subTest('aligned_value_ids'):
      aligned_value_ids = alignment.aligned_value_ids()
      expected_aligned_value_ids = [
          (id(old), id(new)),
          (id(old.first), id(new.first)),
          (id(old.first.z), id(new.second.z)),
      ]
      self.assertCountEqual(aligned_value_ids, expected_aligned_value_ids)

    with self.subTest('aligned_values'):
      aligned_values = alignment.aligned_values()
      expected_aligned_values = [(old, new), (old.first, new.first),
                                 (old.first.z, new.second.z)]
      aligned_values.sort(key=lambda p: id(p[0]))
      expected_aligned_values.sort(key=lambda p: id(p[0]))
      self.assertEqual(aligned_values, expected_aligned_values)

    with self.subTest('__repr__'):
      self.assertEqual(
          repr(alignment),
          "<DiffAlignment from 'old' to 'new': 3 object(s) aligned>")

    with self.subTest('__str__'):
      self.assertEqual(
          str(alignment), '\n'.join([
              'DiffAlignment:',
              '    old -> new',
              '    old.first -> new.first',
              '    old.first.z -> new.second.z',
          ]))

  def test_alignment_errors(self):
    old = fdl.Config(make_pair, fdl.Config(SimpleClass, [1], [2], [3]),
                     fdl.Config(basic_fn, 4, 5, 6))
    new = fdl.Config(make_pair, fdl.Config(basic_fn, [1], [2], 3, 4),
                     fdl.Partial(SimpleClass, z=[12, 13]))

    alignment = diff.DiffAlignment(old, new)
    alignment.align(old.first.x, new.first.arg1)

    with self.subTest('type(old_value) != type(new_value)'):
      with self.assertRaisesRegex(diff.AlignmentError, '.* different types .*'):
        alignment.align(old.second, new.second)

    with self.subTest('old_value already aligned'):
      with self.assertRaisesRegex(
          diff.AlignmentError,
          'An alignment has already been added for old value .*'):
        alignment.align(old.first.x, new.first.arg2)

    with self.subTest('new_value already aligned'):
      with self.assertRaisesRegex(
          diff.AlignmentError,
          'An alignment has already been added for new value .*'):
        alignment.align(old.first.y, new.first.arg1)

    with self.subTest('len(old_value) != len(new_value)'):
      with self.assertRaisesRegex(diff.AlignmentError,
                                  '.* different lengths .*'):
        alignment.align(old.first.z, new.second.z)

    with self.subTest('non-memoizable old_value'):
      with self.assertRaisesRegex(
          diff.AlignmentError,
          'old_value=4 may not be aligned because it is not '
          'memoizable'):
        alignment.align(old.second.arg1, new.second.z)

    with self.subTest('non-memoizable new_value'):
      with self.assertRaisesRegex(
          diff.AlignmentError,
          'new_value=3 may not be aligned because it is not '
          'memoizable'):
        alignment.align(old.first.z, new.first.kwarg1)

  def test_align_by_id(self):
    old = fdl.Config(make_pair, fdl.Config(SimpleClass, 1, 2, [3, 4]),
                     fdl.Config(basic_fn, 5, 6, 7))
    new = fdl.Config(make_pair, old.first,
                     fdl.Partial(SimpleClass, z=old.first.z))
    alignment = diff.align_by_id(old, new)
    self.assertCountEqual(alignment.aligned_values(), [
        (old.first.z, new.second.z),
        (old.first, new.first),
    ])

  def test_align_heuristically(self):
    c = fdl.Config(SimpleClass)  # Shared object (same id) in `old` and `new`
    d = fdl.Config(SimpleClass, x='bop')
    old = fdl.Config(
        make_triple,
        first=fdl.Config(SimpleClass, x=1, y=2, z=[3, 4]),
        second=fdl.Config(basic_fn, arg1=[5], arg2=5, kwarg1=c),
        third=[[1], 2])
    new = fdl.Config(
        make_triple,
        first=fdl.Config(basic_fn, arg1=1, arg2=c, kwarg1=3, kwarg2=4),
        second=fdl.Partial(basic_fn, arg1=[8], arg2=[3, 4], kwarg1=d),
        third=[[1, 2], 2, [3, 4]])
    alignment = diff.align_heuristically(old, new)
    self.assertCountEqual(
        alignment.aligned_values(),
        [
            # Values aligned by id:
            (old.second.kwarg1, new.first.arg2),
            # Values aligned by path:
            (old, new),
            (old.first, new.first),
            (old.second.arg1, new.second.arg1),
            # Values aligned by equality:
            (old.first.z, new.second.arg2),
        ])


if __name__ == '__main__':
  absltest.main()
