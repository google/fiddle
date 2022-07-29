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

"""Tests for daglish_generator."""

from absl.testing import absltest

import fiddle as fdl
from fiddle.experimental import daglish
from fiddle.experimental import daglish_generator


def test_fn(a, b, c, d='default'):  # pylint: disable=unused-argument
  return locals()


def simple_fn(x, y='y'):  # pylint: disable=unused-argument
  pass


class DaglishGeneratorTest(absltest.TestCase):

  def test_simple(self):
    config = fdl.Config(test_fn, 1)
    config.b = 'abc'

    entries = list(daglish_generator.iterate(config))
    self.assertEqual(entries, [
        daglish_generator.IterationItem((daglish.Attr('a'),), config),
        daglish_generator.IterationItem((daglish.Attr('b'),), config),
        daglish_generator.IterationItem((daglish.Attr('c'),), config),
        daglish_generator.IterationItem((daglish.Attr('d'),), config),
    ])

  def test_shallow(self):
    config = fdl.Config(test_fn, 1)
    config.b = 'xyz'

    a, b, c, d = tuple(daglish_generator.iterate(config))

    self.assertTrue(a.is_set)
    self.assertEqual(1, a.value)
    a.value = 2

    self.assertTrue(b.is_set)
    self.assertTrue(b.is_leaf)
    self.assertFalse(b.is_collection)
    self.assertEqual('xyz', b.value)
    del b.value
    self.assertFalse(b.is_set)
    b.value = 'abc'
    self.assertTrue(b.is_set)

    self.assertFalse(c.is_set)
    c.value = 'dynamically_set!'
    self.assertEqual('dynamically_set!', c.value)

    self.assertTrue(d.is_set)  # A default means it's set.
    self.assertEqual(d.value, 'default')
    d.value = 'not_default'
    self.assertEqual(d.value, 'not_default')
    self.assertTrue(d.is_set)

  def test_hierarchical_only_buildables_pre_order(self):
    outer = fdl.Config(test_fn, 1, 2)
    outer.c = fdl.Config(test_fn, 'a', 'b', 'c')

    entries = list(
        daglish_generator.iterate(
            outer, options=daglish_generator.IterateOptions.PRE_ORDER))

    self.assertEqual([e.path for e in entries], [
        (daglish.Attr('a'),),
        (daglish.Attr('b'),),
        (daglish.Attr('c'),),
        (daglish.Attr('c'), daglish.Attr('a')),
        (daglish.Attr('c'), daglish.Attr('b')),
        (daglish.Attr('c'), daglish.Attr('c')),
        (daglish.Attr('c'), daglish.Attr('d')),
        (daglish.Attr('d'),),
    ])

    self.assertTrue(entries[0].is_leaf)  # 'a'
    self.assertTrue(entries[1].is_leaf)  # 'b'
    self.assertTrue(entries[2].is_collection)  # 'c'

  def test_hierarchical_only_buildables_post_order(self):
    outer = fdl.Config(test_fn, 1, 2)
    outer.c = fdl.Config(test_fn, 'a', 'b', 'c')

    entries = list(
        daglish_generator.iterate(
            outer, options=daglish_generator.IterateOptions.POST_ORDER))

    self.assertEqual([e.path for e in entries], [
        (daglish.Attr('a'),),
        (daglish.Attr('b'),),
        (daglish.Attr('c'), daglish.Attr('a')),
        (daglish.Attr('c'), daglish.Attr('b')),
        (daglish.Attr('c'), daglish.Attr('c')),
        (daglish.Attr('c'), daglish.Attr('d')),
        (daglish.Attr('c'),),
        (daglish.Attr('d'),),
    ])

  def test_hierarchical_builtin_collections(self):
    outer = fdl.Config(test_fn, 1)
    outer.b = (fdl.Config(simple_fn, 'b1'), fdl.Config(simple_fn, 'b2'))
    outer.c = {
        'p': [fdl.Config(simple_fn, 'cp0x'),
              fdl.Config(simple_fn, 'cp1x')],
        'q': {
            'qq': fdl.Config(simple_fn, 'cqqx')
        }
    }

    entries = list(
        daglish_generator.iterate(
            outer, options=daglish_generator.IterateOptions.PRE_ORDER))

    self.assertEqual([e.path for e in entries], [
        (daglish.Attr('a'),),
        (daglish.Attr('b'),),
        (daglish.Attr('b'), daglish.Index(0)),
        (daglish.Attr('b'), daglish.Index(0), daglish.Attr('x')),
        (daglish.Attr('b'), daglish.Index(0), daglish.Attr('y')),
        (daglish.Attr('b'), daglish.Index(1)),
        (daglish.Attr('b'), daglish.Index(1), daglish.Attr('x')),
        (daglish.Attr('b'), daglish.Index(1), daglish.Attr('y')),
        (daglish.Attr('c'),),
        (daglish.Attr('c'), daglish.Key('p')),
        (daglish.Attr('c'), daglish.Key('p'), daglish.Index(0)),
        (daglish.Attr('c'), daglish.Key('p'), daglish.Index(0),
         daglish.Attr('x')),
        (daglish.Attr('c'), daglish.Key('p'), daglish.Index(0),
         daglish.Attr('y')),
        (daglish.Attr('c'), daglish.Key('p'), daglish.Index(1)),
        (daglish.Attr('c'), daglish.Key('p'), daglish.Index(1),
         daglish.Attr('x')),
        (daglish.Attr('c'), daglish.Key('p'), daglish.Index(1),
         daglish.Attr('y')),
        (daglish.Attr('c'), daglish.Key('q')),
        (daglish.Attr('c'), daglish.Key('q'), daglish.Key('qq')),
        (daglish.Attr('c'), daglish.Key('q'), daglish.Key('qq'),
         daglish.Attr('x')),
        (daglish.Attr('c'), daglish.Key('q'), daglish.Key('qq'),
         daglish.Attr('y')),
        (daglish.Attr('d'),),
    ])
    self.assertEqual('b1', entries[3].value)


if __name__ == '__main__':
  absltest.main()
