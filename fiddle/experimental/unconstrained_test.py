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

"""Tests for UnconstrainedBuildable."""

import dataclasses
from typing import Any

from absl.testing import absltest
import fiddle as fdl
from fiddle import tagging
from fiddle import testing as fdltest
from fiddle.experimental import daglish
from fiddle.experimental import unconstrained


def test_func(x, y):
  return x + y


@dataclasses.dataclass
class TestClass:
  a: Any
  b: Any


class TestTag(tagging.Tag):
  """A tag for testing."""


class UnconstrainedBuildableTest(fdltest.TestCase):

  def test_unconstrain_round_trip(self):
    cfg = fdl.Config(
        test_func, x=[TestTag.new(1)], y=fdl.Partial(TestClass, b=3))
    actual = unconstrained.unconstrain_structure(cfg)
    expected = unconstrained.UnconstrainedBuildable(
        fdl.Config, test_func, {
            'x': [
                unconstrained.UnconstrainedBuildable(tagging.TaggedValue,
                                                     tagging.tagvalue_fn, {
                                                         'tags': set([TestTag]),
                                                         'value': 1
                                                     })
            ],
            'y':
                unconstrained.UnconstrainedBuildable(fdl.Partial, TestClass,
                                                     {'b': 3})
        })
    self.assertDagEqual(actual, expected)

    restored = unconstrained.constrain_structure(actual)
    self.assertDagEqual(restored, cfg)

  def test_unconstrain_and_change(self):
    cfg = fdl.Config(test_func, x=TestTag.new(1))
    unconstrained_cfg = unconstrained.unconstrain_structure(cfg)
    unconstrained_cfg.x.tags = None
    expected = unconstrained.UnconstrainedBuildable(
        fdl.Config, test_func, {
            'x':
                unconstrained.UnconstrainedBuildable(tagging.TaggedValue,
                                                     tagging.tagvalue_fn, {
                                                         'tags': None,
                                                         'value': 1
                                                     })
        })
    self.assertDagEqual(unconstrained_cfg, expected)

    with self.assertRaisesRegex(ValueError,
                                'At least one tag must be provided'):
      unconstrained.constrain_structure(unconstrained_cfg)

  def test_unconstrain_and_change_with_daglish(self):
    cfg = fdl.Config(test_func, x=TestTag.new(1))
    unconstrained_cfg = unconstrained.unconstrain_structure(cfg)

    def replace_set_with_none(path, value):
      del path  # Unused.
      if isinstance(value, set):
        return None
      else:
        return (yield)

    unconstrained_cfg = daglish.traverse_with_path(replace_set_with_none,
                                                   unconstrained_cfg)

    expected = unconstrained.UnconstrainedBuildable(
        fdl.Config, test_func, {
            'x':
                unconstrained.UnconstrainedBuildable(tagging.TaggedValue,
                                                     tagging.tagvalue_fn, {
                                                         'tags': None,
                                                         'value': 1
                                                     })
        })
    self.assertDagEqual(unconstrained_cfg, expected)

    with self.assertRaisesRegex(ValueError,
                                'At least one tag must be provided'):
      unconstrained.constrain_structure(unconstrained_cfg)

  def test_unconstrain_tagged_value(self):
    tagged_value = TestTag.new(1)

    actual = unconstrained.unconstrain_structure(tagged_value)
    expected = unconstrained.UnconstrainedBuildable(tagging.TaggedValue,
                                                    tagging.tagvalue_fn, {
                                                        'tags': set([TestTag]),
                                                        'value': 1
                                                    })
    self.assertDagEqual(actual, expected)

    restored = unconstrained.constrain_structure(actual)
    self.assertDagEqual(restored, tagged_value)


if __name__ == '__main__':
  absltest.main()
