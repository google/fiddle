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

"""Tests for validators."""

import collections
import inspect

from absl.testing import absltest

import fiddle as fdl
from fiddle.experimental import validators


def test_fn(a, b, c='abc'):  # pylint: disable=unused-argument
  pass


# pylint: disable=g-error-prone-assert-raises


class AllPrimitivesValidatorsTest(absltest.TestCase):

  def test_allow_simple(self):
    config = fdl.Config(test_fn)
    validators.assert_all_primitive_leaves(config)

    config.a = 5
    validators.assert_all_primitive_leaves(config)

    config.b = 10
    validators.assert_all_primitive_leaves(config)

    config.b = inspect.signature(test_fn)
    with self.assertRaisesRegex(
        validators.ValidationFailureError,
        r'ValidationFailure: Inappropriate values in Config.*\n'
        r'  - .b: inspect.Signature'):
      validators.assert_all_primitive_leaves(config)

  def test_allow_hierarchical(self):
    config = fdl.Config(test_fn, 'a')
    config.b = (0, {'inner': fdl.Config(test_fn)})
    validators.assert_all_primitive_leaves(config)

    config.b[1]['inner'].a = 'very_inner'
    validators.assert_all_primitive_leaves(config)

    config.b[1]['something_else'] = inspect.signature(test_fn)
    with self.assertRaises(validators.ValidationFailureError):
      validators.assert_all_primitive_leaves(config)

    del config.b[1]['something_else']
    validators.assert_all_primitive_leaves(config)

    config.b[1]['inner'].b = inspect.signature(test_fn)
    with self.assertRaises(validators.ValidationFailureError):
      validators.assert_all_primitive_leaves(config)

  def test_allowed_nameduples(self):
    TestTuple = collections.namedtuple('TestTuple', 'x y')
    config = fdl.Config(test_fn)
    config.a = TestTuple(5, 10)
    validators.assert_all_primitive_leaves(config)

    config.a = config.a._replace(y=inspect.signature(test_fn))
    with self.assertRaises(validators.ValidationFailureError):
      validators.assert_all_primitive_leaves(config)


class NoAliasingValidatorsTest(absltest.TestCase):

  def test_simple(self):
    config = fdl.Config(test_fn)
    validators.assert_no_aliasing(config)
    config.a = 5
    config.b = 'abc'
    validators.assert_no_aliasing(config)

    # Make a simple alias.
    config.a = fdl.Config(test_fn)
    config.b = config.a

    with self.assertRaises(validators.ValidationFailureError):
      validators.assert_no_aliasing(config)


class RequiredValuesSetValidatorsTest(absltest.TestCase):

  def test_simple(self):
    config = fdl.Config(test_fn)

    with self.assertRaisesRegex(validators.ValidationFailureError,
                                r'Missing required parameters'):
      validators.assert_all_required_values_set(config)

    config.a = 1
    config.b = 2
    validators.assert_all_required_values_set(config)


# pylint: enable=g-error-prone-assert-raises

if __name__ == '__main__':
  absltest.main()
