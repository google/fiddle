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

"""Tests for reraised_exception."""

import pickle

from absl.testing import absltest
from fiddle._src import reraised_exception


def foo():
  raise ValueError("test exception")


def bar():
  with reraised_exception.try_with_lazy_message(lambda: " - test context"):
    foo()


class CannotBeSubclassedError(Exception):

  def __init_subclass__(cls):
    raise ValueError("This class is final.")


class ReraisedExceptionTest(absltest.TestCase):

  def test_reraise_value_error(self):
    self.assertRaises(ValueError, bar)
    try:
      bar()
    except ValueError as e:
      # pylint: disable=g-assert-in-except
      self.assertEqual(e.proxy_message, " - test context")  # pytype: disable=attribute-error
      self.assertEqual(str(e), "test exception - test context")
      # pylint: enable=g-assert-in-except

  def test_decorate_subclassing(self):
    original = ValueError()
    decorated = reraised_exception.decorate_exception(original, " msg")
    self.assertIsNot(decorated, original)

  def test_decorate_subclassing_fails(self):
    original = CannotBeSubclassedError()
    decorated = reraised_exception.decorate_exception(original, "unused")
    self.assertIs(decorated, original)

  def test_pickling(self):
    original = ValueError()
    decorated = reraised_exception.decorate_exception(original, " msg")
    serialized = pickle.dumps(decorated)
    deserialized = pickle.loads(serialized)
    self.assertIs(type(deserialized), type(decorated))
    self.assertIsNot(type(deserialized), type(original))
    self.assertEqual(deserialized.proxy_message, " msg")


if __name__ == "__main__":
  absltest.main()
