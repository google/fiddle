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

"""Tests for daglish_extensions."""


import dataclasses
from typing import Any, NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import fiddle as fdl
from fiddle import daglish
from fiddle._src import daglish_extensions


class SampleNamedTuple(NamedTuple):
  fizz: Any
  buzz: Any


class IsImmutableTest(absltest.TestCase):

  def test_frozen_dataclass(self):
    @dataclasses.dataclass(frozen=True)
    class FrozenFoo:
      bar: Any

    foo = FrozenFoo(1)
    self.assertFalse(daglish_extensions.is_immutable(foo))

  def test_named_tuple(self):
    tuple_foo = SampleNamedTuple(1, 2)
    self.assertTrue(daglish_extensions.is_immutable(tuple_foo))


class DaglishExtensionsTest(parameterized.TestCase):

  def test_register_immutable(self):
    obj = object()
    self.assertFalse(daglish_extensions.is_immutable(obj))
    daglish_extensions.register_immutable(obj)
    self.assertTrue(daglish_extensions.is_immutable(obj))

  def test_register_function_with_immutable_return_value(self):
    def fn(x: int) -> int:
      return x

    config = fdl.Config(fn, 3)
    self.assertFalse(daglish_extensions.is_immutable(config))
    self.assertFalse(daglish_extensions.is_unshareable(config))
    daglish_extensions.register_function_with_immutable_return_value(fn)

    # The config should still be marked as mutable, but is unshareable. In other
    # words, the config API would change by unsharing `config`, but its return
    # value would not.
    self.assertFalse(daglish_extensions.is_immutable(config))
    self.assertTrue(daglish_extensions.is_unshareable(config))

  @parameterized.parameters(
      {
          "path": ".foo.bar",
          "expected": (daglish.Attr("foo"), daglish.Attr("bar")),
      },
      {
          "path": ".asd_fds_1",
          "expected": (daglish.Attr("asd_fds_1"),),
      },
      {
          "path": ".foo[3].bar",
          "expected": (
              daglish.Attr("foo"),
              daglish.Key(3),
              daglish.Attr("bar"),
          ),
      },
      {
          "path": "[0]['foo'][\"bar\"]",
          "expected": (
              daglish.Key(0),
              daglish.Key("foo"),
              daglish.Key("bar"),
          ),
      },
  )
  def test_parse_path(self, path, expected):
    actual = daglish_extensions.parse_path(path)
    self.assertEqual(actual, expected)

  @parameterized.parameters(
      "<foobar>.baz<qux>[3]",
      "..fdsa",
      ".[123]",
  )
  def test_bad_paths(self, path):
    with self.assertRaises(ValueError):
      daglish_extensions.parse_path(path)


if __name__ == "__main__":
  absltest.main()
