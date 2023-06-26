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

"""Tests for strict_validation."""

import dataclasses
from typing import NamedTuple

from absl.testing import absltest
import fiddle as fdl
from fiddle._src import strict_validation


@dataclasses.dataclass
class MyDataclass:
  a: int


class MyNamedTuple(NamedTuple):
  a: int


class StrictValidationTest(absltest.TestCase):

  def test_is_valid_disabled(self):
    self.assertTrue(
        strict_validation.is_valid(
            structure=MyDataclass(4),
            additional_allowed_types=[],
            stop_at_allowed_type=False,
        )
    )

  def test_is_valid_enabled(self):
    with strict_validation.strict_validation():
      self.assertFalse(
          strict_validation.is_valid(
              structure=MyDataclass(4),
              additional_allowed_types=[],
              stop_at_allowed_type=False,
          )
      )


class ConfigIntegrationTest(absltest.TestCase):

  def test_invalid_config(self):
    with strict_validation.strict_validation():
      with self.assertRaisesRegex(ValueError, "Argument 'x' is invalid.*"):
        fdl.Config(lambda x: x, MyDataclass(1))

  def test_invalid_config_namedtuple(self):
    with strict_validation.strict_validation():
      with self.assertRaisesRegex(ValueError, "Argument 'x' is invalid.*"):
        fdl.Config(lambda x: x, MyNamedTuple(1))

  def test_valid_config_plain_tuple(self):
    with strict_validation.strict_validation():
      fdl.Config(lambda x: x, (1,))

  def test_invalid_config_nested(self):
    with strict_validation.strict_validation():
      with self.assertRaisesRegex(ValueError, "Argument 'x' is invalid.*"):
        fdl.Config(lambda x: x, {"inner": MyDataclass(1), "okay": 1})

  def test_valid_config_nested(self):
    with strict_validation.strict_validation():
      fdl.Config(lambda x: x, {"inner": fdl.Config(MyDataclass, 1), "okay": 1})


if __name__ == "__main__":
  absltest.main()
