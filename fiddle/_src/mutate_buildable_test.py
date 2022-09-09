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

"""Tests for mutate_buildable."""

import dataclasses
from typing import Any

from absl.testing import absltest
import fiddle as fdl
from fiddle._src import mutate_buildable


@dataclasses.dataclass
class SampleClass:
  arg1: Any


class MutateBuildableTest(absltest.TestCase):

  def test_move_buildable_internals_history(self):
    source = fdl.Config(SampleClass)
    source.arg1 = 4
    source.arg1 = 5
    destination = fdl.Config(SampleClass)
    mutate_buildable.move_buildable_internals(
        source=source, destination=destination)
    self.assertEqual(destination.arg1, 5)
    self.assertEqual(destination.__argument_history__['arg1'][0].new_value, 4)
    self.assertEqual(destination.__argument_history__['arg1'][1].new_value, 5)

  def test_no_unexpected_attributes(self):
    sample_config = fdl.Config(SampleClass)
    expected = mutate_buildable._buildable_internals_keys
    self.assertEqual(set(expected), set(sample_config.__dict__.keys()))


if __name__ == '__main__':
  absltest.main()
