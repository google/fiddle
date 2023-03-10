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

"""Tests for init_task."""

from absl.testing import absltest

import fiddle as fdl
from fiddle._src.codegen.auto_config import init_task
from fiddle._src.codegen.auto_config import test_fixtures


class InitTaskTest(absltest.TestCase):

  def test_equal_to_fixture(self):
    fixture_task = test_fixtures.simple_ir()
    config = fdl.Config(test_fixtures.foo, x=4)
    initialized_task = init_task.init_task(
        config, top_level_fixture_name="simple_ir_fixture"
    )
    self.assertEqual(initialized_task, fixture_task)


if __name__ == "__main__":
  absltest.main()
