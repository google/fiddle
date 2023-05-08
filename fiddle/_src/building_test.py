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

"""Tests for history."""
import dataclasses
import unittest
import warnings
from absl import logging
from absl.testing import absltest

from fiddle._src import building
from fiddle._src import config


@dataclasses.dataclass(frozen=True)
class Foo:
  x: int
  y: int


@dataclasses.dataclass(frozen=True)
class Bar:
  x: int


class NonBuildableLoggingTest(absltest.TestCase, unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._log_level = logging.get_verbosity()
    logging.set_verbosity(logging.WARNING)

  def tearDown(self):
    super().tearDown()
    logging.set_verbosity(self._log_level)

  def _check_log_output(self, log_output):
    log_txt = log_output.output[0]
    self.assertIn('No Buildables found in value passed to', log_txt)

  def test_config(self):
    foo = config.Config(Foo, 1, 2)
    with warnings.catch_warnings(record=True) as log_output:
      warnings.simplefilter('always')
      building.build(foo)
      self.assertEmpty(log_output)

  def test_non_traversable(self):
    with self.assertLogs(level='WARNING') as log_output:
      building.build(Foo)
      self._check_log_output(log_output)

  def test_empty_traversable(self):
    with self.assertLogs(level='WARNING') as log_output:
      building.build({})
      self._check_log_output(log_output)

  def test_traversable_wo_buildable(self):
    with self.assertLogs(level='WARNING') as log_output:
      value = {'a': Foo, 'b': Bar}
      building.build(value)
      self._check_log_output(log_output)

  def test_traversable_w_buildable(self):
    foo_1 = config.Config(Foo, 1, 2)
    foo_2 = config.Config(Foo, 3, 4)
    value = {'a': {'b': foo_1, 'c': foo_2}}
    with warnings.catch_warnings(record=True) as log_output:
      warnings.simplefilter('always')
      building.build(value)
      self.assertEmpty(log_output)


if __name__ == '__main__':
  unittest.main()
