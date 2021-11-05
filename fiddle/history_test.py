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

from absl.testing import absltest
from fiddle import history


class HistoryTest(absltest.TestCase):

  def test_location_formatting(self):
    location = history.Location(
        filename="my_file.py", line_number=123, function_name=None)
    self.assertEqual(str(location), "my_file.py:123")
    location = history.Location(
        filename="my_other_file.py",
        line_number=321,
        function_name="make_config")
    self.assertEqual(str(location), "my_other_file.py:321:make_config")

  def test_entry_simple(self):
    entry = history.entry("x", 1)
    self.assertEqual(entry.param_name, "x")
    self.assertEqual(entry.value, 1)

  def test_entry_deletion(self):
    entry = history.entry("y", history.DELETED)
    self.assertEqual(entry.param_name, "y")
    self.assertEqual(entry.value, history.DELETED)

  def test_location_provider(self):
    entry = history.entry("x", 123)
    self.assertRegex(entry.location.filename, "history_test.py")
    self.assertEqual(entry.location.function_name, "test_location_provider")

    self.assertEqual(entry.param_name, "x")
    self.assertEqual(entry.value, 123)

  def test_custom_location_provider(self):
    e1 = history.entry("x", 1)
    with history.custom_location(
        lambda: history.Location("other.py", 123, "foo")):
      e2 = history.entry("y", 2)
    e3 = history.entry("z", 3)

    self.assertRegex(e1.location.filename, "history_test.py")
    self.assertEqual(e2.location.filename, "other.py")
    self.assertRegex(e1.location.filename, "history_test.py")

    self.assertEqual(e2.location.line_number, 123)
    self.assertEqual(e1.location.line_number + 4, e3.location.line_number)

    self.assertEqual(e1.location.function_name, "test_custom_location_provider")
    self.assertEqual(e2.location.function_name, "foo")
    self.assertEqual(e3.location.function_name, "test_custom_location_provider")


if __name__ == "__main__":
  absltest.main()
