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
from fiddle import tagging
from fiddle.testing.example import fake_encoder_decoder


class SampleTag(tagging.Tag):
  """A sample tag."""


class AdditionalTag(tagging.Tag):
  """An extra tag."""


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
    entry = history.new_value("x", 1)
    self.assertEqual(entry.param_name, "x")
    self.assertEqual(entry.kind, history.ChangeKind.NEW_VALUE)
    self.assertEqual(entry.new_value, 1)

  def test_entry_deletion(self):
    entry = history.deleted_value("y")
    self.assertEqual(entry.param_name, "y")
    self.assertEqual(entry.kind, history.ChangeKind.NEW_VALUE)
    self.assertEqual(entry.new_value, history.DELETED)

  def test_updating_tags(self):
    tag_set = {SampleTag, AdditionalTag}
    entry = history.update_tags("z", tag_set)
    self.assertEqual(entry.param_name, "z")
    self.assertEqual(entry.kind, history.ChangeKind.UPDATE_TAGS)
    self.assertIsNot(tag_set, entry.new_value)  # Must not be the same!
    self.assertEqual(frozenset(tag_set), entry.new_value)

  def test_location_provider(self):
    entry = history.new_value("x", 123)
    self.assertRegex(entry.location.filename, "history_test.py")
    self.assertEqual(entry.location.function_name, "test_location_provider")

    self.assertEqual(entry.param_name, "x")
    self.assertEqual(entry.new_value, 123)

  def test_custom_location_provider(self):
    e1 = history.new_value("x", 1)
    with history.custom_location(
        lambda: history.Location("other.py", 123, "foo")):
      e2 = history.new_value("y", 2)
    e3 = history.new_value("z", 3)

    self.assertRegex(e1.location.filename, "history_test.py")
    self.assertEqual(e2.location.filename, "other.py")
    self.assertRegex(e1.location.filename, "history_test.py")

    self.assertEqual(e2.location.line_number, 123)
    self.assertEqual(e1.location.line_number + 4, e3.location.line_number)

    self.assertEqual(e1.location.function_name, "test_custom_location_provider")
    self.assertEqual(e2.location.function_name, "foo")
    self.assertEqual(e3.location.function_name, "test_custom_location_provider")

  def test_deleted_repr(self):
    self.assertEqual(repr(history.DELETED), "DELETED")

  def test_auto_config_history_location(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    entry, = config.__argument_history__["encoder"]
    self.assertEndsWith(entry.location.filename, "fake_encoder_decoder.py")

  def test_suspend_tracking_new_value(self):
    arg_history = history.History()
    arg_history.add_new_value("arg_name", 1)
    self.assertLen(arg_history["arg_name"], 1)
    self.assertEqual(arg_history["arg_name"][-1].new_value, 1)
    with history.suspend_tracking():
      arg_history.add_new_value("arg_name", 2)
    self.assertLen(arg_history["arg_name"], 1)

  def test_suspend_tracking_deleted_value(self):
    arg_history = history.History()
    arg_history.add_new_value("arg_name", 1)
    self.assertLen(arg_history["arg_name"], 1)
    with history.suspend_tracking():
      arg_history.add_deleted_value("arg_name")
    self.assertLen(arg_history["arg_name"], 1)
    arg_history.add_deleted_value("arg_name")
    self.assertLen(arg_history["arg_name"], 2)
    self.assertEqual(arg_history["arg_name"][-1].kind,
                     history.ChangeKind.NEW_VALUE)
    self.assertEqual(arg_history["arg_name"][-1].new_value, history.DELETED)

  def test_suspend_tracking_update_tags(self):
    arg_history = history.History()
    arg_history.add_updated_tags("arg_name", {SampleTag})
    self.assertLen(arg_history["arg_name"], 1)
    with history.suspend_tracking():
      arg_history.add_updated_tags("arg_name", {SampleTag, AdditionalTag})
    self.assertLen(arg_history["arg_name"], 1)
    arg_history.add_updated_tags("arg_name", {SampleTag, AdditionalTag})
    self.assertLen(arg_history["arg_name"], 2)
    self.assertEqual(arg_history["arg_name"][-1].kind,
                     history.ChangeKind.UPDATE_TAGS)
    self.assertEqual(arg_history["arg_name"][-1].new_value,
                     {SampleTag, AdditionalTag})


if __name__ == "__main__":
  absltest.main()
