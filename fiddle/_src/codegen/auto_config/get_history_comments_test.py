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

"""Tests for get_history_comments."""

from absl.testing import absltest
from fiddle._src.codegen.auto_config import code_ir
from fiddle._src.codegen.auto_config import get_history_comments
from fiddle._src.testing.example import fake_encoder_decoder


class AddHistoryCommentsTest(absltest.TestCase):

  def test_format_history_for_buildable(self):
    config = fake_encoder_decoder.fixture.as_buildable().encoder.attention
    config.dtype = "bfloat16"
    history = get_history_comments.format_history_for_buildable(
        value=config, shorten_filenames=False, num_history_entries=2
    )
    self.assertIsInstance(history, code_ir.HistoryComments)
    self.assertRegex(
        history.per_field["dtype"],
        r"testing/example/fake_encoder_decoder\.py:\d+:fixture",
    )
    self.assertRegex(
        history.per_field["dtype"],
        r"\d+:test_format_history_for_buildable",
    )

  def test_defunct_no_history(self):
    """Check that the API works even when there are no history entries.

    Generally, history entries should always be present, but it's possible that
    they are not when attributes were added by an internal pass or such.
    """
    config = fake_encoder_decoder.fixture.as_buildable().encoder.attention
    del config.__argument_history__["dtype"]
    config.__argument_history__["kernel_init"] = []
    history = get_history_comments.format_history_for_buildable(config)
    self.assertNotIn("dtype", history.per_field)
    self.assertEqual(history.per_field["kernel_init"], "(no history)")


if __name__ == "__main__":
  absltest.main()
