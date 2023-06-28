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

"""Tests for baseline_style."""


from absl.testing import absltest
from fiddle._src.testing.example import fake_encoder_decoder
from fiddle._src.validation import fake_experiment
from fiddle.validation import baseline_style


class BaselineStyleTest(absltest.TestCase):

  def test_check_baseline_style_okay(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    baseline_style.check_baseline_style(
        config=config, max_files_writing_attributes=1
    )

  def test_check_baseline_style_too_many_files(self):
    config = fake_experiment.fake_experiment()
    with self.assertRaisesRegex(
        ValueError,
        r"More than 1 file\(s\) produced this config.*files which have"
        r" written.*fake_encoder_decoder.*\.encoder.*"
        r"fake_experiment.*encoder\.attention\.dtype",
    ):
      baseline_style.check_baseline_style(
          config=config, max_files_writing_attributes=1
      )


if __name__ == "__main__":
  absltest.main()
