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
from fiddle._src.validation import baseline_style
from fiddle._src.validation import fake_experiment


class BaselineStyleTest(absltest.TestCase):

  def test_check_baseline_style_okay(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    baseline_style.check_baseline_style(
        config=config, max_config_source_files=1
    )

  def test_check_baseline_style_too_many_files(self):
    config = fake_experiment.fake_experiment()
    with self.assertRaisesRegex(
        ValueError,
        r"More than 1 file\(s\) produced this config.*files which have"
        r" written.*fake_encoder_decoder.*"
        r"fake_experiment.*encoder\.attention",
    ):
      baseline_style.check_baseline_style(
          config=config, max_config_source_files=1
      )

  def test_check_baseline_style_too_many_overrides(self):
    config = fake_experiment.fake_experiment()
    # override dtype one more time to int64 for a total of 3 overrides
    config.encoder.attention.dtype = "int64"
    with self.assertRaisesRegex(
        ValueError,
        r"3 value\(s\) written to <config>.*Please limit the number of"
        r" overrides of this config attribute to 2 to maintain baseline"
        r" configuration readability.",
    ):
      baseline_style.check_baseline_style(
          config=config,
          max_config_source_files=3,
          max_writes_per_attribute=2,
      )

  def test_check_baseline_style_too_many_overrides_for_ignored_attribute(self):
    config = fake_experiment.fake_experiment()
    # override dtype one more time to int64 for a total of 3 overrides
    config.encoder.attention.dtype = "int64"
    # Here we set both max_config_source_files and max_writes_per_attribute
    # to 1. This should cause this test to fail as in the other test cases
    # above, but since we ignore the attribute ".encoder.attention.dtype"
    # validation checks for readabiilty succeed.
    baseline_style.check_baseline_style(
        config=config,
        max_config_source_files=1,
        max_writes_per_attribute=1,
        ignore=[config.encoder.attention.dtype],
    )

  def test_check_baseline_style_ignored_attribute(self):
    # Create a config and write to its encoder.embedders attribute which has
    # been set once before in fake_encoder_decoder.py
    config = fake_encoder_decoder.fixture.as_buildable()
    config.encoder.embedders["tokens"].dtype = "float64"

    # Setting max_writes_per_attribute to 1 should throw an error since the
    # encoder.embedders attribute is being written to twice.
    with self.assertRaisesRegex(
        ValueError,
        r"2 value\(s\) written to <config>.*Please limit the number of"
        r" overrides of this config attribute to 1 to maintain baseline"
        r" configuration readability.",
    ):
      baseline_style.check_baseline_style(
          config=config,
          max_config_source_files=3,
          max_writes_per_attribute=1,
      )

    # However, if the encoder.embedders is set in ignored_attributes, all the
    # validations should succeed.
    baseline_style.check_baseline_style(
        config=config,
        max_config_source_files=3,
        max_writes_per_attribute=1,
        ignore=[
            config.encoder.embedders["tokens"].dtype,
            config.decoder.embedders["tokens"].dtype,
        ],
    )

  def test_check_baseline_style_ignored_non_attribute(self):
    # Create a config and write to its encoder.embedders attribute which has
    # been set once before in fake_encoder_decoder.py
    config = fake_encoder_decoder.fixture.as_buildable()
    config.encoder.embedders["tokens"].dtype = "float64"
    # Create config wrappers that wrap `config`
    foo = {"foo": config}
    bar = {"bar": foo}
    qux = {"qux": bar}

    # Ignoring `bar` here should cause the style checks to not recurse into
    # `foo`, thereby passing the style check that enforces a maximum of 1 write
    # to `config`'s attributes
    baseline_style.check_baseline_style(
        qux, max_writes_per_attribute=1, ignore=[bar]
    )

    # Not ignoring any sub-objects of the `config` wrapper `qux` here should
    # result in the style check code recursing into all attributes of `config`
    # and discover the violation of the style check that enforces a maximum of
    # 1 write to `config`'s attributes
    with self.assertRaisesRegex(
        ValueError,
        r"2 value\(s\) written to <config>.*Please limit the number of"
        r" overrides of this config attribute to 1 to maintain baseline"
        r" configuration readability.",
    ):
      baseline_style.check_baseline_style(qux, max_writes_per_attribute=1)


if __name__ == "__main__":
  absltest.main()
