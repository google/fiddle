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

import importlib
import os
import random
import sys
import uuid

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from fiddle._src.codegen.auto_config import experimental_top_level_api
from fiddle._src.codegen.auto_config import test_fixtures
from fiddle._src.testing import nested_values
from fiddle._src.testing import test_util
from fiddle._src.testing.example import fake_encoder_decoder


class ExperimentalTopLevelApiTest(test_util.TestCase, parameterized.TestCase):

  def get_tempdir(self):
    try:
      flags.FLAGS.test_tmpdir
    except flags.UnparsedFlagAccessError:
      # Sometimes need to initialize flags when running `pytest`.
      flags.FLAGS(sys.argv)
    return self.create_tempdir().full_path

  @parameterized.named_parameters(
      [
          {
              "testcase_name": "shared_config",
              "config": (
                  test_fixtures.unprocessed_shared_config().original_config
              ),
          },
          {
              "testcase_name": "encoder_decoder_config",
              "config": fake_encoder_decoder.fixture.as_buildable(),
          },
          {
              "testcase_name": "random_config_0",
              "config": nested_values.generate_nested_value(random.Random(0)),
          },
          # TODO(b/269518512): Support tags.
          # {
          #     "testcase_name": "random_config_5",
          #     "config": nested_values.generate_nested_value(random.Random(5)),
          # },
          # {
          #     "testcase_name": "random_config_7",
          #     "config": nested_values.generate_nested_value(random.Random(7)),
          # },
      ]
  )
  def test_generates_equivalent_configs(self, config):
    code = experimental_top_level_api.auto_config_codegen(config)

    # Write the generated config to a module, and load it. We can't just exec()
    # it because auto_config needs access to the source code.
    temp_dir = self.get_tempdir()
    sys.path.append(temp_dir)
    uuid_str = str(hash(uuid.uuid4()))
    file_path = os.path.join(temp_dir, f"config_{uuid_str}.py")
    with open(file_path, "w") as f:
      f.write(code)
    module = importlib.import_module(f"config_{uuid_str}")
    generated_config = module.config_fixture.as_buildable()

    # Check that the generated auto_config fixture produces the same config.
    self.assertDagEqual(config, generated_config)


if __name__ == "__main__":
  absltest.main()
