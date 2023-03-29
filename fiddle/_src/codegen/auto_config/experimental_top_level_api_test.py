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
import types
import uuid

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import fiddle as fdl
from fiddle import daglish
from fiddle import tagging
from fiddle._src.codegen.auto_config import experimental_top_level_api
from fiddle._src.codegen.auto_config import test_fixtures
from fiddle._src.testing import nested_values
from fiddle._src.testing import test_util
from fiddle._src.testing.example import fake_encoder_decoder


def make_arg_factory_config():
  return fdl.Partial(
      test_fixtures.EncoderLayer,
      attention=fdl.ArgFactory(
          test_fixtures.Attention,
          kernel_init=fdl.ArgFactory(
              test_fixtures.initializer, name="const", dtype="float32"
          ),
      ),
  )


class ExperimentalTopLevelApiTest(test_util.TestCase, parameterized.TestCase):

  def get_tempdir(self):
    try:
      flags.FLAGS.test_tmpdir
    except flags.UnparsedFlagAccessError:
      # Sometimes need to initialize flags when running `pytest`.
      flags.FLAGS(sys.argv)
    return self.create_tempdir().full_path

  def _load_code_as_module(self, code: str) -> types.ModuleType:
    # Write the generated config to a module, and load it. We can't just exec()
    # it because auto_config needs access to the source code.
    temp_dir = self.get_tempdir()
    sys.path.append(temp_dir)
    uuid_str = str(hash(uuid.uuid4()))
    file_path = os.path.join(temp_dir, f"config_{uuid_str}.py")
    with open(file_path, "w") as f:
      f.write(code)
    return importlib.import_module(f"config_{uuid_str}")

  @parameterized.named_parameters([
      {
          "testcase_name": "shared_config",
          "config": test_fixtures.unprocessed_shared_config().original_config,
      },
      {
          "testcase_name": "arg_factory_config",
          "config": make_arg_factory_config(),
      },
      {
          "testcase_name": "encoder_decoder_config",
          "config": fake_encoder_decoder.fixture.as_buildable(),
      },
  ])
  def test_generates_equivalent_configs(self, config):
    code = experimental_top_level_api.auto_config_codegen(config)
    module = self._load_code_as_module(code)
    generated_config = module.config_fixture.as_buildable()
    self.assertDagEqual(config, generated_config)

  def test_fuzz(self):
    # TODO(b/272826193): Test on more RNG seeds.
    for i in range(10):
      # TODO(b/269518512): Support tags.
      config = tagging.materialize_tags(
          nested_values.generate_nested_value(random.Random(i)),
          clear_field_tags=True,
      )
      # `auto_config` checks that a Buildable is returned, to avoid user errors.
      has_buildables = any(
          isinstance(value, fdl.Buildable)
          for value, _ in daglish.iterate(config)
      )
      if has_buildables:
        with self.subTest(f"rng_{i}"):
          code = experimental_top_level_api.auto_config_codegen(config)
          module = self._load_code_as_module(code)
          generated_config = module.config_fixture.as_buildable()
          self.assertDagEqual(config, generated_config)


if __name__ == "__main__":
  absltest.main()
