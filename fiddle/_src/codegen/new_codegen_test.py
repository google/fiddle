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

"""Tests for new_codegen."""

import importlib
import os
import re
import sys
import types
import uuid

from absl import flags
from absl.testing import absltest
from fiddle._src.codegen import new_codegen
from fiddle._src.testing.example import fake_encoder_decoder


class NewCodegenTest(absltest.TestCase):

  def get_tempdir(self):
    try:
      flags.FLAGS.test_tmpdir
    except flags.UnparsedFlagAccessError:
      # Sometimes need to initialize flags when running `pytest`.
      program, *rest = sys.argv
      rest = [flag for flag in rest if "test_tmpdir" in flag]
      flags.FLAGS([program, *rest])
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

  def test_codegen(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    code = new_codegen.new_codegen(config=config, max_expression_complexity=4)
    self.assertNotIn(code, "auto_config")
    fixture = self._load_code_as_module(code).config_fixture
    self.assertEqual(fixture(), config)

  def test_sub_fixtures(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    code = new_codegen.new_codegen(
        config=config,
        max_expression_complexity=4,
        sub_fixtures={
            "encoder_fixture": config.encoder,
            "decoder_fixture": config.decoder,
        },
    )

    self.assertNotIn(code, "auto_config")

    # Ensure that sub-fixtures were actually created.
    matches = re.findall(r"def\ (?P<name>[\w_]+)\(", code)
    self.assertEqual(
        matches, ["config_fixture", "encoder_fixture", "decoder_fixture"]
    )

    fixture = self._load_code_as_module(code).config_fixture
    self.assertEqual(fixture(), config)

  def test_code_output(self):
    config = fake_encoder_decoder.fixture.as_buildable().encoder
    code = new_codegen.new_codegen(
        config=config,
        max_expression_complexity=4,
    )
    expected = """
    import fiddle as fdl
    from fiddle._src.testing.example import fake_encoder_decoder


    def config_fixture() -> fdl.Config[fake_encoder_decoder.FakeEncoder]:
        mlp = fdl.Config(fake_encoder_decoder.Mlp, dtype='float32',
          use_bias=False, sharding_axes=['embed', 'num_heads', 'head_dim'])
        return fdl.Config(fake_encoder_decoder.FakeEncoder, embedders={'tokens':
          fdl.Config(fake_encoder_decoder.TokenEmbedder, dtype='float32'),
          'position': None},
          attention=fdl.Config(fake_encoder_decoder.Attention, dtype='float32',
          kernel_init='uniform()', bias_init='zeros()'), mlp=mlp)
    """
    self.assertEqual(code.split(), expected.split())


if __name__ == "__main__":
  absltest.main()
