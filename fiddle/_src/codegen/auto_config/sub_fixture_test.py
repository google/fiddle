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

"""Tests for transform_sub_fixtures pass."""

from absl.testing import absltest
import fiddle as fdl
from fiddle._src.codegen.auto_config import code_ir
from fiddle._src.codegen.auto_config import init_task
from fiddle._src.codegen.auto_config import sub_fixture
from fiddle._src.codegen.auto_config import test_fixtures
from fiddle._src.testing.example import fake_encoder_decoder


class SubFixtureTest(absltest.TestCase):

  def test_fake_encoder_decoder(self):
    cfg = fake_encoder_decoder.fixture.as_buildable()
    cfg.encoder.embedders = None
    cfg.decoder.embedders = None

    task = init_task.init_task(cfg)
    sub_fixture.transform_sub_fixtures(
        task, {"fake_encoder": cfg.encoder, "fake_decoder": cfg.decoder}
    )
    self.assertLen(task.top_level_call.children, 2)

    with self.subTest("call_nodes"):
      names = ["fake_encoder", "fake_decoder"]
      for idx, node in enumerate(task.top_level_call.children):
        self.assertIsInstance(node, code_ir.Call)
        self.assertIsInstance(node.name, code_ir.Name)
        self.assertEqual(node.name.value, names[idx])

  def test_arg_factory(self):
    cfg = fdl.Partial(
        test_fixtures.EncoderLayer,
        attention=fdl.ArgFactory(
            test_fixtures.Attention,
            kernel_init=fdl.ArgFactory(
                test_fixtures.initializer, name="const", dtype="float32"
            ),
        ),
    )
    task = init_task.init_task(cfg)
    with self.assertRaisesRegex(ValueError, "fdl.ArgFactory is not supported"):
      sub_fixture.transform_sub_fixtures(task, {"attention": cfg.attention})

  def test_nested_sub_fixture(self):
    cfg = fake_encoder_decoder.fixture.as_buildable()
    cfg.encoder.embedders = None
    cfg.decoder.embedders = None

    task = init_task.init_task(cfg)
    sub_fixture.transform_sub_fixtures(
        task, {"fake_encoder": cfg.encoder, "attention": cfg.encoder.attention}
    )
    self.assertLen(task.top_level_call.children, 2)

  def test_conflicting_name_raises_errors(self):
    cfg = fake_encoder_decoder.fixture.as_buildable()
    task = init_task.init_task(cfg)
    with self.assertRaisesRegex(
        ValueError, "already exists in the top level fixture"
    ):
      sub_fixture.transform_sub_fixtures(task, {"config_fixture": cfg.encoder})


if __name__ == "__main__":
  absltest.main()
