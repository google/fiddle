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
from fiddle._src.codegen.auto_config import init_task
from fiddle._src.codegen.auto_config import sub_fixture
from fiddle._src.codegen.auto_config import test_fixtures
from fiddle._src.testing.example import fake_encoder_decoder


class SubFixtureTest(absltest.TestCase):

  def test_find_shared_nodes(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    fixtures = {"fake_encoder": config.encoder, "fake_decoder": config.decoder}
    shared_node_lib, fixture_share_nodes_lib = sub_fixture._find_shared_nodes(
        config, sub_fixtures=fixtures
    )
    shared_node = config.encoder.embedders["tokens"]

    with self.subTest("shared_node_lib"):
      self.assertIn(id(shared_node), shared_node_lib)
      self.assertLen(shared_node_lib, 1)

    with self.subTest("fixture_share_nodes_lib"):
      self.assertLen(fixture_share_nodes_lib, 2)
      for fixture in fixtures.values():
        self.assertIn(id(fixture), fixture_share_nodes_lib)
        self.assertSetEqual(
            fixture_share_nodes_lib[id(fixture)], {id(shared_node)}
        )

  def test_fake_encoder_decoder(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    config.encoder.embedders = None
    config.decoder.embedders = None

    task = init_task.init_task(config)
    sub_fixture.transform_sub_fixtures(
        task, {"fake_encoder": config.encoder, "fake_decoder": config.decoder}
    )
    self.assertLen(task.top_level_call.children, 2)

  def test_nested_sub_fixture(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    task = init_task.init_task(config)
    sub_fixture.transform_sub_fixtures(
        task,
        {"fake_encoder": config.encoder, "attention": config.encoder.attention},
    )
    self.assertLen(task.top_level_call.children, 2)

  def test_conflicting_name_raises_errors(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    task = init_task.init_task(config)
    with self.assertRaisesRegex(
        ValueError, "already exists in the top level fixture"
    ):
      sub_fixture.transform_sub_fixtures(
          task, {"config_fixture": config.encoder}
      )

  # TODO(b/284359119): Support fdl.ArgFactory as sub-fixture.
  def test_arg_factory_raises_errors(self):
    config = fdl.Partial(
        test_fixtures.EncoderLayer,
        attention=fdl.ArgFactory(
            test_fixtures.Attention,
            kernel_init=fdl.ArgFactory(
                test_fixtures.initializer, name="const", dtype="float32"
            ),
        ),
    )
    task = init_task.init_task(config)
    with self.assertRaisesRegex(ValueError, "fdl.ArgFactory is not supported"):
      sub_fixture.transform_sub_fixtures(task, {"attention": config.attention})


if __name__ == "__main__":
  absltest.main()
