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

"""Tests for dataclasses."""

import dataclasses

from absl.testing import absltest
import fiddle as fdl
from fiddle import daglish
from fiddle._src.testing import test_util
from fiddle._src.testing.example import fake_encoder_decoder
from fiddle.experimental import dataclasses as fdl_dc


@dataclasses.dataclass
class PostInitDataclass:
  a: int = 1
  b: int = 2
  c: int = dataclasses.field(init=False)

  def __post_init__(self):
    self.b += 1
    self.c = self.a + 30


class DataclassesTest(test_util.TestCase):

  def test_convert_dataclasses_to_configs(self):
    model = fake_encoder_decoder.fixture()
    config = fdl_dc.convert_dataclasses_to_configs(model)
    self.assertIsInstance(model, fake_encoder_decoder.FakeEncoderDecoder)
    self.assertIsInstance(config, fdl.Config)
    self.assertEqual(
        fdl.get_callable(config), fake_encoder_decoder.FakeEncoderDecoder
    )
    self.assertIsInstance(config.encoder.mlp, fdl.Config)
    self.assertEqual(
        fdl.get_callable(config.encoder.mlp), fake_encoder_decoder.Mlp
    )
    self.assertEqual(model, fdl.build(config))

  def test_post_init_dataclass_conversion(self):
    dc = PostInitDataclass()
    with self.assertRaisesRegex(ValueError, 'Dataclasses.*__post_init__.*'):
      fdl_dc.convert_dataclasses_to_configs(dc)

    config = fdl_dc.convert_dataclasses_to_configs(dc, allow_post_init=True)
    self.assertNotEqual(dc, fdl.build(config))

  def test_convert_reference_to_dataclass(self):
    value = {'foo': PostInitDataclass}
    self.assertEqual(fdl_dc.convert_dataclasses_to_configs(value), value)

  def test_traverse_dataclass_values(self):
    dc = fake_encoder_decoder.Mlp(
        dtype=32, use_bias=False, sharding_axes=['foo', 'bar']
    )

    def traverse(value, state: daglish.State):
      if isinstance(value, int) and not isinstance(value, bool):
        return {'value': value}
      else:
        return state.map_children(value)

    state = daglish.MemoizedTraversal(
        traverse, dc, fdl_dc.daglish_dataclass_registry
    ).initial_state()
    transformed = traverse(dc, state)
    self.assertEqual(
        transformed,
        fake_encoder_decoder.Mlp(
            dtype={'value': 32}, use_bias=False, sharding_axes=['foo', 'bar']
        ),
    )


if __name__ == '__main__':
  absltest.main()
