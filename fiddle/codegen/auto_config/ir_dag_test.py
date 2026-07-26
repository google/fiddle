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

"""Tests for ir_dag."""

from absl.testing import absltest
from fiddle import daglish
from fiddle.codegen.auto_config import ir_dag
from fiddle.testing.example import fake_encoder_decoder


class IrDagTest(absltest.TestCase):

  def test_generate_ir_dag(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    ir = ir_dag.generate_ir_dag(config_or_value=config)

    debug_strings = []

    def traverse(node, state: daglish.State):
      if not isinstance(node, ir_dag.IrDagNode):
        raise AssertionError(
            "All IR dag node traversal children should be other IR dag nodes")
      paths = "paths: " + ", ".join(
          daglish.path_str(path) for path in node.all_paths)

      # For now, the set of Daglish memoized traversal paths seems to sometimes
      # be greater due to more sharing of immutables. We probably want less
      # sharing, so this test will probably be adjusted.
      if not set(node.all_paths) <= set(state.get_all_paths()):
        raise ValueError(f"Expected paths to match, got {node.all_paths} and "
                         f"{state.get_all_paths()}")

      value_type = type(node.value)
      if isinstance(value_type, type):
        value_type = value_type.__name__
      value_type = f"value type: {value_type}"  # pytype: disable=missing-parameter
      debug_strings.append(f"Node({paths}, {value_type})")
      state.map_children(node)

    daglish.MemoizedTraversal.run(traverse, ir)
    shared_tokens_paths = (
        ".encoder.embedders['tokens'], .decoder.embedders['tokens']")
    shared_tokens_dtype_paths = (
        ".encoder.embedders['tokens'].dtype, .decoder.embedders['tokens'].dtype"
    )
    self.assertEqual(debug_strings, [
        "Node(paths: , value type: Config)",
        "Node(paths: .encoder, value type: Config)",
        "Node(paths: .encoder.embedders, value type: dict)",
        f"Node(paths: {shared_tokens_paths}, value type: Config)",
        f"Node(paths: {shared_tokens_dtype_paths}, value type: str)",
        "Node(paths: .encoder.embedders['position'], value type: NoneType)",
        "Node(paths: .encoder.attention, value type: Config)",
        "Node(paths: .encoder.attention.kernel_init, value type: str)",
        "Node(paths: .encoder.attention.bias_init, value type: str)",
        "Node(paths: .encoder.mlp, value type: Config)",
        "Node(paths: .encoder.mlp.use_bias, value type: bool)",
        "Node(paths: .encoder.mlp.sharding_axes, value type: list)",
        "Node(paths: .encoder.mlp.sharding_axes[0], value type: str)",
        "Node(paths: .encoder.mlp.sharding_axes[1], value type: str)",
        "Node(paths: .encoder.mlp.sharding_axes[2], value type: str)",
        "Node(paths: .decoder, value type: Config)",
        "Node(paths: .decoder.embedders, value type: dict)",
        "Node(paths: .decoder.self_attention, value type: Config)",
        "Node(paths: .decoder.encoder_decoder_attention, value type: Config)",
        "Node(paths: .decoder.mlp, value type: Config)",
        "Node(paths: .decoder.mlp.sharding_axes, value type: list)"
    ])


if __name__ == "__main__":
  absltest.main()
