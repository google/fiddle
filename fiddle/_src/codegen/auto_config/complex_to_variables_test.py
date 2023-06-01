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

from absl.testing import absltest
import fiddle as fdl
from fiddle._src.codegen.auto_config import code_ir
from fiddle._src.codegen.auto_config import complex_to_variables
from fiddle._src.codegen.auto_config import init_task
from fiddle._src.codegen.auto_config import ir_printer
from fiddle._src.codegen.auto_config import shared_to_variables
from fiddle._src.codegen.auto_config import split_arg_factories
from fiddle._src.codegen.auto_config import test_fixtures
from fiddle._src.testing.example import fake_encoder_decoder


class MoveComplexNodesToVariablesTest(absltest.TestCase):

  def test_medium_extraction(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    task = init_task.init_task(config)
    self.assertEmpty(task.top_level_call.fn.variables)
    shared_to_variables.move_shared_nodes_to_variables(task)
    self.assertLen(task.top_level_call.fn.variables, 1)
    complex_to_variables.move_complex_nodes_to_variables(
        task, is_complex=complex_to_variables.more_complex_than(10)
    )
    self.assertLen(task.top_level_call.fn.variables, 3)
    code = ir_printer.format_task(task)
    self.assertIn("fake_encoder =", code)
    self.assertIn("fake_decoder =", code)

  def test_high_extraction(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    task = init_task.init_task(config)
    self.assertEmpty(task.top_level_call.fn.variables)
    shared_to_variables.move_shared_nodes_to_variables(task)
    self.assertLen(task.top_level_call.fn.variables, 1)
    complex_to_variables.move_complex_nodes_to_variables(
        task, is_complex=complex_to_variables.more_complex_than(4)
    )
    self.assertLen(task.top_level_call.fn.variables, 5)
    code = ir_printer.format_task(task)
    self.assertIn("mlp =", code)
    self.assertIn("mlp_2 =", code)

  def test_extract_everything(self):
    """Test that the stage doesn't fail if everything is extracted."""
    config = fake_encoder_decoder.fixture.as_buildable()
    task = init_task.init_task(config)
    self.assertEmpty(task.top_level_call.fn.variables)
    shared_to_variables.move_shared_nodes_to_variables(task)
    self.assertLen(task.top_level_call.fn.variables, 1)
    complex_to_variables.move_complex_nodes_to_variables(
        task, is_complex=lambda x: True
    )
    self.assertLen(task.top_level_call.fn.variables, 13)
    code = ir_printer.format_task(task)
    self.assertIn("self_attention =", code)
    self.assertIn("cross_attention =", code)
    self.assertIn("sharding_axes =", code)
    self.assertIn("sharding_axes_2 =", code)

  def test_doesnt_fail_if_cant_name(self):
    config = [[0] * 20 for _ in range(20)]
    task = init_task.init_task(config)
    self.assertEmpty(task.top_level_call.fn.variables)
    shared_to_variables.move_shared_nodes_to_variables(task)
    self.assertEmpty(task.top_level_call.fn.variables)
    complex_to_variables.move_complex_nodes_to_variables(
        task, is_complex=lambda x: True
    )
    self.assertEmpty(task.top_level_call.fn.variables)

  def test_doesnt_break_arg_factory_expressions(self):
    config = fdl.Partial(
        test_fixtures.Attention,
        kernel_init=fdl.ArgFactory(
            test_fixtures.initializer, name="const", dtype="float32"
        ),
    )
    task = init_task.init_task(config=config)
    split_arg_factories.lower_arg_factories(task=task)
    complex_to_variables.move_complex_nodes_to_variables(
        task, is_complex=lambda x: True
    )
    code = ir_printer.format_task(task)
    self.assertIn("kernel_init=ArgFactoryExpr[initializer]", code)
    fn = task.top_level_call.fn
    self.assertIsInstance(fn.output_value.kernel_init, code_ir.ArgFactoryExpr)


if __name__ == "__main__":
  absltest.main()
