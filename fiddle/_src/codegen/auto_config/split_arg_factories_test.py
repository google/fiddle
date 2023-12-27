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

"""Tests for split_arg_factories."""

from absl.testing import absltest
import fiddle as fdl
from fiddle._src.codegen.auto_config import code_ir
from fiddle._src.codegen.auto_config import init_task
from fiddle._src.codegen.auto_config import ir_printer
from fiddle._src.codegen.auto_config import split_arg_factories
from fiddle._src.codegen.auto_config import test_fixtures


class SplitArgFactoriesTest(absltest.TestCase):

  def test_lower_arg_factories(self):
    config = fdl.Partial(  # pytype: disable=wrong-arg-types  # use-fiddle-overlay
        test_fixtures.Attention,
        kernel_init=fdl.ArgFactory(
            test_fixtures.initializer, name='const', dtype='float32'
        ),
    )
    task = init_task.init_task(config=config)
    split_arg_factories.lower_arg_factories(task=task)
    code = ir_printer.format_task(task=task)
    self.assertIn('kernel_init=ArgFactoryExpr[fdl.Partial', code)

  def test_lower_nested_arg_factories(self):
    config = fdl.Partial(  # pytype: disable=wrong-arg-types  # use-fiddle-overlay
        test_fixtures.EncoderLayer,
        attention=fdl.ArgFactory(
            test_fixtures.Attention,
            kernel_init=fdl.ArgFactory(
                test_fixtures.initializer, name='const', dtype='float32'
            ),
        ),
    )
    task = init_task.init_task(config=config)
    split_arg_factories.lower_arg_factories(task=task)
    output_value = task.top_level_call.fn.output_value
    self.assertIsInstance(output_value.attention, code_ir.ArgFactoryExpr)
    self.assertIsInstance(output_value.attention.expression, fdl.Partial)
    attention = output_value.attention.expression
    self.assertIsInstance(attention.kernel_init, code_ir.ArgFactoryExpr)
    self.assertIsInstance(attention.kernel_init.expression, fdl.Partial)


if __name__ == '__main__':
  absltest.main()
