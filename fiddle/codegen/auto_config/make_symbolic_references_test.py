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

"""Tests for make_symbolic_references pass."""

from absl.testing import absltest
from fiddle.codegen.auto_config import code_ir
from fiddle.codegen.auto_config import init_task
from fiddle.codegen.auto_config import make_symbolic_references
from fiddle.codegen.auto_config import test_fixtures


class MakeSymbolicReferencesTest(absltest.TestCase):

  def test_import_symbols(self):
    task = test_fixtures.simple_ir()

    # By default, this contains a bunch of builtins.
    task.import_manager.namespace.names = set()

    make_symbolic_references.import_symbols(task)

    # The imported symbols are auto_config, and the fixture module, since it
    # contains the `foo` function.
    self.assertEqual(
        task.import_manager.namespace.names, {'auto_config', 'test_fixtures'}
    )

  def test_import_symbols_empty(self):
    task = init_task.init_task(config=[])
    task.import_manager.namespace.names = set()
    make_symbolic_references.import_symbols(task)
    self.assertEqual(task.import_manager.namespace.names, {'auto_config'})

  def test_replace_config_callable(self):
    task = test_fixtures.simple_ir()
    make_symbolic_references.import_symbols(task)
    make_symbolic_references.replace_callables_and_configs_with_symbols(task)
    self.assertEqual(
        task.top_level_call.fn,
        code_ir.FixtureFunction(
            name=code_ir.Name(
                value='simple_ir_fixture', is_generated=False, previous=None
            ),
            parameters=[],
            variables=[],
            output_value=code_ir.SymbolCall(
                symbol_expression='test_fixtures.foo',
                arg_expressions={'x': 4},
            ),
        ),
    )


if __name__ == '__main__':
  absltest.main()
