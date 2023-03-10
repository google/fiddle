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
import fiddle as fdl
from fiddle._src.codegen.auto_config import code_ir
from fiddle._src.codegen.auto_config import init_task
from fiddle._src.codegen.auto_config import ir_to_cst
from fiddle._src.codegen.auto_config import make_symbolic_references
from fiddle._src.codegen.auto_config import test_fixtures


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
                positional_arg_expressions=[],
                arg_expressions={'x': 4},
            ),
        ),
    )

  def test_replaces_arg_factory_partial(self):
    # This test goes from auto_config to a config object and back again.

    config = test_fixtures.auto_config_arg_factory_fn.as_buildable()
    task = init_task.init_task(config=config)
    make_symbolic_references.import_symbols(task)
    make_symbolic_references.replace_callables_and_configs_with_symbols(task)
    code = ir_to_cst.code_for_task(task).code

    expected = """
    from fiddle._src.codegen.auto_config import test_fixtures
    from fiddle._src.experimental import auto_config
    from fiddle import arg_factory
    import functools


    @auto_config.auto_config
    def config_fixture():
        return functools.partial(arg_factory.partial(test_fixtures.SharedType,
            x=functools.partial(test_fixtures.count, increment=3)), z=4.7)
    """
    self.assertEqual(code.split(), expected.split(), msg=code)

  def test_two_arg_factories(self):
    config = fdl.Partial(
        test_fixtures.SharedType,
        x=fdl.ArgFactory(test_fixtures.foo, x=7),
        z=fdl.ArgFactory(test_fixtures.foo, x=3.2),
    )
    task = init_task.init_task(config=config)
    make_symbolic_references.import_symbols(task)
    make_symbolic_references.replace_callables_and_configs_with_symbols(task)
    code = ir_to_cst.code_for_task(task).code
    self.assertIn('7', code)
    self.assertIn('3.2', code)


if __name__ == '__main__':
  absltest.main()
