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

"""Tests for newcg_symbolic_references pass."""

import inspect

from absl.testing import absltest
import fiddle as fdl
from fiddle._src.codegen import newcg_symbolic_references
from fiddle._src.codegen.auto_config import code_ir
from fiddle._src.codegen.auto_config import complex_to_variables
from fiddle._src.codegen.auto_config import get_history_comments
from fiddle._src.codegen.auto_config import init_task
from fiddle._src.codegen.auto_config import ir_to_cst
from fiddle._src.codegen.auto_config import test_fixtures


class MakeSymbolicReferencesTest(absltest.TestCase):

  def test_import_symbols(self):
    task = test_fixtures.simple_ir()

    # By default, this contains a bunch of builtins.
    task.import_manager.namespace.names = set()

    newcg_symbolic_references.import_symbols(task)

    # The imported symbols are auto_config, and the fixture module, since it
    # contains the `foo` function.
    self.assertEqual(
        task.import_manager.namespace.names, {'fdl', 'test_fixtures'}
    )

  def test_import_symbols_empty(self):
    task = init_task.init_task(config=[])
    task.import_manager.namespace.names = set()
    newcg_symbolic_references.import_symbols(task)
    self.assertEqual(task.import_manager.namespace.names, set())

  def test_replace_config_callable(self):
    task = test_fixtures.simple_ir()
    newcg_symbolic_references.import_symbols(task)
    newcg_symbolic_references.replace_callables_and_configs_with_symbols(
        task, format_history=get_history_comments.format_history_for_buildable
    )
    (base_line,) = (
        i
        for i, line in enumerate(inspect.getsource(test_fixtures).splitlines())
        if 'def simple_ir(' in line
    )
    line = base_line + 9  # Feel free to adjust if the test fixture is changed.
    self.assertEqual(
        task.top_level_call.fn,
        code_ir.FixtureFunction(
            name=code_ir.Name(
                value='simple_ir_fixture', is_generated=False, previous=None
            ),
            parameters=[],
            variables=[],
            output_value=code_ir.SymbolOrFixtureCall(
                symbol_expression=code_ir.AttributeExpression(
                    code_ir.ModuleReference(code_ir.Name('fdl')),
                    'Config',
                ),
                positional_arg_expressions=[
                    code_ir.AttributeExpression(
                        code_ir.ModuleReference(code_ir.Name('test_fixtures')),
                        'foo',
                    )
                ],
                arg_expressions={'x': 4},
                history_comments=code_ir.HistoryComments(
                    per_field={
                        '__fn_or_cls__': (
                            f'Set in .../auto_config/test_fixtures.py:{line}'
                            ':simple_ir'
                        ),
                        'x': (
                            f'Set in .../auto_config/test_fixtures.py:{line}'
                            ':simple_ir'
                        ),
                    }
                ),
            ),
        ),
    )

  def test_replaces_arg_factory_partial(self):
    # Tests arg factories.

    config = test_fixtures.auto_config_arg_factory_fn.as_buildable()
    task = init_task.init_task(config=config)
    task.auto_config_fn = None
    newcg_symbolic_references.import_symbols(task)
    newcg_symbolic_references.replace_callables_and_configs_with_symbols(task)
    code = ir_to_cst.code_for_task(task).code

    expected = """
    import fiddle as fdl
    from fiddle._src.codegen.auto_config import test_fixtures

    def config_fixture():
      return fdl.Partial(test_fixtures.SharedType, x=fdl.ArgFactory(test_fixtures.count, increment=3), z=4.7)
   """
    self.assertEqual(code.split(), expected.split(), msg=code)

  def test_two_arg_factories(self):
    config = fdl.Partial(
        test_fixtures.SharedType,
        x=fdl.ArgFactory(test_fixtures.foo, x=7),
        z=fdl.ArgFactory(test_fixtures.foo, x=3.2),
    )
    task = init_task.init_task(config=config)
    newcg_symbolic_references.import_symbols(task)
    newcg_symbolic_references.replace_callables_and_configs_with_symbols(task)
    code = ir_to_cst.code_for_task(task).code
    self.assertIn('7', code)
    self.assertIn('3.2', code)

  def test_nested_arg_factories(self):
    config = fdl.Partial(
        test_fixtures.Attention,
        kernel_init=fdl.ArgFactory(
            test_fixtures.initializer, name='const', dtype='float32'
        ),
    )
    task = init_task.init_task(config=config)
    task.auto_config_fn = None
    newcg_symbolic_references.import_symbols(task)
    complex_to_variables.move_complex_nodes_to_variables(
        task,
        is_complex=complex_to_variables.more_complex_than(2),
    )
    newcg_symbolic_references.replace_callables_and_configs_with_symbols(task)
    code = ir_to_cst.code_for_task(task).code

    # Note: Wrapped some lines on spaces, since we don't compare whitespace.
    expected = """
    import fiddle as fdl
    from fiddle._src.codegen.auto_config import test_fixtures

    def config_fixture():
        initializer = fdl.ArgFactory(test_fixtures.initializer,
            name='const', dtype='float32')
        return fdl.Partial(test_fixtures.Attention, kernel_init=initializer)
    """
    self.assertEqual(code.split(), expected.split(), msg=code)

  def test_bare_symbols(self):
    task = test_fixtures.bare_symbol_reference_config()
    task.auto_config_fn = None
    newcg_symbolic_references.import_symbols(task)
    newcg_symbolic_references.replace_callables_and_configs_with_symbols(task)
    code = ir_to_cst.code_for_task(task).code
    expected = """
    from fiddle._src.codegen.auto_config import test_fixtures


    def config_fixture():
        return test_fixtures.Attention
    """
    self.assertEqual(code.split(), expected.split(), msg=code)


if __name__ == '__main__':
  absltest.main()
