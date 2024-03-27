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

"""Tests for ir_to_cst."""

import inspect

from absl.testing import absltest
from fiddle._src.codegen.auto_config import code_ir
from fiddle._src.codegen.auto_config import get_history_comments
from fiddle._src.codegen.auto_config import init_task
from fiddle._src.codegen.auto_config import ir_to_cst
from fiddle._src.codegen.auto_config import make_symbolic_references
from fiddle._src.codegen.auto_config import shared_to_variables
from fiddle._src.codegen.auto_config import test_fixtures
import jax.sharding
import libcst as cst


def _code_for_expr(node: cst.CSTNode) -> str:
  """Returns Python code for a CST node.

  This assumes two spaces of indentation. It is helpful because we CST nodes
  don't seem to implement the equality operator very well.

  Args:
    node: CST node.
  """
  module = cst.parse_module("def foo():\n  return")
  return module.code_for_node(node)


class IrToCstTest(absltest.TestCase):

  def test_code_for_expr(self):
    self_var = code_ir.VariableReference(
        code_ir.Name("self", is_generated=True)
    )
    attr = code_ir.AttributeExpression(self_var, "foo")
    self.assertEqual(_code_for_expr(ir_to_cst.code_for_expr(attr)), "self.foo")

  def test_code_for_expr_attribute_in_call_symbol_and_args(self):
    self_var = code_ir.VariableReference(
        code_ir.Name("self", is_generated=True)
    )
    attr = code_ir.AttributeExpression(self_var, "foo")
    call = code_ir.SymbolOrFixtureCall(
        attr, [], {"bar": code_ir.AttributeExpression(self_var, "bar")}
    )
    self.assertEqual(
        _code_for_expr(ir_to_cst.code_for_expr(call)), "self.foo(bar=self.bar)"
    )

  def test_code_for_positional_arg_call(self):
    self_var = code_ir.VariableReference(
        code_ir.Name("self", is_generated=True)
    )
    attr = code_ir.AttributeExpression(self_var, "foo")
    call = code_ir.SymbolOrFixtureCall(
        attr, [123.4], {"bar": code_ir.AttributeExpression(self_var, "bar")}
    )
    self.assertEqual(
        _code_for_expr(ir_to_cst.code_for_expr(call)),
        "self.foo(123.4, bar=self.bar)",
    )

  def test_code_for_expr_jax_partition_spec(self):
    """This is a very weird overridden tuple."""
    value = jax.sharding.PartitionSpec("data")
    with self.assertRaisesRegex(
        TypeError,
        r"Failed to map.*PartitionSpec.*subclasses list or tuple.*",
    ):
      ir_to_cst.code_for_expr(value)

  def test_basic_ir(self):
    task = test_fixtures.simple_ir()
    make_symbolic_references.import_symbols(task)
    make_symbolic_references.replace_callables_and_configs_with_symbols(task)
    code = ir_to_cst.code_for_task(task).code
    expected = """
    from fiddle._src.codegen.auto_config import test_fixtures
    from fiddle.experimental import auto_config


    @auto_config.auto_config
    def simple_ir_fixture():
        return test_fixtures.foo(x=4)
    """
    self.assertEqual(code.split(), expected.split(), msg=code)

  def test_two_shared_config(self):
    task = test_fixtures.unprocessed_two_shared_config()
    make_symbolic_references.import_symbols(task)
    shared_to_variables.move_shared_nodes_to_variables(task)
    make_symbolic_references.replace_callables_and_configs_with_symbols(
        task, format_history=get_history_comments.format_history_for_buildable
    )
    code = ir_to_cst.code_for_task(task).code
    (base_line,) = (
        i
        for i, line in enumerate(inspect.getsource(test_fixtures).splitlines())
        if "def unprocessed_two_shared_config(" in line
    )
    line = base_line + 11  # Feel free to adjust if the test fixture is changed.
    expected = f"""
    from fiddle._src.codegen.auto_config import test_fixtures
    from fiddle.experimental import auto_config


    @auto_config.auto_config
    def unprocessed_two_shared_fixture():
      foo = test_fixtures.foo(
        x=3,  # Set in .../auto_config/test_fixtures.py:{line}:unprocessed_two_shared_config
        )
      shared_type = test_fixtures.SharedType(
        x=foo,  # Set in .../auto_config/test_fixtures.py:{line + 1}:unprocessed_two_shared_config
        z=7.0,  # Set in .../auto_config/test_fixtures.py:{line + 1}:unprocessed_two_shared_config
        )
      return [shared_type, shared_type, foo]
    """
    self.assertEqual(code.split(), expected.split(), msg=code)

  def test_complex_dict_node_generation(self):
    # These kinds of dictionaries aren't really supported by other passes, so
    # please don't actually put them in your configs.
    #
    # Also sets are not supported by daglish by default, so they are only
    # generated for primitive values here thanks to py_val_to_cst_converter.
    config = {7.2: ("hi", {3, 4})}
    task = init_task.init_task(config)
    code = ir_to_cst.code_for_task(task).code
    expected = """
    from fiddle.experimental import auto_config


    @auto_config.auto_config
    def config_fixture():
        return {7.2: ('hi', {3, 4})}
    """
    self.assertEqual(code.split(), expected.split(), msg=code)

  def test_tags_generation_config(self):
    task = test_fixtures.simple_ir_with_tags()
    make_symbolic_references.import_symbols(task)
    make_symbolic_references.replace_callables_and_configs_with_symbols(task)
    code = ir_to_cst.code_for_task(task).code
    expected = """
    from fiddle._src.codegen.auto_config import test_fixtures
    from fiddle.experimental import auto_config


    @auto_config.auto_config
    def config_fixture():
        return test_fixtures.bar(x=auto_config.with_tags(1,
          test_fixtures.ATag),
          y=auto_config.with_tags(2, test_fixtures.ATag, test_fixtures.BTag))
    """
    self.assertEqual(code.split(), expected.split(), msg=code)

  def test_tags_generation_partial(self):
    task = test_fixtures.simple_partial_ir_with_tags()
    make_symbolic_references.import_symbols(task)
    make_symbolic_references.replace_callables_and_configs_with_symbols(task)
    code = ir_to_cst.code_for_task(task).code
    expected = """
    from fiddle._src.codegen.auto_config import test_fixtures
    from fiddle.experimental import auto_config
    import functools


    @auto_config.auto_config
    def config_fixture():
        return functools.partial(test_fixtures.bar,
            x=auto_config.with_tags(1, test_fixtures.ATag),
            y=auto_config.with_tags(2, test_fixtures.ATag, test_fixtures.BTag))
    """
    self.assertEqual(code.split(), expected.split(), msg=code)


if __name__ == "__main__":
  absltest.main()
