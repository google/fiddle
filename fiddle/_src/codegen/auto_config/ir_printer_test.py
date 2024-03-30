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

"""Tests for ir_printer."""

import textwrap
from typing import Any, Optional

from absl.testing import absltest
from fiddle._src.codegen.auto_config import code_ir
from fiddle._src.codegen.auto_config import ir_printer
from fiddle._src.codegen.auto_config import test_fixtures


class A:

  def __str__(self):
    return "A_str"

  def __repr__(self):
    return "A_repr"


class IrPrinterTest(absltest.TestCase):

  def test_format_expr_primitives(self):
    self.assertEqual(ir_printer.format_expr(1), "1")
    self.assertEqual(ir_printer.format_expr(None), "None")
    self.assertEqual(ir_printer.format_expr(0.123), "0.123")
    self.assertEqual(ir_printer.format_expr(A()), "<<<custom:A_repr>>>")
    self.assertEqual(ir_printer.format_expr(int), "int")
    self.assertEqual(ir_printer.format_expr(Any), "typing.Any")
    self.assertIn(
        ir_printer.format_expr(Optional[str]),
        # This differs based on the Python verison.
        {"typing.Union[str, NoneType]", "typing.Optional[str]"},
    )

  def test_format_containers(self):
    foo = {"a": int, "b": str, "c": Any}
    self.assertEqual(
        ir_printer.format_expr(foo), '{"a": int, "b": str, "c": typing.Any}'
    )
    example_list = [3, 2.5, str]
    self.assertEqual(ir_printer.format_expr(example_list), "[3, 2.5, str]")
    example_tuple = (3, 2.5, str)
    self.assertEqual(ir_printer.format_expr(example_tuple), "(3, 2.5, str)")
    example_tuple = ()
    self.assertEqual(ir_printer.format_expr(example_tuple), "()")
    example_tuple = (3,)
    self.assertEqual(ir_printer.format_expr(example_tuple), "(3,)")

  def test_format_attributes(self):
    # Not working Python but we should print it anyway.
    attr = code_ir.AttributeExpression([1, 2], "foo")
    self.assertEqual(ir_printer.format_expr(attr), "[1, 2].foo")

    self_var = code_ir.VariableReference(
        code_ir.Name("self", is_generated=True)
    )
    attr = code_ir.AttributeExpression(self_var, "foo")
    self.assertEqual(ir_printer.format_expr(attr), "self.foo")

  def test_format_with_tags(self):
    with_tags = code_ir.WithTagsCall(["module.Tag"], item_to_tag=123.4)
    self.assertEqual(
        ir_printer.format_expr(with_tags), "WithTagsCall[module.Tag](123.4)"
    )

  def test_format_calls(self):
    call = code_ir.SymbolOrFixtureCall(
        symbol_expression=code_ir.Name("foo"),
        positional_arg_expressions=[code_ir.Name("bar")],
        arg_expressions={"baz": code_ir.Name("qux")},
    )
    self.assertEqual(
        ir_printer.format_expr(call), 'call:<foo(*[[bar]], **{"baz": qux})>'
    )

  def test_format_module_reference(self):
    module_reference = code_ir.ModuleReference(code_ir.Name("foo"))
    self.assertEqual(ir_printer.format_expr(module_reference), "foo")

  def test_format_simple_ir(self):
    task = test_fixtures.simple_ir()
    code = "\n".join(ir_printer.format_fn(task.top_level_call.fn))
    self.assertEqual(
        code,
        textwrap.dedent(
            """
    def simple_ir_fixture():
      return fdl.Config(fiddle._src.codegen.auto_config.test_fixtures.foo, x=4)
    """
        ).strip(),
    )

  def test_format_simple_shared_variable_ir(self):
    task = test_fixtures.simple_shared_variable_ir()
    code = "\n".join(ir_printer.format_fn(task.top_level_call.fn))
    self.assertEqual(
        code,
        textwrap.dedent(
            """
    def simple_shared_variable_ir_fixture():
      shared = {"a": 7}
      return [shared, shared]
    """
        ).strip(),
    )


if __name__ == "__main__":
  absltest.main()
