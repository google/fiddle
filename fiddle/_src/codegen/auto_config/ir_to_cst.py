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

"""Converts from the auto_config codegen representation to LibCST nodes.

Note: We do not generally do a lot of formatting, instead relying on existing
source code formatters (pyformat, yapf, etc.).
"""

from typing import Any

from fiddle import daglish
from fiddle._src import config as config_lib
from fiddle._src.codegen import py_val_to_cst_converter
from fiddle._src.codegen.auto_config import code_ir
import libcst as cst


def code_for_expr(expr: Any) -> cst.CSTNode:
  """Generates CST nodes for an expression.

  Args:
    expr: Arbitrary Python value expression to generate code for.

  Returns:
    CST node for expression.
  """

  def traverse(value, state: daglish.State) -> cst.CSTNode:
    if isinstance(value, config_lib.Buildable):
      raise ValueError(
          "Internal Fiddle error: you must run the make_symbolic_reference "
          "passes before CST generation."
      )
    elif isinstance(value, (list, tuple)):
      value = state.map_children(value)
      if isinstance(value, list):
        cst_cls = cst.List
      else:
        cst_cls = cst.Tuple
      return cst_cls([cst.Element(elt) for elt in value])
    elif isinstance(value, dict):
      elements = []
      for key, sub_value in value.items():
        key_node = state.call(key, daglish.Key(f"__key_{key}"))
        sub_value_node = state.call(sub_value, daglish.Attr(key))
        elements.append(cst.DictElement(key_node, sub_value_node))
      return cst.Dict(elements)
    elif isinstance(value, code_ir.VariableReference):
      return cst.Name(value.name.value)
    elif isinstance(value, code_ir.SymbolCall):
      attr = daglish.Attr("arg_expressions")
      args = []
      for i, arg_value in enumerate(value.positional_arg_expressions):
        arg_value = state.call(arg_value, attr, daglish.Key(i))
        args.append(cst.Arg(arg_value))
      for arg_name, arg_value in value.arg_expressions.items():
        arg_value = state.call(arg_value, attr, daglish.Key(arg_name))
        args.append(
            cst.Arg(
                arg_value,
                keyword=cst.Name(arg_name),
                equal=cst.AssignEqual(
                    whitespace_before=cst.SimpleWhitespace(""),
                    whitespace_after=cst.SimpleWhitespace(""),
                ),
            )
        )
      return cst.Call(
          cst.parse_expression(value.symbol_expression),
          args=args,
      )
    elif isinstance(value, code_ir.SymbolReference):
      return cst.parse_expression(value.expression)
    elif state.is_traversable(value):
      raise NotImplementedError(
          f"Expression generation is not implemented for {value!r}"
      )
    else:
      # Convert primitives with existing logic.
      try:
        return py_val_to_cst_converter.convert_py_val_to_cst(value)
      except:
        print(f"\n\nERROR CONVERTING: {value!r}")
        print(f"\n\nTYPE: {type(value)}")
        raise

  return daglish.MemoizedTraversal.run(traverse, expr)


def code_for_fn(
    fn: code_ir.FixtureFunction, *, task: code_ir.CodegenTask
) -> cst.FunctionDef:
  """Generates LibCST for a fixture function.

  Args:
    fn: Fixture function to generate code for.
    task: Codegen task.

  Returns:
    LibCST FunctionDef node.
  """
  auto_config_expr = cst.parse_expression(
      task.import_manager.add(task.auto_config_fn)
  )
  params = cst.Parameters(
      params=[
          cst.Param(name=cst.Name(param.name.value)) for param in fn.parameters
      ]
  )
  variable_lines = []
  for variable_decl in fn.variables:
    name = variable_decl.name.value
    assign = cst.Assign(
        targets=[cst.AssignTarget(target=cst.Name(name))],
        value=code_for_expr(variable_decl.expression),
    )
    variable_lines.append(cst.SimpleStatementLine(body=[assign]))
  body = cst.IndentedBlock(
      body=[
          *variable_lines,
          cst.SimpleStatementLine(
              body=[cst.Return(code_for_expr(fn.output_value))]
          ),
      ]
  )
  if fn.parameters:
    whitespace_before_params = cst.ParenthesizedWhitespace(
        cst.TrailingWhitespace(),
        indent=True,
        last_line=cst.SimpleWhitespace("  "),
    )
  else:
    whitespace_before_params = cst.SimpleWhitespace("")
  return cst.FunctionDef(
      cst.Name(fn.name.value),
      params,
      body,
      decorators=[cst.Decorator(auto_config_expr)],
      whitespace_before_params=whitespace_before_params,
      leading_lines=[cst.EmptyLine(), cst.EmptyLine()],
  )


def code_for_task(
    task: code_ir.CodegenTask,
) -> cst.Module:
  """Generates LibCST for a codegen task.

  Args:
    task: Codegen task.

  Returns:
    LibCST module.
  """
  body = []
  for fn in reversed(task.top_level_call.all_fixture_functions()):
    body.append(code_for_fn(fn, task=task))
  return cst.Module(body=task.import_manager.sorted_import_lines() + body)
