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

"""Debugging utility to print the intermediate representation of code.

This often will not print out real Python code, and may contain many errors. It
is only to be used for debugging purposes.
"""

import inspect
import typing
from typing import Any, List

import fiddle as fdl
from fiddle import daglish
from fiddle._src.codegen.auto_config import code_ir
import libcst as cst

# Values from the `typing` module. This probably pulls in too much stuff, but
# we don't have to be 100% precise in this module, it's just for debug printing.
typing_consts = [getattr(typing, name) for name in dir(typing)]


def format_py_reference(value: Any) -> str:
  module_name = inspect.getmodule(value).__name__
  if module_name in ("fiddle._src.config", "fiddle._src.partial"):
    module_name = "fdl"
  cls_name = value.__qualname__
  return f"{module_name}.{cls_name}"


def format_expr(expr: Any):
  """Formats an expression, which may contain other IR nodes.

  Args:
    expr: Expression of a flexible type. This may be a basic Python value, a
      Fiddle configuration, list/tuple/dict, or IR nodes like a call to another
      function, reference to a variable, etc.

  Returns:
    String representation of code.
  """

  def traverse(value, state: daglish.State) -> str:
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
      return repr(value)
    elif isinstance(value, fdl.Buildable):
      arguments = fdl.ordered_arguments(state.map_children(value))
      arguments = ", ".join(
          f"{name}={sub_value}" for name, sub_value in arguments.items()
      )
      buildable_type = format_py_reference(type(value))
      fn = format_py_reference(fdl.get_callable(value))
      return f"{buildable_type}({fn}, {arguments})"
    elif isinstance(value, (list, tuple)):
      children = ", ".join(
          str(sub_value) for sub_value in state.map_children(value)
      )
      if isinstance(value, tuple):
        if len(value) == 1:
          children += ","
        return f"({children})"
      else:
        return f"[{children}]"
    elif isinstance(value, dict):
      value = state.map_children(value)
      return (
          "{"
          + ", ".join(f'"{key}": {value}' for key, value in value.items())
          + "}"
      )
    elif isinstance(value, code_ir.BaseNameReference):
      return value.name.value
    elif isinstance(value, code_ir.AttributeExpression):
      base_obj = state.call(value.base, daglish.Attr("base"))
      return f"{base_obj}.{value.attribute}"
    elif isinstance(value, code_ir.ArgFactoryExpr):
      sub_value = state.map_children(value).expression
      return f"ArgFactoryExpr[{sub_value}]"
    elif isinstance(value, code_ir.WithTagsCall):
      tags_str = ", ".join(value.tag_symbol_expressions)
      sub_value = state.map_children(value.item_to_tag)
      return f"WithTagsCall[{tags_str}]({sub_value})"
    elif isinstance(value, code_ir.SymbolOrFixtureCall):
      symbol_expression = state.call(
          value.symbol_expression, daglish.Attr("symbol_expression")
      )
      positional_arg_expressions = state.call(
          value.positional_arg_expressions,
          daglish.Attr("positional_arg_expressions"),
      )
      arg_expressions = state.call(
          value.arg_expressions, daglish.Attr("arg_expressions")
      )
      return (
          f"call:<{symbol_expression}"
          f"(*[{positional_arg_expressions}],"
          f" **{arg_expressions})>"
      )
    elif isinstance(value, code_ir.ParameterizedTypeExpression):
      base_expression = state.call(
          value.base_expression, daglish.Attr("base_expression")
      )
      param_expressions = state.call(
          value.param_expressions, daglish.Attr("param_expressions")
      )
      return f"{base_expression}{param_expressions}"
    elif isinstance(value, code_ir.Name):
      return value.value
    elif isinstance(value, type) and value is not typing.Any:
      return value.__name__
    elif value in typing_consts or type(value) in typing_consts:
      return str(value)
    else:
      return f"<<<custom:{repr(value)}>>>"

  return daglish.MemoizedTraversal.run(traverse, expr)


def format_fn(fn: code_ir.FixtureFunction) -> List[str]:
  """Formats code for a fixture function.

  Args:
    fn: IR node for the function definition.

  Returns:
    List of code lines.
  """
  name = fn.name.value
  parameters_str = ", ".join(
      f"{param.name.value}: {format_expr(param.value_type)}"
      for param in fn.parameters
  )
  result = [f"def {name}({parameters_str}):"]
  for variable in fn.variables:
    result.append(
        f"  {variable.name.value} = {format_expr(variable.expression)}"
    )
  result.append(f"  return {format_expr(fn.output_value)}")
  return result


def format_task(task: code_ir.CodegenTask) -> str:
  """Returns a string representation of a codegen task."""
  import_lines = cst.Module(body=task.import_manager.sorted_import_lines()).code
  if import_lines:
    import_lines += "\n\n"
  return import_lines + "\n\n".join(
      "\n".join(format_fn(fn))
      for fn in task.top_level_call.all_fixture_functions()
  )
