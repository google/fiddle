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

from typing import Any, List, Optional, Union

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

    def _prepare_args_helper(
        names: List[Union[str, int]],
        values: List[Any],
        attr: daglish.Attr,
        *,
        history: Optional[code_ir.HistoryComments] = None,
    ) -> List[cst.Arg]:
      """Prepare code_ir.Arg node based on names and values."""
      cst_args = {}
      cst_kwargs = []
      for arg_name, arg_value in zip(names, values):
        if not isinstance(arg_name, (int, str)):
          raise TypeError(
              f"Unsupported arg name type: {type(arg_name)} (value: {arg_name})"
          )
        arg_value = state.call(arg_value, attr, daglish.Key(arg_name))
        kwargs = {}
        if history and arg_name in history.per_field:
          kwargs["comma"] = cst.Comma(
              whitespace_after=cst.ParenthesizedWhitespace(
                  first_line=cst.TrailingWhitespace(
                      whitespace=cst.SimpleWhitespace("  "),
                      comment=cst.Comment(f"# {history.per_field[arg_name]}"),
                      newline=cst.Newline(),
                  ),
                  last_line=cst.SimpleWhitespace(" " * 6),
              )
          )
        if isinstance(arg_name, str):
          cst_kwargs.append(
              cst.Arg(
                  arg_value,
                  keyword=cst.Name(arg_name),
                  equal=cst.AssignEqual(
                      whitespace_before=cst.SimpleWhitespace(""),
                      whitespace_after=cst.SimpleWhitespace(""),
                  ),
                  **kwargs,
              )
          )
        elif isinstance(arg_name, int):
          if arg_name in cst_args:
            raise ValueError(f"Duplicate positional arg: {arg_name}")
          cst_args[arg_name] = cst.Arg(arg_value, **kwargs)

      positional_arg_keys = set(cst_args.keys())
      if positional_arg_keys != set(range(len(positional_arg_keys))):
        raise ValueError(
            "Positional args supplied were not contiguous! "
            f"Got {positional_arg_keys}"
        )
      return [value for _, value in sorted(cst_args.items())] + cst_kwargs

    if isinstance(value, config_lib.Buildable):
      raise ValueError(
          "Internal Fiddle error: you must run the make_symbolic_references "
          "passes before CST generation."
      )
    elif isinstance(value, (list, tuple)):
      original = value
      value = state.map_children(value)
      if isinstance(value, list):
        cst_cls = cst.List
      else:
        cst_cls = cst.Tuple
      elements = []
      for sub_value in value:
        if not isinstance(sub_value, cst.CSTNode):
          raise TypeError(
              f"Failed to map children of {original!r}, this might be because "
              f"{type(original)} subclasses list or tuple. If this is the case,"
              " please replace these objects in your input config, likely "
              "with fdl.Config nodes."
          )
        elements.append(cst.Element(sub_value))
      return cst_cls(elements)
    elif isinstance(value, dict):
      elements = []
      for key, sub_value in value.items():
        key_node = state.call(key, daglish.Key(f"__key_{key}"))
        sub_value_node = state.call(sub_value, daglish.Attr(key))
        elements.append(cst.DictElement(key_node, sub_value_node))
      return cst.Dict(elements)
    elif isinstance(value, code_ir.BaseNameReference):
      return cst.Name(value.name.value)
    elif isinstance(value, code_ir.AttributeExpression):
      base = state.call(value.base, daglish.Attr("base"))
      return cst.Attribute(value=base, attr=cst.Name(value.attribute))
    elif isinstance(value, code_ir.ParameterizedTypeExpression):
      return cst.Subscript(
          value=code_for_expr(value.base_expression),
          slice=[
              cst.SubscriptElement(cst.Index(code_for_expr(param)))
              for param in value.param_expressions
          ],
      )
    elif isinstance(value, code_ir.SymbolOrFixtureCall):
      attr = daglish.Attr("arg_expressions")
      args = []
      for i, arg_value in enumerate(value.positional_arg_expressions):
        arg_value = state.call(arg_value, attr, daglish.Key(i))
        args.append(cst.Arg(arg_value))

      any_args_have_history = False
      if value.arg_expressions:
        names, values = zip(*value.arg_expressions.items())
        any_args_have_history = set(names) & set(
            value.history_comments.per_field
        )
        args.extend(
            _prepare_args_helper(
                names, values, attr, history=value.history_comments
            )
        )
      if any_args_have_history:
        whitespace_before_args = cst.ParenthesizedWhitespace(
            first_line=cst.TrailingWhitespace(newline=cst.Newline()),
            last_line=cst.SimpleWhitespace(" " * 6),
        )
      else:
        whitespace_before_args = cst.SimpleWhitespace("")
      return cst.Call(
          state.call(
              value.symbol_expression, daglish.Attr("symbol_expression")
          ),
          args=args,
          whitespace_before_args=whitespace_before_args,
      )
    elif isinstance(value, code_ir.WithTagsCall):
      attr = daglish.Attr("item_to_tag")
      item_to_tag = state.call(value.item_to_tag, attr)
      call_args = [cst.Arg(item_to_tag)]
      sorted_tags = sorted([tag for tag in value.tag_symbol_expressions])
      for tag in sorted_tags:
        tag_name = cst.parse_expression(tag)
        call_args.append(cst.Arg(tag_name))
      with_tags = cst.parse_expression("auto_config.with_tags")
      return cst.Call(with_tags, args=call_args)
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
        print(f"\n\nPATH: {daglish.path_str(state.current_path)}")
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
  if task.auto_config_fn:
    auto_config_expr = cst.parse_expression(
        task.import_manager.add(task.auto_config_fn)
    )
  else:
    auto_config_expr = None
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
  if fn.return_type_annotation:
    returns = cst.Annotation(
        annotation=code_for_expr(fn.return_type_annotation)
    )
  else:
    returns = None
  if fn.parameters and len(fn.parameters) > 1:
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
      returns=returns,
      decorators=[cst.Decorator(auto_config_expr)] if auto_config_expr else [],
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
  for fn in task.top_level_call.all_fixture_functions():
    body.append(code_for_fn(fn, task=task))
  return cst.Module(body=task.import_manager.sorted_import_lines() + body)
