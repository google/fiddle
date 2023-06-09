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

"""Experimental high-level API for auto_config codegen.

Do NOT depend on these interfaces for non-experimental code.
"""

from typing import Any, Callable, Dict, Optional

from fiddle._src.codegen import namespace as namespace_lib
from fiddle._src.codegen.auto_config import complex_to_variables
from fiddle._src.codegen.auto_config import get_history_comments
from fiddle._src.codegen.auto_config import init_task
from fiddle._src.codegen.auto_config import ir_printer
from fiddle._src.codegen.auto_config import ir_to_cst
from fiddle._src.codegen.auto_config import make_symbolic_references
from fiddle._src.codegen.auto_config import naming
from fiddle._src.codegen.auto_config import shared_to_variables
from fiddle._src.codegen.auto_config import split_arg_factories
from fiddle._src.codegen.auto_config import sub_fixture


def auto_config_codegen(
    config,
    *,
    top_level_fixture_name: str = "config_fixture",
    sub_fixtures: Optional[Dict[str, Any]] = None,
    max_expression_complexity: Optional[int] = None,
    variable_namer: Callable[
        [namespace_lib.Namespace], naming.Namer
    ] = naming.PathFirstNamer,
    include_history: bool = False,
    debug_print: bool = False,
) -> str:
  """Generates code for an auto_config fixture.

  Args:
    config: Configuration to generate auto_config code for.
    top_level_fixture_name: Name of the top-level fixture. When its as_buildable
      path is called, e.g. `config_fixture.as_buildable()`, this should return a
      config equivalent to `config`.
    sub_fixtures: Dictionary from function name to sub-configuration object,
      declaring sub-functions to create. This helps manage complexity of huge
      configurations.
    max_expression_complexity: Breaks complex expressions into variables for
      readability.
    variable_namer: Determines how variables are named when created for shared
      nodes or complex expressions.
    include_history: Whether history should be included. These currently appear
      as trailing comments in the field of Buildable's.
    debug_print: Whether to use the IR printer to print intermediate
      representations as various passes run to generate code.

  Returns:
    Python module code.
  """
  task = init_task.init_task(
      config, top_level_fixture_name=top_level_fixture_name
  )
  if debug_print:
    print("\n\nAfter initializing:", ir_printer.format_task(task), sep="\n")

  # import_symbols should be the first pass in general
  make_symbolic_references.import_symbols(task)

  if sub_fixtures is not None:
    sub_fixture.transform_sub_fixtures(task, sub_fixtures)
    if debug_print:
      print(
          "\n\nAfter transforming subfixtures:",
          ir_printer.format_task(task),
          sep="\n",
      )

  split_arg_factories.lower_arg_factories(task)
  if debug_print:
    print(
        "\n\nAfter lowering arg factories:",
        ir_printer.format_task(task),
        sep="\n",
    )

  shared_to_variables.move_shared_nodes_to_variables(
      task, make_namer=variable_namer
  )
  if debug_print:
    print(
        "\n\nAfter moving shared nodes to variables:",
        ir_printer.format_task(task),
        sep="\n",
    )

  if max_expression_complexity is not None:
    complex_to_variables.move_complex_nodes_to_variables(
        task,
        is_complex=complex_to_variables.more_complex_than(
            max_expression_complexity
        ),
        make_namer=variable_namer,
    )
    if debug_print:
      print(
          "\n\nAfter moving complex sub-expressions to variables:",
          ir_printer.format_task(task),
          sep="\n",
      )

  if include_history:
    format_history = get_history_comments.format_history_for_buildable
  else:
    format_history = make_symbolic_references.noop_history_comments
  make_symbolic_references.replace_callables_and_configs_with_symbols(
      task, format_history=format_history
  )
  if debug_print:
    print(
        "\n\nAfter replacing callables with symbols:",
        ir_printer.format_task(task),
        sep="\n",
    )
  return ir_to_cst.code_for_task(task).code
