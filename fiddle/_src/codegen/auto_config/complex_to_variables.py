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

"""Extracts complex nodes/expressions into variables.

Note: This stage should be run AFTER shared_to_variables, since
shared_to_variables does NOT recognize shared sub-expressions in variables.

Therefore, this stage DOES have the ability to factor out complex expressions
ValueErrorfrom both variables and return statements.
"""

import copy
from typing import Any, Callable

from fiddle import daglish
from fiddle._src.codegen import namespace as namespace_lib
from fiddle._src.codegen.auto_config import code_ir
from fiddle._src.codegen.auto_config import naming


def more_complex_than(level: int) -> Callable[[Any], bool]:
  """Returns a comparison function that will return True if a node is complex.

  In this case, "complex" means that its number of sub-nodes is greater than
  `level`.

  Args:
    level: Level of complexity.
  """

  def traverse(value, state: daglish.State) -> int:
    if isinstance(value, code_ir.Name):
      # Names don't count towards node complexity. Usually they are sub-fields
      # of another node (like code_ir.VariableReference), where the container
      # has already incremented the complexity.
      return 0
    elif not state.is_traversable(value):
      return 1
    else:
      return 1 + sum(state.yield_map_child_values(value))

  return lambda x: daglish.MemoizedTraversal.run(traverse, x) > level


def move_complex_nodes_to_variables(
    task: code_ir.CodegenTask,
    *,
    is_complex: Callable[[Any], bool],
    make_namer: Callable[
        [namespace_lib.Namespace], naming.Namer
    ] = naming.TypeFirstNamer,
) -> None:
  """Moves complex nodes (expressions) to variables.

  While having reasonable-sized nested trees of configuration can be nice,
  showing a clear, immutable call structure, the nesting can become too much
  and hide details for very large configs.

  Therefore, we enable extraction of expressions into variables that have a
  complexity greater than a certain amount.

  Args:
    task: Codegen task.
    is_complex: Function used to determine when a node should be moved to a
      variable.
    make_namer: Function that will create a Namer, used for assigning new names
      to extracted variables. Note: Each path is relative to a FixtureFunction,
      not the overall config.
  """

  task_existing_names = naming.get_task_existing_names(task)

  def _process_fn(fn: code_ir.FixtureFunction) -> None:
    names = copy.copy(task_existing_names)
    names.update(naming.get_fn_existing_names(fn))
    namer = make_namer(namespace_lib.Namespace(names))

    new_variables = []

    def traverse(value, state: daglish.State):
      """Extracts complex values to variables."""
      if not state.is_traversable(value):
        return value

      if isinstance(value, code_ir.SymbolOrFixtureCall):
        value.arg_expressions = state.map_children(value.arg_expressions)
        return value

      if isinstance(value, code_ir.VariableDeclaration):
        value.expression = state.map_children(value.expression)
        return value

      value = state.map_children(value)

      if isinstance(
          value, (code_ir.VariableDeclaration, code_ir.ArgFactoryExpr)
      ):
        # If something is already a variable, don't create another variable.
        # Likewise, if it is an ArgFactoryExpr, keep it inline with its parent
        # fdl.Partial, since it's really more of an annotation on the argument.
        return value
      elif is_complex(value) and state.current_path:
        # Extract a new variable!
        try:
          name = namer.name_for(value, state.get_all_paths())
        except naming.NameGenerationError:
          # Don't fail if we couldn't extract a name, this pass is not essential
          # (unlike shared_to_variables, which should fail, since failure to
          # extract a variable there will mean a shared value is not actually
          # shared, which has incorrect semantics).
          return value
        else:
          name = code_ir.Name(name, is_generated=True)
          new_variables.append(code_ir.VariableDeclaration(name, value))
          return code_ir.VariableReference(name)
      else:
        return value

    for variable in fn.variables:
      rewritten_variable = daglish.MemoizedTraversal.run(traverse, variable)
      assert isinstance(
          rewritten_variable, code_ir.VariableDeclaration
      ), "Internal error! Type of VariableDeclaration changed!"
      new_variables.append(rewritten_variable)

    rewritten_output_value = daglish.MemoizedTraversal.run(
        traverse, fn.output_value
    )
    fn.variables = new_variables
    fn.output_value = rewritten_output_value

  for fn in task.top_level_call.all_fixture_functions():
    _process_fn(fn)
