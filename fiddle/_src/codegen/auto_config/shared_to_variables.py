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

"""Moves shared nodes to variables."""

import copy
from typing import Any, Callable, List

from fiddle import daglish
from fiddle._src.codegen import namespace as namespace_lib
from fiddle._src.codegen.auto_config import code_ir
from fiddle._src.codegen.auto_config import naming
from fiddle._src.codegen.auto_config import parents_first_traversal


def _strip_paths(paths: List[daglish.Path]) -> List[daglish.Path]:
  """Strips prefixes from paths."""
  seen = set()
  cleaned_paths = []
  for path in paths:
    if path not in seen:
      seen.add(path)
      cleaned_paths.append(path)
  return cleaned_paths


def move_shared_nodes_to_variables(
    task: code_ir.CodegenTask,
    *,
    make_namer: Callable[
        [namespace_lib.Namespace], naming.Namer
    ] = naming.TypeFirstNamer,
) -> None:
  """Moves any shared nodes in functions' output values to variables.

  Args:
    task: Codegen task.
    make_namer: Function that will create a Namer, used for assigning new names
      to extracted variables. Note: Each path is relative to a FixtureFunction,
      not the overall config.
  """

  task_existing_names = naming.get_task_existing_names(task)

  def _process_fn(fn: code_ir.FixtureFunction) -> None:
    # Object IDs of values to extract into variables.
    to_extract_ids = set()

    def find_to_extract(value, parent_results: Any) -> Any:
      """Processes a node in a function definition."""
      if isinstance(
          value,
          (code_ir.BaseNameReference, code_ir.Name),
      ):
        # Don't double-extract variables, and repeated symbol references are
        # fine.
        return
      elif len(parent_results) > 1:
        to_extract_ids.add(id(value))

    parents_first_traversal.traverse_parents_first(
        find_to_extract, fn.output_value
    )

    # Create a namer for new variables. But don't try to fix pre-existing bugs
    # if there are already conflicting names.
    names = copy.copy(task_existing_names)
    names.update(naming.get_fn_existing_names(fn))
    namer = make_namer(namespace_lib.Namespace(names))

    new_variables = []

    def traverse(value, state: daglish.State):
      """Actually moves shared values to variables."""
      original_value_id = id(value)
      value = state.map_children(value)

      all_paths = state.get_all_paths()
      last_path_elts = {path[-1] for path in all_paths if path}

      # There are two main technical conditions when we need to pull out a
      # shared object into a variable.
      #
      # 1. There are multiple actual parent nodes (this is the common case).
      # 2. There are two references from the same parent node, which can be
      #    determined by there being multiple different last path elements.
      #
      # The latter check is an over-generalization, but should not catch any
      # undesired cases.
      if original_value_id in to_extract_ids or len(last_path_elts) > 1:
        name = namer.name_for(value, _strip_paths(state.get_all_paths()))
        name = code_ir.Name(name, is_generated=True)
        new_variables.append(code_ir.VariableDeclaration(name, value))
        return code_ir.VariableReference(name)
      else:
        return value

    new_value = daglish.MemoizedTraversal.run(traverse, fn.output_value)
    fn.output_value = new_value
    fn.variables.extend(new_variables)

  for fn in task.top_level_call.all_fixture_functions():
    _process_fn(fn)
