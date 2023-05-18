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

"""Transform sub-fixtures into separate functions."""

import copy
from typing import Dict, Set

from fiddle import daglish
from fiddle._src import config as config_lib
from fiddle._src.codegen.auto_config import code_ir


# TODO(b/283273520): Deduplicate this logics in different passes.
def _get_existing_names(
    task: code_ir.CodegenTask,
) -> Set[str]:
  """Extract existing names from CodegenTask."""
  all_fn_names = {
      fn.name.value for fn in task.top_level_call.all_fixture_functions()
  }
  fn = task.top_level_call.fn
  names = copy.copy(task.global_namespace.names)
  names.update(all_fn_names)
  names.update(parameter.name.value for parameter in fn.parameters)
  names.update(variable.name.value for variable in fn.variables)
  return names


def _transform_sub_fixtures(
    task: code_ir.CodegenTask, sub_fixtures: Dict[str, config_lib.Buildable]
) -> None:
  """Transform sub-fixtures into separate functions."""
  sub_fixture_names = {id(value): name for name, value in sub_fixtures.items()}
  new_fixtures = []
  fixture_call_mapping = {}

  # Replace sub-fixture with code_ir.call
  def traverse(value, state: daglish.State):
    """Actually moves shared values to variables."""
    original_id = id(value)
    value = state.map_children(value)
    if original_id in sub_fixture_names:
      name = sub_fixture_names[original_id]
      name = code_ir.Name(name, is_generated=True)
      fixture_fn = code_ir.FixtureFunction(
          name=name, parameters=[], variables=[], output_value=value
      )
      new_fixtures.append(fixture_fn)
      call_node = code_ir.Call(name=name, arg_expressions={})
      fixture_call_mapping[id(fixture_fn)] = call_node
      return call_node
    else:
      return value

  fn = task.top_level_call.fn
  new_fn = daglish.MemoizedTraversal.run(traverse, fn)
  fn.replace_with(new_fn)

  # Add new fixtures to the task.top_level_call
  for fixture in new_fixtures:
    call_node = fixture_call_mapping[id(fixture)]
    task.top_level_call.children[call_node] = code_ir.CallInstance(
        fixture, parent=None, children={}, parameter_values={}
    )


def transform_sub_fixtures(
    task: code_ir.CodegenTask, sub_fixtures: Dict[str, config_lib.Buildable]
) -> None:
  """Moves any shared nodes in functions' output values to variables.

  Note that this API is still under development. Be aware that there are edge
  cases that this API may fail to handle. For example, when sub-fixtures have
  shared nodes.

  Args:
    task: Codegen task.
    sub_fixtures: A dict that maps the sub-fixture name to the actual
      sub-fixtures config object.
  """
  # Validate sub_fixtures names
  existing_names = _get_existing_names(task)
  for name in sub_fixtures:
    if name in existing_names:
      msg = (
          f"'{name}' already exists in the top level fixture. Please use "
          "another name for the sub-fixture"
      )
      raise ValueError(msg)

  for fixture in sub_fixtures.values():
    if isinstance(fixture, config_lib.ArgFactory):
      raise ValueError(
          "fdl.ArgFactory is not supported in auto-config codegen sub-fixture"
          " transformation."
      )

  _transform_sub_fixtures(task, sub_fixtures)
