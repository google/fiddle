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
from typing import Any, Callable, Dict, List, Tuple

from fiddle import daglish
from fiddle._src import config as config_lib
from fiddle._src.codegen import namespace as namespace_lib
from fiddle._src.codegen.auto_config import code_ir
from fiddle._src.codegen.auto_config import naming


def _find_shared_nodes(
    top_level_fn: code_ir.FixtureFunction,
    sub_fixtures: Dict[str, config_lib.Buildable],
) -> Tuple[Dict[int, Any], Dict[int, Any]]:
  """Find out shared nodes by traversing each sub-fixture.

  The definition of shared node is any node that needs to be defined in the
  top-level fixture and then be passed as arguments to sub-fixtures.

  Args:
    top_level_fn: Top-level fixture function.
    sub_fixtures: A dict that maps the sub-fixture name to the actual
      sub-fixture object.

  Returns:
    A dict that maps id(shared_node) -> shared_node object.
    A dict that maps id(fixture) -> set of (id(shared_node), its path).
  """
  is_visited = {}
  shared_nodes = {}
  sub_fixture_ids = {id(value) for value in sub_fixtures.values()}
  top_fixture_node_ids = set()

  def traverse(value: Any, state: daglish.State) -> Any:
    """Mark all nodes within the top-level fixture."""
    value_id = id(value)
    # Do not traverse over sub-fixtures.
    if value_id in sub_fixture_ids:
      return value
    top_fixture_node_ids.add(value_id)
    return state.map_children(value)

  # 1st pass: mark all nodes that are used by the top-level fixture. If this
  # node is also used within sub-fixtures, it's a shared node.
  daglish.BasicTraversal.run(traverse, top_level_fn)

  # 2nd pass: find out shared nodes within sub-fixtures
  for idx, fixture in enumerate(sub_fixtures.values()):
    for value, path in daglish.iterate(fixture, memoized=False):
      if not daglish.is_unshareable(value):
        used_by_sub_fixture = False
        if id(value) in is_visited:
          # Ensure the node is shared among multiple sub-fixtures.
          if len(is_visited[id(value)]) > 1 or (
              idx not in is_visited[id(value)]
          ):
            used_by_sub_fixture = True
        else:
          is_visited[id(value)] = set()
        is_visited[id(value)].add(idx)
        used_by_top_fixture = id(value) in top_fixture_node_ids
        if used_by_top_fixture or used_by_sub_fixture:
          # Do not identify user specified sub-fixtures as shared nodes
          if id(value) not in sub_fixture_ids:
            shared_nodes[id(value)] = (value, path)

  # 3rd pass: identify and record shared nodes in each fixture
  # Map id(fixture) -> set of id(shared_node) within this sub-fixture
  fixture_share_node_ids = {}
  for fixture in sub_fixtures.values():
    fixture_share_node_ids[id(fixture)] = set()
    for value, _ in daglish.iterate(fixture, memoized=False):
      if not daglish.is_unshareable(value):
        if id(value) in shared_nodes or id(value) in top_fixture_node_ids:
          fixture_share_node_ids[id(fixture)].add(id(value))

  return shared_nodes, fixture_share_node_ids


def _prepare_node_names(
    nodes: Dict[int, Any], namer: naming.Namer
) -> Dict[int, code_ir.Name]:
  """Create a dict that maps id(node) -> code_ir.Name object."""
  node_names = {}
  for node, path in nodes.values():
    name = namer.name_for(node, [path])
    name = code_ir.Name(name, is_generated=True)
    node_names[id(node)] = name
  return node_names


def _add_shared_node_as_var_declarations(
    task: code_ir.CodegenTask,
    shared_nodes: Dict[int, Any],
    shared_node_names: Dict[int, code_ir.Name],
) -> None:
  """Add VarDeclaration for shared nodes in the top-level fixture."""
  fn = task.top_level_call.fn
  for node_id in shared_nodes:
    assert node_id in shared_node_names
    node, _ = shared_nodes[node_id]
    name = shared_node_names[node_id]
    var = code_ir.VariableDeclaration(name=name, expression=node)
    fn.variables.append(var)


def _add_fixtures_to_task(
    task: code_ir.CodegenTask, fixtures: List[code_ir.FixtureFunction]
) -> None:
  """Add new fixtures as the children of task.top_level_call."""
  for fixture in fixtures:
    call_instance = code_ir.CallInstance(
        fixture, parent=None, children=[], parameter_values={}
    )
    task.top_level_call.children.append(call_instance)


def _transform_sub_fixtures(
    task: code_ir.CodegenTask,
    sub_fixtures: Dict[str, config_lib.Buildable],
    namer: naming.Namer,
) -> None:
  """Transform sub-fixtures into separate functions.

  The complexity here is when sub-fixtures shared some nodes in common, or
  the sub-fixture shares some nodes with the top-level fixture. The point of
  of "shared node" in any nodes whose definitions should reside in the top-level
  fixture and passed as arguments to sub-fixtures.

  Take the config below as an example, where A is the top-level config fixture.
  B and C are sub-fixtures to be extracted, and they share common parameter D.
  In this case, D should be defined in the top-level fixture and be passed in
  to sub-fixtures B and C as an argument, instead of being defined seprately
  to conserve the sharing relation.

          ┌───────┐
          │   A   │
          └───┬───┘
              │
  ┌───────┐   │   ┌───────┐
  │   B   ◄───┴───►   C   │
  └───┬───┘       └───┬───┘
      │               │
      │   ┌───────┐   │
      └───►   D   ◄───┘
          └───────┘

  Below is another example, where B is the only sub-fixture. But C is also used
  within the top-level fixture A, so C needs to be extracted and defined in A
  as well.

           ┌───────┐
      ┌────┤   A   ├────┐
      │    └───────┘    │
      │                 │
      │                 │
  ┌───▼───┐         ┌───▼───┐
  │   B   ├─────────►   C   │
  └───────┘         └───────┘

  Overall, the transformation consists of following steps in high level:

  1. Detect shared nodes among all sub-fixtures, the D in the example above.
  2. Prepare proper names for shared nodes. The top-level fixture and
     sub-fixtures will use the same name for a given shared node.
  3. Add shared nodes to the top-level fixture as variable declarations.
  4. Transform sub-fixture nodes into function calls to the new sub-fixture
     function.
  5. Transform shared nodes within sub-fixtures into var references to the
     sub-fixture function arguments.

  Args:
    task: Codegen task.
    sub_fixtures: A dict that maps the sub-fixture name to the actual
      sub-fixture config object.
    namer: A Namer used for assigning new names to extracted variables.
  """
  shared_nodes, fixture_shared_nodes = _find_shared_nodes(
      task.top_level_call.fn, sub_fixtures
  )
  shared_node_names = _prepare_node_names(shared_nodes, namer)
  _add_shared_node_as_var_declarations(task, shared_nodes, shared_node_names)

  # Map id(sub_fixture) -> names in str
  sub_fixture_names = {id(value): name for name, value in sub_fixtures.items()}
  new_fixtures = []

  def traverse(value: Any, state: daglish.State) -> Any:
    """Actually replace sub-fixtures and shared nodes."""

    def prepare_params(fixture_id: int) -> List[code_ir.Parameter]:
      """Prepare code_ir.Parameter objects for a given fixture."""
      params = []
      for node_id in fixture_shared_nodes[fixture_id]:
        name = shared_node_names[node_id]
        params.append(code_ir.Parameter(name=name))
      return params

    # Remember the original id as the value will be changed very soon.
    original_id = id(value)
    value = state.map_children(value)

    if original_id in shared_nodes:
      # Transform shared nodes as VariableReference to the argument.
      name = shared_node_names[original_id]
      return code_ir.VariableReference(name=name)
    elif original_id in sub_fixture_names:
      # Transform sub-fixure as VariableReference to separate fixture function.
      name = sub_fixture_names[original_id]
      name = code_ir.Name(name, is_generated=True)
      params = prepare_params(original_id)
      # TODO(b/285208948): Improve naming for sub-fixture nodes.
      fixture_fn = code_ir.FixtureFunction(
          name=name, parameters=params, variables=[], output_value=value
      )
      new_fixtures.append(fixture_fn)
      args = {}
      for param in params:
        args[param.name.value] = code_ir.VariableReference(name=param.name)
      return code_ir.SymbolOrFixtureCall(
          symbol_expression=code_ir.FixtureReference(name=name),
          positional_arg_expressions=[],
          arg_expressions=args,
      )
    else:
      return value

  value = task.top_level_call.fn.output_value
  new_value = daglish.MemoizedTraversal.run(traverse, value)
  task.top_level_call.fn.output_value = new_value
  _add_fixtures_to_task(task, new_fixtures)


def transform_sub_fixtures(
    task: code_ir.CodegenTask,
    sub_fixtures: Dict[str, config_lib.Buildable],
    make_namer: Callable[
        [namespace_lib.Namespace], naming.Namer
    ] = naming.PathFirstNamer,
) -> None:
  """Moves any shared nodes in functions' output values to variables.

  Args:
    task: Codegen task.
    sub_fixtures: A dict that maps the sub-fixture name to the actual
      sub-fixture config object.
    make_namer: Function that will create a Namer, used for assigning new names
      to extracted variables.
  """
  # Validate sub_fixtures names
  existing_names = naming.get_task_existing_names(task)
  existing_names.update(naming.get_fn_existing_names(task.top_level_call.fn))
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

  namer = make_namer(namespace_lib.Namespace(existing_names))
  _transform_sub_fixtures(task, sub_fixtures, namer)
