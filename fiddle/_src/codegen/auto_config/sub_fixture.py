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
from typing import Any, Callable, Dict, List, Set, Tuple

from fiddle import daglish
from fiddle._src import config as config_lib
from fiddle._src import partial
from fiddle._src.codegen import namespace as namespace_lib
from fiddle._src.codegen.auto_config import code_ir
from fiddle._src.codegen.auto_config import naming
from fiddle._src.codegen.auto_config import parents_first_traversal


def _get_node_to_parents_mapping(
    config: config_lib.Config,
) -> Dict[int, Set[Any]]:
  """Get a mapping from node id -> list of its parents."""
  node_to_parents_by_id = {}

  def traverse(value, parent_results):
    node_to_parents_by_id[id(value)] = {id(elt) for elt in parent_results}
    return value

  parents_first_traversal.traverse_parents_first(traverse, config)

  return node_to_parents_by_id


def _is_super_ancestor(
    ancestor: int, nodes: Set[int], node_to_parents_by_id: Dict[int, Set[Any]]
) -> bool:
  """Check if ancestor is `super ancestor` (see definition below) of all nodes.

  The super ancestor of a node means it's the ancestor of all the parents of
  that node. Check the below figure as an example. A is super ancestor of
  B, C, D. But B is NOT a super ancestor of D, because D has two parents (B, C),
  and B is not an ancestor of C. This method check if one node is a super
  ancestor for all nodes in a given set.

          ┌───────┐
      ┌───►   A   ◄────┐
      │   └───────┘    │
      │                │
  ┌───┴───┐        ┌───┴───┐
  │   B   │        │   C   │
  └───▲───┘        └───▲───┘
      │                │
      │   ┌───────┐    │
      └───┤   D   ├────┘
          └───────┘

  Args:
    ancestor: The potential super ancestor node to be checked. Note that this
      node does not need to be ancestor of any node.
    nodes: A set of nodes specified by object id.
    node_to_parents_by_id: A dict that maps node id to all of its parent
      objects.

  Returns:
    A bool value that indicates if `ancestor` is a super ancestor of all nodes.
  """
  nodes = list(nodes)
  if len(nodes) == 0:  # pylint: disable=g-explicit-length-test
    return False
  if len(nodes) == 1:
    if ancestor in nodes:
      return True
    all_parents = node_to_parents_by_id[nodes[0]]
    if len(all_parents) == 1 and ancestor in all_parents:
      return True
    return _is_super_ancestor(ancestor, all_parents, node_to_parents_by_id)
  results = []
  for node in nodes:
    if node != ancestor:
      res = _is_super_ancestor(
          ancestor, node_to_parents_by_id[node], node_to_parents_by_id
      )
      results.append(res)
  return all(results)


def _find_least_common_ancestor(
    node_ids: Set[int], node_to_parents_by_id: Dict[int, Set[Any]]
) -> int:
  """Find the least common ancestor of all nodes."""

  node_ids = list(node_ids)
  if len(node_ids) == 0:  # pylint: disable=g-explicit-length-test
    raise ValueError("Input nodes must not be empty.")
  if len(node_ids) == 1:
    return node_ids[0]
  if len(node_ids) == 2:
    x, y = node_ids
    if _is_super_ancestor(x, {y}, node_to_parents_by_id):
      return x
    if _is_super_ancestor(y, {x}, node_to_parents_by_id):
      return y
    x_parents = {parent for parent in node_to_parents_by_id[x]}
    y_parents = {parent for parent in node_to_parents_by_id[y]}
    return _find_least_common_ancestor(
        x_parents.union(y_parents), node_to_parents_by_id
    )
  first_two = set(node_ids[:2])
  rest = set(node_ids[2:])
  first_two_ancestor = _find_least_common_ancestor(
      first_two, node_to_parents_by_id
  )
  rest.add(first_two_ancestor)
  return _find_least_common_ancestor(rest, node_to_parents_by_id)


def _find_least_common_fixture_ancestor(
    node_ids: Set[int],
    node_to_parents_by_id: Dict[int, Set[int]],
    sub_fixture_ids: Set[int],
    root_id: int,
) -> Any:
  """Find the least common sub-fixture."""
  # The definition of `least` here is also refered as `lowest` common ancestor.
  # Basically when both A and B are common ancestor of a set of nodes, and
  # assume A is the ancestor of B, the node with a larger distance to the root
  # is the least common ancestor (B in this case).
  lca = _find_least_common_ancestor(node_ids, node_to_parents_by_id)
  while lca not in sub_fixture_ids and lca != root_id:
    lca = _find_least_common_ancestor(
        node_to_parents_by_id[lca], node_to_parents_by_id
    )
  return lca


def _find_shared_nodes(
    top_level_fn: code_ir.FixtureFunction,
    sub_fixtures: Dict[str, config_lib.Buildable],
    node_to_parents_by_id: Dict[int, Set[int]],
) -> Tuple[Dict[int, Any], Dict[int, Any]]:
  """Find out shared nodes by traversing each sub-fixture.

  The definition of shared nodes are nodes that are shared among multiple
  sub-fixtures. They need to be passed as arguments to sub-fixtures.

  Args:
    top_level_fn: Top-level fixture function.
    sub_fixtures: A dict that maps the sub-fixture name to the actual
      sub-fixture object.
    node_to_parents_by_id: A dict that maps node id to all of its parent
      objects.

  Returns:
    A dict that maps id(shared_node) -> shared_node object.
    A dict that maps id(shared_node) -> path to the shared_node.
  """
  sub_fixture_ids = {id(value) for value in sub_fixtures.values()}
  top_fixture_node_ids = set()

  shared_nodes = {}
  shared_node_paths = {}

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
  ancestor_fixture = {}
  sub_fixture_values = list(sub_fixtures.values())
  for idx, fixture in enumerate(sub_fixture_values):
    for value, path in daglish.iterate(fixture, memoized=False):
      if not daglish.is_unshareable(value):
        used_by_sub_fixture = False
        if id(value) in ancestor_fixture:
          if ancestor_fixture[id(value)] != idx:
            # Check if the node is shared among multiple sub-fixtures.
            # If sub-fixture A contains sub-fixture B, nodes within B should
            # not be classified as shared if they are not used somewhere else.
            ancestor = ancestor_fixture[id(value)]
            ancestor_id = id(sub_fixture_values[ancestor])
            value_id = id(sub_fixture_values[idx])
            if _is_super_ancestor(
                ancestor_id, {value_id}, node_to_parents_by_id
            ):
              ancestor_fixture[id(value)] = idx
            elif not _is_super_ancestor(
                value_id, {ancestor_id}, node_to_parents_by_id
            ):
              used_by_sub_fixture = True
        else:
          ancestor_fixture[id(value)] = idx
        used_by_top_fixture = id(value) in top_fixture_node_ids
        if used_by_top_fixture or used_by_sub_fixture:
          # Do not identify user specified sub-fixtures as shared nodes
          if id(value) not in sub_fixture_ids:
            shared_nodes[id(value)] = value
            shared_node_paths[id(value)] = path

  return shared_nodes, shared_node_paths


def _prepare_node_names(
    shared_nodes: Dict[int, Any],
    shared_node_paths: Dict[int, Any],
    namer: naming.Namer,
) -> Dict[int, code_ir.Name]:
  """Create a dict that maps id(node) -> code_ir.Name object."""
  node_names = {}
  for node_id in shared_nodes:
    node = shared_nodes[node_id]
    path = shared_node_paths[node_id]
    name = namer.name_for(node, [path])
    name = code_ir.Name(name, is_generated=True)
    node_names[id(node)] = name
  return node_names


def _add_var_declarations(
    node_id: int,
    fixture: code_ir.FixtureFunction,
    shared_nodes: Dict[int, Any],
    shared_node_names: Dict[int, code_ir.Name],
) -> None:
  """Add VarDeclaration for shared nodes in the top-level fixture."""
  assert node_id in shared_node_names
  node = shared_nodes[node_id]
  name = shared_node_names[node_id]
  var = code_ir.VariableDeclaration(name=name, expression=node)
  fixture.variables.append(var)


def _prepare_fixture_vars_and_params(
    task: code_ir.CodegenTask,
    sub_fixtures: Dict[str, Any],
    shared_nodes: Dict[int, Any],
    shared_node_names: Dict[int, code_ir.Name],
    node_to_parents_by_id: Dict[int, Set[int]],
):
  """Prepare fixture vars and params."""
  fixture_vars = {}
  fixture_params = {}
  top_fixture = task.top_level_call.fn.output_value
  fixture_ids = {id(fixture) for fixture in sub_fixtures.values()}
  for shared in shared_nodes:
    lcf_id = _find_least_common_fixture_ancestor(
        node_to_parents_by_id[shared],
        node_to_parents_by_id,
        fixture_ids,
        id(top_fixture),
    )
    if lcf_id == id(top_fixture):
      # Add the declaration in the top level fixture.
      _add_var_declarations(
          shared, task.top_level_call.fn, shared_nodes, shared_node_names
      )
      lcf_obj = top_fixture
    else:
      # Prepare sub-fixture var declarations
      if lcf_id not in fixture_vars:
        fixture_vars[lcf_id] = []
      fixture_vars[lcf_id].append(shared)
      lcf_obj = None
      for fixture in sub_fixtures.values():
        if id(fixture) == lcf_id:
          lcf_obj = fixture
          break
      assert lcf_obj is not None
    # Prepare sub-fixture params
    name = shared_node_names[shared]
    for value, _ in daglish.iterate(lcf_obj, memoized=False):
      if not daglish.is_unshareable(value):
        if id(value) in fixture_ids and value is not lcf_obj:
          if id(value) not in fixture_params:
            fixture_params[id(value)] = []
          fixture_params[id(value)].append(code_ir.Parameter(name=name))
  return fixture_vars, fixture_params


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
  3. Add shared nodes to the least common fixture as variable declarations.
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
  top_fixture = task.top_level_call.fn.output_value
  node_to_parents_by_id = _get_node_to_parents_mapping(top_fixture)

  shared_nodes, shared_node_paths = _find_shared_nodes(
      task.top_level_call.fn, sub_fixtures, node_to_parents_by_id
  )
  shared_node_names = _prepare_node_names(
      shared_nodes, shared_node_paths, namer
  )

  fixture_vars, fixture_params = _prepare_fixture_vars_and_params(
      task, sub_fixtures, shared_nodes, shared_node_names, node_to_parents_by_id
  )

  # Map id(sub_fixture) -> names in str
  sub_fixture_names = {id(value): name for name, value in sub_fixtures.items()}
  new_fixtures = []

  def traverse(value: Any, state: daglish.State) -> Any:
    """Actually replace sub-fixtures and shared nodes."""
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
      params = fixture_params.get(original_id, [])
      # TODO(b/285208948): Improve naming for sub-fixture nodes.
      fixture_fn = code_ir.FixtureFunction(
          name=name, parameters=params, variables=[], output_value=value
      )
      if original_id in fixture_vars:
        for var in fixture_vars[original_id]:
          _add_var_declarations(
              var, fixture_fn, shared_nodes, shared_node_names
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
    if isinstance(fixture, partial.ArgFactory):
      raise ValueError(
          "fdl.ArgFactory is not supported in auto-config codegen sub-fixture"
          " transformation."
      )

  namer = make_namer(namespace_lib.Namespace(existing_names))
  _transform_sub_fixtures(task, sub_fixtures, namer)
