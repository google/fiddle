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

"""Defines an intermediate representation DAG node.

At the beginning and throughout codegen, we convert the original DAG to
an IR DAG, and then mutate the IR DAG.

The IR DAG is a set of nodes in the same structure as the original DAG, where
each IR DAG node points to one of the original nodes. The IR DAG nodes will
add metadata like where a node is created, if there are several auto_config
fixtures which call each other.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Type

from fiddle import daglish


@dataclasses.dataclass
class FixtureFunction:
  """Represents a fixture function; see call_stack.py."""
  name: str
  return_type: Type[Any]


@dataclasses.dataclass
class Call:
  """Represents a function call; see call_stack.py."""
  fn: FixtureFunction
  parent: Optional[Call] = None  # Call stack parent.

  def __hash__(self):
    return id(self)

  def to_stack(self) -> list[Call]:
    current = self
    result = [self]
    while current.parent is not None:
      current = current.parent
      result.insert(0, current)
    return result


class Variable:
  name: str
  function: FixtureFunction


class VariableInstance:
  variable: Variable
  call: Call  # Call where this variable is created.


@dataclasses.dataclass
class IrDagNode:
  """An intermediate representation DAG node.

  Attributes:
    children_flat: Child IR DAG nodes, which should correspond to flattened
      value nodes. Currently an empty tuple if `value` is not traversable, but
      we might switch to having a sentinel for clarity.
    children_path_elements: Path elements for the children. Usually just for
      daglish traversal purposes.
    value: Original DAG value.
    all_paths: All paths to original DAG value.
    call: Call (function invocation) where this DAG node is created.
    variable: Variable where this DAG node is assigned. Currently return values
      are not assigned variables, and we also don't reach into configs/values,
      since this is not compatible with the Python mode of auto_config in
      general. So there is only one variable instance. It may be in the parent
      call of `call_stack`.
  """
  children_flat: tuple[IrDagNode]
  children_path_elements: tuple[daglish.PathElement, ...]
  value: Any
  all_paths: list[daglish.Path]
  call: Call
  variable: Optional[VariableInstance]


# Small sentinel object for reusing IrDagNode as its own traversal
# metadata.
class _IrTraversalSentinel:
  pass


_sentinel = _IrTraversalSentinel()


def flatten_ir_dag_node(node: IrDagNode) -> tuple[Any, Any]:
  metadata = dataclasses.replace(node, children_flat=_sentinel)
  return node.children_flat, metadata


def unflatten_ir_dag_node(children_flat, metadata):
  return dataclasses.replace(metadata, children_flat=children_flat)


def ir_dag_node_path_elements(
    node: IrDagNode) -> tuple[daglish.PathElement, ...]:
  return node.children_path_elements


daglish.register_node_traverser(
    IrDagNode,
    flatten_fn=flatten_ir_dag_node,
    unflatten_fn=unflatten_ir_dag_node,
    path_elements_fn=ir_dag_node_path_elements)


def generate_ir_dag(config_or_value: Any,
                    root_fn_name: Optional[str] = None) -> IrDagNode:
  """Creates intermediate DAG nodes for a given config/value."""

  if not root_fn_name:
    root_fn_name = "config_fixture"

  root_call = Call(
      FixtureFunction(name=root_fn_name, return_type=type(config_or_value)))

  def _traverse(value, state: daglish.State) -> IrDagNode:
    all_paths = state.get_all_paths()
    if state.is_traversable(value):
      sub_result = state.flattened_map_children(value)
      children = tuple(sub_result.values)
      children_path_elements = sub_result.path_elements
    else:
      children = ()
      children_path_elements = ()
    return IrDagNode(
        children_flat=children,
        children_path_elements=children_path_elements,
        value=value,
        all_paths=all_paths,
        call=root_call,
        variable=None,
    )

  return daglish.MemoizedTraversal.run(_traverse, config_or_value)
