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

"""APIs for mapping a config DAG to a fixture call stack.

For example, suppose we have this flat config,

x1 = fdl.Config(Foo, 33)
x2 = fdl.Config(Foo, 44)
y1 = fdl.Config(Bar, a=fdl.Config(B1, x1), b=fdl.Config(B2, x1))
y2 = fdl.Config(Bar, a=fdl.Config(B1, x2), b=fdl.Config(B2, x2))
config = fdl.Config(Baz, a=y1, b=y2)

Then it's reasonable to extract a fixture for any Bar, e.g.

def bar_fixture(x):
  return fdl.Config(Bar, a=fdl.Config(B1, x), b=fdl.Config(B2, x))

and then write the main fixture like,

def fixture():
  return fdl.Config(
      Baz,
      a=bar_fixture(fdl.Config(Foo, 33)),
      b=bar_fixture(fdl.Config(Foo, 44)),
  )
"""

import abc
import dataclasses

from fiddle.codegen.auto_config import ir_dag
from fiddle.codegen.auto_config import parents_first_traversal


class CallStackAssigner(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def assign(self, tree: ir_dag.IrDagNode) -> None:
    """Creates fixtures and assigns them to an IR DAG."""
    raise NotImplementedError()


def most_specific_common_call(call_stacks: list[ir_dag.Call]) -> ir_dag.Call:
  """Returns the most specific call that is shared by all call stacks."""
  shared = set(call_stacks[0].to_stack())
  for other in call_stacks[1:]:
    shared &= set(other.to_stack())
  for call in reversed(call_stacks[0].to_stack()):
    if call in shared:
      return call
  raise ValueError(
      f"Expected some shared call node between stacks {call_stacks!r}")


@dataclasses.dataclass
class SpecificValueFixtures(CallStackAssigner):
  """Creates fixtures for specific values in a DAG.

  As many sub-nodes as possible will be assigned/created in the fixture.
  """
  value_id_to_names: dict[int, str]

  def assign(self, tree: ir_dag.IrDagNode) -> None:
    # Generally nodes should only be visited once, so this mapping is usually
    # unnecessary. However, it is permissible to have multiple copies of an
    # IrDagNode for a single *immutable* value, and in this case we don't want
    # to generate copies of the fixture that generates that immutable value.
    node_to_fns = {}

    def traverse(node: ir_dag.IrDagNode,
                 parent_results: list[ir_dag.Call]) -> ir_dag.Call:
      node_id = id(node.value)
      current = node.call
      if node_id in self.value_id_to_names:
        fn = node_to_fns.get(node_id)
        if fn is None:
          fn = ir_dag.FixtureFunction(
              self.value_id_to_names[node_id], return_type=type(node.value))
          node_to_fns[node_id] = fn
        result = ir_dag.Call(node_to_fns[node_id], parent=current)
      elif not parent_results:
        result = current
      else:
        result = most_specific_common_call(parent_results)
      node.call = result
      return result

    parents_first_traversal.traverse_parents_first(traverse, tree)
