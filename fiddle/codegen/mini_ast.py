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

"""Very minimal AST library.

We are transitioning to LibCST, so please don't use this library. It is only for
supporting older code.
"""

import abc
import dataclasses
from typing import List, Union

import libcst as cst


class CodegenNode(metaclass=abc.ABCMeta):
  """Super-lightweight codegen node."""

  @abc.abstractmethod
  def lines(self) -> List[str]:
    """Returns lines of code for this element.

    If the final lines should be indented, any outer nodes should modify the
    output of this method.

    Returns:
      Lines of code.
    """
    pass


@dataclasses.dataclass(frozen=True)
class Noop(CodegenNode):
  """No-op node, when emitting something is convenient."""

  def lines(self) -> List[str]:
    return []


@dataclasses.dataclass(frozen=True)
class Assignment(CodegenNode):
  """Assigns an expression to a left-hand-side expression."""
  lhs: str  # Dot-separate path.
  expr: str  # Expression to assign.

  def lines(self) -> List[str]:
    return [f"{self.lhs} = {self.expr}"]


@dataclasses.dataclass(frozen=True)
class ReturnStmt(CodegenNode):
  """Returns an expression."""
  expr: str  # Expression to return.

  def lines(self) -> List[str]:
    return [f"return {self.expr}"]


@dataclasses.dataclass(frozen=True)
class TrailingComment(CodegenNode):
  """Attaches a trailing comment to the last line of another node."""
  child: CodegenNode
  comment: str

  def lines(self) -> List[str]:
    comment = f"# {self.comment}"
    lines = self.child.lines()
    return [*lines[:-1], f"{lines[-1]}  {comment}"] if lines else [comment]


def block(sub_nodes_or_lines: List[Union[str, List[str], CodegenNode]],
          separator: List[str]) -> List[str]:
  """Helper to generate lines for a block of code.

  Args:
    sub_nodes_or_lines: Items to join, which are either string constants or
      codegen nodes.
    separator: Lines to insert between items to join.

  Returns:
    List of lines, taken from `sub_nodes_or_lines`.
  """
  sub_nodes_or_lines = filter(None, sub_nodes_or_lines)  # Remove empty items.

  result = []
  for i, item in enumerate(sub_nodes_or_lines):
    if i > 0:
      result.extend(separator)

    if isinstance(item, str):
      result.append(item)
    elif isinstance(item, list) and all(isinstance(x, str) for x in item):
      result.extend(item)
    elif isinstance(item, CodegenNode):
      result.extend(item.lines())
    else:
      raise TypeError(f"Unsupported item in argument to block: {item}")
  return result


@dataclasses.dataclass(frozen=True)
class ImmediateAttrsBlock(CodegenNode):
  """Block of code setting up immediate objects (as opposed to nested ones)."""
  nodes: List[CodegenNode]

  def lines(self) -> List[str]:
    return block(self.nodes, [])


@dataclasses.dataclass(frozen=True)
class SharedThenResultAssignment(CodegenNode):
  """Block of code setting shared instances, then a tree."""
  shared_instances: List[CodegenNode]
  tree_blocks: List[CodegenNode]

  def lines(self) -> List[str]:
    return block(self.shared_instances + self.tree_blocks, [""])


@dataclasses.dataclass(frozen=True)
class ConfigBuilder(CodegenNode):
  """Top-level node for imports, followed by builder code.

  By default, this generates a `build_config()` method.
  """
  imports: List[cst.SimpleStatementLine]
  builder_body: List[CodegenNode]

  def lines(self) -> List[str]:
    import_lines = cst.Module(body=self.imports).code.splitlines()
    fn_body = ["  " + line for line in block(self.builder_body, [""])]
    return block([block([import_lines], []), ["def build_config():"] + fn_body],
                 ["", ""])
