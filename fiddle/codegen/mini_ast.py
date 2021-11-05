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

These nodes are not super useful compared to something like libcst, since they
often have opaque string values. Their purpose is primarily to help with the
formatting of Python code.
"""

import abc
import dataclasses
from typing import List, Union


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
class ImportNode(CodegenNode):
  name: str  # Name of the module; should precede any class names in code.

  @abc.abstractmethod
  def sortkey(self) -> str:
    """Returns a string module key, which can be used for sorting imports."""
    raise NotImplementedError()

  @abc.abstractmethod
  def change_alias(self, alias_name: str) -> "ImportNode":
    """Creates a version of this import with an alias (`import ... as`)."""


@dataclasses.dataclass(frozen=True)
class DirectImport(ImportNode):
  """Imports a module as a fully-qualified name."""

  def lines(self) -> List[str]:
    return [f"import {self.name}"]

  def sortkey(self) -> str:
    return self.name

  def change_alias(self, alias_name: str) -> ImportNode:
    return ImportAs(alias_name, module=self.name)


@dataclasses.dataclass(frozen=True)
class FromImport(ImportNode):
  """Imports a module name from a base package."""

  parent: str

  def lines(self) -> List[str]:
    return [f"from {self.parent} import {self.name}"]

  def sortkey(self) -> str:
    return f"{self.parent}.{self.name}"

  def change_alias(self, alias_name: str) -> ImportNode:
    return FromImportAs(alias_name, parent=self.parent, module=self.name)


@dataclasses.dataclass(frozen=True)
class ImportAs(ImportNode):
  """Same as DirectImport but with an `as alias` suffix."""

  module: str

  def lines(self) -> List[str]:
    return [f"import {self.module} as {self.name}"]

  def sortkey(self) -> str:
    return self.module

  def change_alias(self, alias_name: str) -> ImportNode:
    return ImportAs(alias_name, self.module)


@dataclasses.dataclass(frozen=True)
class FromImportAs(ImportNode):
  """Same as FromImport but with an `as alias` suffix."""

  parent: str
  module: str

  def lines(self) -> List[str]:
    return [f"from {self.parent} import {self.module} as {self.name}"]

  def sortkey(self) -> str:
    return f"{self.parent}.{self.module}"

  def change_alias(self, alias_name: str) -> ImportNode:
    return FromImportAs(alias_name, self.parent, self.module)


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
  imports: List[CodegenNode]
  builder_body: List[CodegenNode]

  def lines(self) -> List[str]:
    fn_body = ["  " + line for line in block(self.builder_body, [""])]
    return block([block(self.imports, []), ["def build_config():"] + fn_body],
                 ["", ""])
