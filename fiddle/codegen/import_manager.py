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

"""Manages import statements for code generation."""

import dataclasses
import functools
import inspect
from typing import Any, Dict, Union

from absl import logging
from fiddle.codegen import namespace
import libcst as cst

AnyImport = Union[cst.Import, cst.ImportFrom]
_dummy_module_for_formatting = cst.Module([])


@functools.lru_cache(maxsize=128)
def parse_import(stmt: str) -> AnyImport:
  """Returns a LibCST node for an import statement.

  Args:
    stmt: String form of the input.
  """
  parsed = cst.helpers.parse_template_statement(stmt)
  if not isinstance(parsed, cst.SimpleStatementLine):
    raise ValueError(f"Got unexpected type {parsed} from {stmt!r}")
  if len(parsed.body) != 1:
    raise ValueError("Expected only one line in {stmt!r}")
  result = parsed.body[0]
  if not isinstance(result, (cst.Import, cst.ImportFrom)):
    raise TypeError(
        f"Unexpected import {result}, expected an Import or ImportFrom node.")
  return result


def _get_import_name_node(node: AnyImport) -> cst.ImportAlias:
  if len(node.names) != 1:
    raise ValueError(
        f"CST nodes with more than 1 name are not supported; got {node}")
  return node.names[0]


def get_import_name(node: AnyImport) -> str:
  """Returns the name for an import.

  For most imports, this is a single string for the Python accessible name. For
  dotted imports, like "import fiddle.tagging", it is the dotted name.

  Args:
    node: Any import.
  """
  name = _get_import_name_node(node)
  if name.asname:
    return _dummy_module_for_formatting.code_for_node(name.asname.name)
  else:
    return _dummy_module_for_formatting.code_for_node(name.name)


def get_namespace_name(node: AnyImport) -> str:
  """Returns the name of an import taken up in the Python namespace.

  This is the same as get_import_name except for dotted imports like `foo.bar`,
  where it is the first name (`foo`).

  Args:
    node: Any import.
  """
  return get_import_name(node).split(".", 1)[0]


def change_alias(node: AnyImport, new_name: str) -> AnyImport:
  name = _get_import_name_node(node)
  return node.with_changes(
      names=[name.with_changes(asname=cst.AsName(cst.Name(new_name)))])


def get_full_module_name(node: AnyImport) -> str:
  """Returns the fully-qualified module name for an import."""
  name_str = _dummy_module_for_formatting.code_for_node(
      _get_import_name_node(node).name)
  if isinstance(node, cst.ImportFrom):
    module_str = _dummy_module_for_formatting.code_for_node(node.module)
    return f"{module_str}.{name_str}"
  else:
    return name_str


# Project-specific import aliases.
_SPECIAL_IMPORT_ALIASES = {
    "fiddle.config": parse_import("import fiddle as fdl"),
    "fiddle": parse_import("import fiddle as fdl"),
}


def register_import_alias(name: str, import_stmt: str) -> None:
  """Registers an import alias.

  Typically this is called by extensions in `fiddle.extensions`.

  Args:
    name: Full module name to alias. Often, this is what can be found in
      `type(py_value).__module__.__name__`.
    import_stmt: Import statement for this module, which will be parsed by
      LibCST.
  """
  _SPECIAL_IMPORT_ALIASES[name] = parse_import(import_stmt)


def _make_import(full_module_name: str) -> AnyImport:
  """Makes an import statement from a string module name."""
  if "." in full_module_name:
    parent_name, name = full_module_name.rsplit(".", 1)
    return parse_import(f"from {parent_name} import {name}")
  else:
    return parse_import(f"import {full_module_name}")


@dataclasses.dataclass
class ImportManager:
  """Helper class to maintain a list of import statements."""

  namespace: namespace.Namespace
  imports_by_full_name: Dict[str, AnyImport] = dataclasses.field(
      default_factory=dict)

  def _compatible_with_existing(self, stmt: AnyImport) -> bool:
    """Returns whether an import is compatible with existing ones.

    For example, if the current imports are,

    from foo import bar
    import foo.baz

    then 'import foo.bar' is fine, and 'from qux import foo' is not. As an edge
    case, 'from foo import foo' is also not okay, because this is trying to
    import 'foo.foo' as 'foo', which will cause problems when importing
    'foo.bar'.

    Args:
      stmt: Import statement being added.
    """
    if isinstance(stmt, cst.ImportFrom):
      # Just return False for all `from _ import _` statements.
      return False

    base_module_name = get_full_module_name(stmt).split(".")[0]
    namespace_name = get_namespace_name(stmt)
    for node in self.imports_by_full_name.values():
      if get_namespace_name(node) == namespace_name:
        return get_full_module_name(node).split(".")[0] == base_module_name
    return None

  def add_by_name(self, full_module_name: str) -> str:
    """Adds an import given a module name.

    This is a slightly lower-level API than `add`; you should only use it if
    you don't have access to a function or class to pass to `add`.

    Args:
      full_module_name: String module name to try to import.

    Returns:
      Name for the imported module. This is usually the last name, possibly
      followed by a numeric suffix if necessary to disambiguate from other
      imports or variables. In a few cases where special import aliases are
      applied, then a name with a "." may be emitted.
    """
    result = _SPECIAL_IMPORT_ALIASES.get(full_module_name)
    if result is None:
      result = _make_import(full_module_name)

    # Since multiple things could be aliased to the same import, rewrite
    # the module name to the alias' module name.
    full_module_name = get_full_module_name(result)

    if full_module_name in self.imports_by_full_name:
      return get_import_name(self.imports_by_full_name[full_module_name])

    # Add a new import.
    namespace_name = get_namespace_name(result)
    if self._compatible_with_existing(result):
      pass
    elif namespace_name in self.namespace:
      # Create or adjust the alias for the import.
      new_name = self.namespace.get_new_name(namespace_name, prefix="")
      result = change_alias(result, new_name)
    else:
      self.namespace.add(namespace_name)

    self.imports_by_full_name[full_module_name] = result
    return get_import_name(result)

  def add(self, fn_or_cls: Any) -> str:
    """Adds an import if it doesn't exist.

    This adds an import statement to this manager.

    Args:
      fn_or_cls: Function or class.

    Returns:
      Relative-qualified name for the instance.
    """
    module_name = inspect.getmodule(fn_or_cls).__name__
    fn_or_cls_name = fn_or_cls.__qualname__
    if module_name == "__main__":
      logging.warning(
          "%s's module is __main__, so an import couldn't be added.",
          fn_or_cls_name)
      return fn_or_cls_name

    imported_name = self.add_by_name(module_name)
    return f"{imported_name}.{fn_or_cls_name}"

  def sorted_import_nodes(self):
    """Returns imports sorted lexicographically."""
    # Note: It's a bit iffy to deduplicate LibCST nodes, because they can
    # contain a lot of trivial variance. We mostly use this to deduplicate
    # cases where we added the same import name for multiple aliases.
    return sorted(
        frozenset(self.imports_by_full_name.values()), key=get_full_module_name)

  def sorted_import_lines(self):
    """Returns imports sorted lexicographically."""
    return [
        cst.SimpleStatementLine(body=[import_stmt])
        for import_stmt in self.sorted_import_nodes()
    ]
