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

"""Library for generating code from a Config or Partial object."""

import collections
import dataclasses
import inspect
import itertools
import keyword
import re
from typing import Any, Callable, Dict, List, Sequence, Set

from absl import logging
from fiddle import config as fdl
from fiddle.codegen import mini_ast
from fiddle.codegen import special_value_codegen
from fiddle.experimental import daglish
import tree


def _camel_to_snake(name: str) -> str:
  """Converts a camel or studly-caps name to a snake_case name."""
  return re.sub(r"(?<=.)([A-Z])", lambda m: "_" + m.group(0).lower(),
                name).lower()


def map_buildables(
    buildable: fdl.Buildable,
    pre_traverse_fn: Callable[[fdl.Buildable], fdl.Buildable] = lambda x: x,
    post_traverse_fn: Callable[[fdl.Buildable, Dict[str, Any]],
                               fdl.Buildable] = lambda x, new_args: x,
    leaf_fn: Callable[[Any], Any] = lambda x: x,
) -> fdl.Buildable:
  """Maps over a tree of builders.

  Args:
    buildable: Config or Partial instance.
    pre_traverse_fn: Function applied before traversing into a child.
    post_traverse_fn: Function applied after traversing into a child.
    leaf_fn: Function applied to transform any leaf values.

  Returns:
    Dictionary of new arguments, after mapping with this function.
  """

  def map_fn(leaf):
    if isinstance(leaf, fdl.Buildable):
      map_buildables(leaf, pre_traverse_fn, post_traverse_fn, leaf_fn)
    else:
      return leaf_fn(leaf)

  buildable = pre_traverse_fn(buildable)
  new_args = tree.map_structure(map_fn, buildable.__arguments__)
  return post_traverse_fn(buildable, new_args)


# Project-specific import aliases.
_SPECIAL_IMPORT_ALIASES = {}


def register_import_alias(name: str, import_stmt: mini_ast.ImportNode) -> None:
  """Registers an import alias.

  Typically this is called by extensions in `fiddle.extensions`.

  Args:
    name: Full module name to alias. Often, this is what can be found in
      `type(py_value).__module__.__name__`.
    import_stmt: Import statement to emit for this module.
  """
  _SPECIAL_IMPORT_ALIASES[name] = import_stmt


def _get_import_aliases() -> Dict[str, mini_ast.ImportNode]:
  """Dictionary of import aliases."""
  return {
      **_SPECIAL_IMPORT_ALIASES,
      "fiddle.config":
          mini_ast.ImportAs(name="fdl", module="fiddle"),
  }


@dataclasses.dataclass
class Namespace:
  """Manages active Python instance names."""

  names: Set[str] = dataclasses.field(
      default_factory=lambda: set(keyword.kwlist))

  def __contains__(self, key: str) -> bool:
    return key in self.names

  def add(self, name: str) -> str:
    """Adds a name, checking if it already exists, and returns it."""
    if name in self.names:
      raise ValueError(
          f"Tried to add {name!r} (e.g. an import), but it already exists!")
    self.names.add(name)
    return name

  def get_new_name(self, base_name, prefix: str = "shared_") -> str:
    """Generates a new name for a given instance.

    Example:
      self.names = ["foo"]
      get_new_name("bar") == "shared_bar"

    Example 2:
      self.names = ["shared_foo", "bar"]
      get_new_name("Foo") == "shared_foo_2"

    Args:
      base_name: Base name to derive a new name for. This will be converted to
        snake case automatically.
      prefix: Prefix to prepend to any names.

    Returns:
      New unique name for a variable representing a config for that function
      or class.
    """

    name = prefix + _camel_to_snake(base_name)
    if name not in self.names:
      return self.add(name)
    for i in itertools.count(start=2):
      if f"{name}_{i}" not in self.names:
        return self.add(f"{name}_{i}")
    raise AssertionError("pytype helper -- itertools.count() is infinite")


@dataclasses.dataclass
class ImportManager:
  """Helper class to maintain a list of import statements."""

  namespace: Namespace
  imports: List[mini_ast.ImportNode] = dataclasses.field(default_factory=list)
  aliases: Dict[str, mini_ast.ImportNode] = dataclasses.field(
      default_factory=_get_import_aliases)

  def add_by_name(self, module_name: str) -> str:
    """Adds an import given a module name.

    This is a slightly lower-level API than `add`; you should only use it if
    you don't have access.

    Args:
      module_name: String module name to try to import.

    Returns:
      Alias for the imported module.
    """
    result = self.aliases.get(module_name)
    if not result:
      if "." in module_name:
        parent, module_name = module_name.rsplit(".", 1)
        result = mini_ast.FromImport(name=module_name, parent=parent)
      else:
        result = mini_ast.DirectImport(module_name)

    if result not in self.imports:
      if result.name in self.namespace:
        # Create or adjust the alias for the import.
        new_name = self.namespace.get_new_name(result.name, prefix="")
        result = result.change_alias(new_name)
      else:
        self.namespace.add(result.name)
      self.imports.append(result)
    return result.name

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

  def sorted_imports(self):
    """Returns imports sorted lexicographically."""
    return sorted(self.imports, key=lambda x: x.sortkey())


def assignment_path(base_var: str, path: Sequence[daglish.PathElement]) -> str:
  """Generates the LHS of an assignment, given a traversal path.

  Example: ["foo", 3, "bar"] -> "foo[3].bar".

  Args:
    base_var: Base variable name.
    path: Attribute path on `base_var` to assign to.

  Returns:
    Python code string for the LHS of an assignment.

  Raises:
    TypeError: If the first path element is not a string, or if any path element
      is not a string or an int.
  """

  return base_var + "".join(x.code for x in path)


@dataclasses.dataclass(frozen=True)
class _VarReference:
  """Helper class whose repr() is just the provided string."""
  name: str

  def __repr__(self):
    return self.name


def _get_shared_buildables(buildable: fdl.Buildable) -> List[fdl.Buildable]:
  """Finds any sub-buildable nodes which are referenced by multiple parents."""
  # Find shared nodes.
  to_count = collections.Counter()
  children_by_id = {}

  def _count_map_fn(child: fdl.Buildable):
    to_count[id(child)] += 1
    children_by_id[id(child)] = child
    return child

  map_buildables(buildable, _count_map_fn)
  return [
      children_by_id[child_hash]
      for child_hash, count in to_count.items()
      if count > 1
  ]


def _has_child_buildables(value: Any) -> bool:
  result = False

  def leaf_fn(leaf):
    nonlocal result
    result = result or isinstance(leaf, fdl.Buildable)

  tree.map_structure(leaf_fn, value)
  return result


@dataclasses.dataclass
class SharedBuildableManager:
  """Helper class to manage shared configuration objects."""

  namespace: Namespace
  import_manager: ImportManager
  instances: List[mini_ast.CodegenNode] = dataclasses.field(
      default_factory=list)
  instance_names_by_id: Dict[int, str] = dataclasses.field(default_factory=dict)

  def __contains__(self, buildable: fdl.Buildable):
    return id(buildable) in self.instance_names_by_id

  def add(self, name: str, buildable: fdl.Buildable,
          decl: mini_ast.CodegenNode) -> None:
    """Adds a shared instance.

    Args:
      name: Variable name for code representing this instance.
      buildable: Actual Python fdl.Buildable object represented by the code
        being generated.
      decl: Code declaration for this instance.
    """
    self.instances.append(decl)
    self.instance_names_by_id[id(buildable)] = name

  def assign(self, lhs_var: str, lhs_path: Sequence[daglish.PathElement],
             attr_value: Any) -> mini_ast.CodegenNode:
    """Returns an assignment for a Python value.

    When this value references shared builder objects, then those objects are
    replaced with name references to the shared objects.

    Args:
      lhs_var: Variable name of the left-hand side assignment.
      lhs_path: Attribute path on `lhs_var` to assign to.
      attr_value: Python value representing the right-hand side of the
        expression.

    Returns:
      Codegen node representing the assignment.
    """

    if lhs_path and isinstance(lhs_path[-1], daglish.Index):
      # Skip if we're re-assigning to a list element. The only case when we want
      # to override list elements is when we're assigning to sub-configurations,
      # which were initially assigned NotImplemented. However, these are set in
      # the top stanza of configure_main_tree_block(), not in calls to this
      # method, so we can just return here without further checks.
      return mini_ast.Noop()

    used_not_implemented = False

    def _map_fn(child):
      nonlocal used_not_implemented
      if child in self:
        return _VarReference(self.instance_names_by_id[id(child)])
      elif isinstance(child, fdl.Buildable):
        used_not_implemented = True
        return _VarReference("NotImplemented")
      else:
        return special_value_codegen.transform_py_value(child,
                                                        self.import_manager)

    lhs = assignment_path(lhs_var, lhs_path)
    assignment = mini_ast.Assignment(
        lhs, repr(tree.map_structure(_map_fn, attr_value)))
    if used_not_implemented:
      return mini_ast.TrailingComment(
          assignment, "fdl.Config sub-nodes will replace NotImplemented")
    else:
      return assignment


def _configure_shared_object(
    buildable: fdl.Buildable,
    shared_manager: SharedBuildableManager,
    import_manager: ImportManager,
    variable_name_prefix: str = "shared_",
) -> None:
  """Generates configuration for a shared object.

  The objects configured will be added to `shared_manager`.

  Args:
    buildable: Buildable for which we will generate configuration code. Its
      dependencies can form a DAG, and any leaf nodes will be added first.
    shared_manager: Shared object manager.
    import_manager: Import manager.
    variable_name_prefix: Prefix for any variables introduced.
  """

  def _inner(child: fdl.Buildable, new_args: Dict[str, Any]) -> fdl.Buildable:
    """Generates code for a shared instance."""
    if child in shared_manager:
      # Already added by another pass (for chain dependencies of shared
      # objects).
      return child

    del new_args  # unused
    # Name this better..
    name = shared_manager.namespace.get_new_name(
        child.__fn_or_cls__.__name__, prefix=variable_name_prefix)
    relname = import_manager.add(child.__fn_or_cls__)
    buildable_subclass_str = import_manager.add(child.__class__)
    nodes = [mini_ast.Assignment(name, f"{buildable_subclass_str}({relname})")]
    for key, value in child.__arguments__.items():
      path = [daglish.BuildableAttr(key)]
      nodes.append(shared_manager.assign(name, path, value))
    shared_manager.add(name, child, mini_ast.ImmediateAttrsBlock(nodes))
    return child

  map_buildables(buildable, post_traverse_fn=_inner)


def codegen_dot_syntax(buildable: fdl.Buildable) -> mini_ast.CodegenNode:
  """Generates code, preferring nested dot-attribute assignment when possible.

  Example code output (abbreviated and with additional comments):

  # Shared instances are on top, since this format supports DAGs. However
  # choosing names is hard.
  shared_foo = fdl.Config(Foo)
  shared_foo.a = 1

  # Subsequent blocks set up a tree, doing attribute dot-assignment.
  root = fdl.Config(Baz)
  root.foo = shared_foo
  root.bar = fdl.Config(Bar)
  root.bar.foo2 = shared_foo

  Args:
    buildable: Config or Partial instance.

  Returns:
    Codegen node representing output code.
  """
  namespace = Namespace()
  import_manager = ImportManager(namespace)
  shared_manager = SharedBuildableManager(
      namespace, import_manager=import_manager)

  # In this method, we will configure any shared objects. This method is fully
  # DAG compliant, and we'd consider using it for the whole codegen, if it were
  # easier to name things.
  for shared_obj in _get_shared_buildables(buildable):
    _configure_shared_object(
        shared_obj,
        shared_manager=shared_manager,
        import_manager=import_manager,
    )

  # Once we have shared objects detected and set up, the rest of the config
  # becomes a tree.
  main_tree_blocks = []

  def configure_main_tree_block(child: fdl.Buildable, path: List[Any]):
    """Configures a tree node for the main configuration block."""
    relname = import_manager.add(child.__fn_or_cls__)
    buildable_subclass_str = import_manager.add(child.__class__)
    nodes = [
        mini_ast.Assignment(
            assignment_path("root", path),
            f"{buildable_subclass_str}({relname})")
    ]
    deferred = []  # Defer configuring sub-Buildable nodes.

    def handle_child_attr(sub_path, value):
      """Inner handler for traverse_with_path, assigning attributes."""

      if not sub_path:
        # Skip top level __arguments__ dict.
        return

      # Convert sub_path to a daglish path. In the future, this traversal will
      # be part of daglish.
      arg_dict_key, *rest = sub_path
      rest_path_elts = []
      for tree_lib_elt in rest:
        if isinstance(tree_lib_elt, int):
          rest_path_elts.append(daglish.Index(tree_lib_elt))
        else:
          rest_path_elts.append(daglish.Key(tree_lib_elt))
      full_path = [*path, daglish.BuildableAttr(arg_dict_key), *rest_path_elts]

      if rest_path_elts and not _has_child_buildables(value):
        pass
      elif isinstance(value, fdl.Buildable) and value not in shared_manager:
        deferred.append((value, full_path))
      else:
        nodes.append(shared_manager.assign("root", full_path, value))

    tree.traverse_with_path(handle_child_attr, child.__arguments__)
    main_tree_blocks.append(mini_ast.ImmediateAttrsBlock(nodes))

    # Recurses to configure sub-Buildable nodes.
    for sub_child, sub_path in deferred:
      configure_main_tree_block(sub_child, sub_path)

  configure_main_tree_block(buildable, [])

  # Adds the final return statement, glues together the shared instance block
  # with the main tree block, and adds imports.
  main_tree_blocks.append(mini_ast.ReturnStmt("root"))
  return mini_ast.ConfigBuilder(import_manager.sorted_imports(), [
      mini_ast.SharedThenResultAssignment(shared_manager.instances,
                                          main_tree_blocks)
  ])
