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

"""Library for converting generating fiddlers from diffs."""

import collections
import functools
import re
import types
from typing import Any, Callable, Dict, List, Set, Tuple

from fiddle import config
from fiddle import daglish
from fiddle.codegen import codegen
from fiddle.codegen import py_val_to_cst_converter
from fiddle.experimental import daglish_legacy
from fiddle.experimental import diff as fdl_diff

import libcst as cst


def fiddler_from_diff(diff: fdl_diff.Diff,
                      old: Any = None,
                      func_name: str = 'fiddler',
                      param_name: str = 'cfg'):
  """Returns the CST for a fiddler function that applies the changes in `diff`.

  The returned `cst.Module` consists of a set of `import` statements for any
  necessary imports, followed by a function definition for a function whose
  name is `func_name`, which takes a single parameter named `param_name`
  containing a `fdl.Config` (or other `Buildable` or structure), and mutates
  that `Config` in-place as described by `diff`.

  The body of the returned function has three sections:

  * The first section creates variables for any new shared values that are
    added by the diff (i.e., values in `diff.new_shared_values`).
  * The second section creates variables to act as aliases for values in the
    in the input `Config`.  This ensures that we can still reference those
    values even after we've made mutations to the `Config` that might make
    them unreachable from their original location.
  * The final section modifies the `Config` in-place, as described by
    `diff.changes`.  Changes are grouped by the parent object that they modify.
    This section contains one statement for each change.

  Args:
    diff: A `fdl.Diff` describing the change that should be made by the fiddler
      function.
    old: The original config that is transformed by `diff`.  If specified, then
      this is used when creating aliases for values in the input `Config` to
      determine which paths need to have aliases created.  (In particular, it is
      used to determine which paths are aliases for one another.)  If not
      specified, then pessimistically assume that aliases must be created for
      all referenced paths.
    func_name: The name for the fiddler function.
    param_name: The name for the parameter to the fiddler function.

  Returns:
    An `cst.Module` object.  You can convert this to a string using
    `result.code`.
  """
  # Create a namespace to keep track of variables that we add.  Reserve the
  # names of the param & func.
  namespace = codegen.Namespace()
  namespace.add(param_name)
  namespace.add(func_name)

  import_manager = codegen.ImportManager(namespace)

  # Get a list of paths that are referenced by the diff.
  used_paths = _find_used_paths(diff)

  # Add variables for any used paths where the value (or any of the value's
  # ancestors) will be replaced by a change in the diff.  If we don't have an
  # `old` structure, then we pessimistically assume that we need to create
  # variables for all used paths.
  moved_value_names = {}
  if old is not None:
    modified_paths = set([change.target for change in diff.changes])
    _add_path_aliases(modified_paths, old)
    for path in sorted(used_paths, key=daglish.path_str):
      if any(path[:i] in modified_paths for i in range(len(path) + 1)):
        moved_value_names[path] = namespace.get_new_name(
            _path_to_name(path), f'moved_{param_name}_')
  else:
    for path in sorted(used_paths, key=daglish.path_str):
      moved_value_names[path] = namespace.get_new_name(
          _path_to_name(path), f'original_{param_name}_')

  # Add variables for new shared values added by the diff.
  new_shared_value_names = [
      namespace.get_new_name(_name_for_value(value))
      for value in diff.new_shared_values
  ]

  # Construct a PyValToCstConverter to convert constants to CST.
  value_converters = [
      py_val_to_cst_converter.ValueConverter(
          matcher=types.ModuleType,
          priority=200,
          converter=functools.partial(
              _convert_module, import_manager=import_manager)),
      py_val_to_cst_converter.ValueConverter(
          matcher=fdl_diff.Reference,
          priority=200,
          converter=functools.partial(
              _convert_reference,
              param_name=param_name,
              moved_value_names=moved_value_names,
              new_shared_value_names=new_shared_value_names)),
  ]

  pyval_to_cst = functools.partial(
      py_val_to_cst_converter.convert_py_val_to_cst,
      additional_converters=value_converters)

  body = []
  body += _cst_for_new_shared_value_variables(diff.new_shared_values,
                                              new_shared_value_names,
                                              pyval_to_cst)
  body += _cst_for_moved_value_variables(param_name, moved_value_names,
                                         pyval_to_cst)
  body += _cst_for_changes(diff, param_name, moved_value_names, pyval_to_cst)

  imports = _cst_for_imports(import_manager)
  fiddler = _cst_for_fiddler(func_name, param_name, body, bool(imports))

  result = cst.Module(body=imports + [fiddler])

  return result


def _cst_for_imports(
    import_manager: codegen.ImportManager) -> List[cst.CSTNode]:
  """Returns a list of `cst.CSTNode` for import satements in `import_manager`."""
  imp_lines = []
  for imp in import_manager.sorted_imports():
    imp_lines.extend(imp.lines())
  import_str = '\n'.join(imp_lines) + '\n'
  module = cst.parse_module(import_str)
  return module.body


def _cst_for_fiddler(func_name: str, param_name: str, body: List[cst.CSTNode],
                     add_leading_blank_line: bool) -> cst.FunctionDef:
  """Returns an `cst.FunctionDef` for the fiddler function.

  Args:
    func_name: The name of the fiddler function.
    param_name: The name of the fiddler function's parameter.
    body: The body of the fiddler function.
    add_leading_blank_line: If true, add a leading blank line.
  """
  return cst.FunctionDef(
      name=cst.Name(func_name),
      params=cst.Parameters(
          params=[cst.Param(name=cst.Name(param_name), star='')]),
      body=cst.IndentedBlock(body),
      leading_lines=[cst.EmptyLine()] if add_leading_blank_line else [])


# A function that takes any python value, and returns a CSTNode.
PyValToCstFunc = Callable[[Any], cst.CSTNode]


def _cst_for_new_shared_value_variables(
    values: Tuple[Any], names: List[str],
    pyval_to_cst: PyValToCstFunc) -> List[cst.CSTNode]:
  """Returns a list of `CSTNode`s for creating new shared value variables."""
  statements = []
  for value, name in sorted(zip(values, names), key=lambda item: item[1]):
    statements.append(
        cst.Assign(
            targets=[cst.AssignTarget(target=cst.Name(name))],
            value=pyval_to_cst(value)))
  return [cst.SimpleStatementLine([stmt]) for stmt in statements]


def _cst_for_moved_value_variables(
    param_name: str, moved_value_names: Dict[daglish.Path, str],
    pyval_to_cst: PyValToCstFunc) -> List[cst.CSTNode]:
  """Returns a list of `CSTNode`s for creating moved value alias variables."""
  statements = []
  sorted_moved_value_names = sorted(
      moved_value_names.items(), key=lambda item: daglish.path_str(item[0]))
  for path, name in sorted_moved_value_names:
    statements.append(
        cst.Assign(
            targets=[cst.AssignTarget(target=cst.Name(name))],
            value=_cst_for_path(param_name, path, pyval_to_cst)))
  return [cst.SimpleStatementLine([stmt]) for stmt in statements]


def _find_used_paths(diff: fdl_diff.Diff) -> Set[daglish.Path]:
  """Returns a list of paths referenced in `diff`.

  This list includes paths for any values we might need to create aliases
  for, if that value moved.  In particular, it includes the parent path
  for each change in `diff.changes`, plus the target path for any
  `diff.Reference` in `diff` whose root is `'old'`.

  Args:
    diff: The `fdl.Diff` that should be scanned for used paths.
  """
  # For each change, we need the path to its *parent* object.
  used_paths = set(change.target[:-1] for change in diff.changes)

  # For each Reference to `old`, we need the target path.
  def collect_ref_targets(path, node):
    del path  # Unused.
    yield
    if isinstance(node, fdl_diff.Reference) and node.root == 'old':
      used_paths.add(node.target)

  for change in diff.changes:
    if isinstance(change, (fdl_diff.SetValue, fdl_diff.ModifyValue)):
      daglish_legacy.traverse_with_path(collect_ref_targets, change.new_value)
  daglish_legacy.traverse_with_path(collect_ref_targets, diff.new_shared_values)

  return used_paths


def _add_path_aliases(paths: Set[daglish.Path], structure: Any):
  """Update `paths` to include any other paths that reach the same objects.

  If any value `v` reachable by a path `p` in `paths` is also reachable by one
  or more other paths, then add those paths to `paths`.  E.g., if a shared
  object is reachable by paths `.x.y` and `.x.z', and `paths` includes
  only `.x.y`, then this will add `.x.z` to `paths`.

  Args:
    paths: A set of paths to values in `structure`.
    structure: The structure used to determine the paths for shared values.
  """
  path_to_value = daglish_legacy.collect_value_by_path(
      structure, memoizable_only=True)
  id_to_paths = daglish_legacy.collect_paths_by_id(
      structure, memoizable_only=True)

  for path in list(paths):
    value = path_to_value.get(path, None)  # None if not memoizable.
    if value is not None:
      paths.update(id_to_paths[id(value)])


ChangesByParent = List[Tuple[daglish.Path, List[fdl_diff.DiffOperation]]]


def _group_changes_by_parent(diff: fdl_diff.Diff) -> ChangesByParent:
  """Returns a sorted list of changes in `diff`, grouped by their parent."""
  # Group changes by parent path.
  changes_by_parent = collections.defaultdict(list)
  for change in diff.changes:
    path = change.target
    if not path:
      raise ValueError('Changing the root object is not supported')
    changes_by_parent[path[:-1]].append(change)

  # Sort by path (converted to path_str).
  return sorted(
      changes_by_parent.items(), key=lambda item: daglish.path_str(item[0]))


def _cst_for_changes(diff: fdl_diff.Diff, param_name: str,
                     moved_value_names: Dict[daglish.Path, str],
                     pyval_to_cst: PyValToCstFunc) -> List[cst.CSTNode]:
  """Returns a list of CST nodes that apply the changes described in `diff`.

  Args:
    diff: The `fdl.Diff` whose changes should be applied.
    param_name: The name of the parameter to the fiddler function.
    moved_value_names: Dictionary mapping any paths that might become
      unreachable once the config is mutated to alias variables that can be used
      to reach those values.
    pyval_to_cst: A function used to convert Python values to CST.
  """
  body = []

  # Apply changes to a single parent at a time.
  for parent_path, changes in _group_changes_by_parent(diff):

    # Get a CST expression that can be used to refer to the parent.
    if parent_path in moved_value_names:
      parent_cst = cst.Name(moved_value_names[parent_path])
    else:
      parent_cst = _cst_for_path(param_name, parent_path, pyval_to_cst)

    # Add CST statements that apply the changes to the parent.  Ensure that
    # all DeleteValues occur before Buildable.__fn_or_cls__ is changed, and
    # that all SetValues occur after Buildable.__fn_or_cls__ is changed
    # (because changing __fn_or_cls__ can change the set of parameters that
    # a Buildable is allowed to have).
    deletes = []
    update_callable = None
    assigns = []
    for change in changes:
      child_path_elt = change.target[-1]
      child_cst = _cst_for_child(parent_cst, child_path_elt, pyval_to_cst)

      if isinstance(child_path_elt, daglish.BuildableFnOrCls):
        assert isinstance(change, fdl_diff.ModifyValue)
        assert update_callable is None
        new_value_cst = pyval_to_cst(change.new_value)
        update_callable = cst.Expr(
            cst.Call(
                func=pyval_to_cst(config.update_callable),
                args=[cst.Arg(parent_cst),
                      cst.Arg(new_value_cst)]))

      elif isinstance(change, fdl_diff.DeleteValue):
        deletes.append(cst.Del(target=child_cst))

      elif isinstance(change, fdl_diff.RemoveTag):
        arg_name = change.target[-1].name
        deletes.append(
            cst.Expr(
                cst.Call(
                    func=pyval_to_cst(config.remove_tag),
                    args=[
                        cst.Arg(parent_cst),
                        cst.Arg(pyval_to_cst(arg_name)),
                        cst.Arg(pyval_to_cst(change.tag))
                    ])))

      elif isinstance(change, (fdl_diff.SetValue, fdl_diff.ModifyValue)):
        new_value_cst = pyval_to_cst(change.new_value)
        assigns.append(
            cst.Assign(
                targets=[cst.AssignTarget(child_cst)], value=new_value_cst))

      elif isinstance(change, fdl_diff.AddTag):
        arg_name = change.target[-1].name
        assigns.append(
            cst.Expr(
                value=cst.Call(
                    func=pyval_to_cst(config.add_tag),
                    args=[
                        cst.Arg(parent_cst),
                        cst.Arg(pyval_to_cst(arg_name)),
                        cst.Arg(pyval_to_cst(change.tag))
                    ])))

      else:
        raise ValueError(f'Unsupported DiffOperation {type(change)}')

    body.extend(deletes)
    if update_callable is not None:
      body.append(update_callable)
    body.extend(assigns)

  return [cst.SimpleStatementLine([stmt]) for stmt in body]


def _cst_for_child(parent_cst: cst.CSTNode, child_path_elt: daglish.PathElement,
                   pyval_to_cst: PyValToCstFunc) -> cst.CSTNode:
  """Returns a CST expression that can be used to access a child of a parent.

  Args:
    parent_cst: CST expression for the parent object.
    child_path_elt: A PathElement specifying a child of the parent.
    pyval_to_cst: A function used to convert Python values to CST.
  """
  if isinstance(child_path_elt, daglish.Attr):
    return cst.Attribute(value=parent_cst, attr=cst.Name(child_path_elt.name))
  elif isinstance(child_path_elt, daglish.Index):
    index_cst = pyval_to_cst(child_path_elt.index)
    return cst.Subscript(
        value=parent_cst,
        slice=[cst.SubscriptElement(slice=cst.Index(index_cst))])
  elif isinstance(child_path_elt, daglish.Key):
    key_cst = pyval_to_cst(child_path_elt.key)
    return cst.Subscript(
        value=parent_cst,
        slice=[cst.SubscriptElement(slice=cst.Index(key_cst))])
  else:
    raise ValueError(f'Unsupported PathElement {type(child_path_elt)}')


def _cst_for_path(name: str, path: daglish.Path, pyval_to_cst: PyValToCstFunc):
  """Converts a `daglish.Path` to an `cst.CSTNode` expression."""
  node = cst.Name(name)
  for path_elt in path:
    node = _cst_for_child(node, path_elt, pyval_to_cst)
  return node


def _camel_to_snake(name: str) -> str:
  """Converts a camel or studly-caps name to a snake_case name."""
  return re.sub(r'(?<=.)([A-Z])', lambda m: '_' + m.group(0).lower(),
                name).lower()


def _name_for_value(value: Any) -> str:
  """Returns a name for a value, based on its type."""
  if isinstance(value, config.Buildable):
    return _camel_to_snake(value.__fn_or_cls__.__name__)
  else:
    return _camel_to_snake(type(value).__name__)


def _path_to_name(path: daglish.Path) -> str:
  """Converts a path to a variable name."""
  name = daglish.path_str(path)
  name = re.sub('[^a-zA-Z_0-9]+', '_', name)
  return name.strip('_').lower()


def _convert_reference(value, convert_child, param_name, moved_value_names,
                       new_shared_value_names) -> cst.CSTNode:
  """Converts a `Reference` to a CST expression."""
  if value.root == 'old':
    if value.target in moved_value_names:
      return cst.Name(moved_value_names[value.target])
    else:
      return _cst_for_path(param_name, value.target, convert_child)
  else:
    assert isinstance(value.target[0], daglish.Index)
    var_name = new_shared_value_names[value.target[0].index]
    return _cst_for_path(var_name, value.target[1:], convert_child)


def _convert_module(value, convert_child, import_manager) -> cst.CSTNode:
  """Converts a Module to CST, using an ImportManager."""
  del convert_child  # Unused.
  name = import_manager.add_by_name(value.__name__)
  return py_val_to_cst_converter.dotted_name_to_cst(name)
