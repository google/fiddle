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

"""Library for manipulating DAGs."""

import inspect
from typing import Any, Callable, Dict, Generator, List

from fiddle import daglish


def is_namedtuple_instance(value: Any):
  return daglish.is_namedtuple_subclass(type(value))


TraverseWithPathFn = Callable[[daglish.Path, Any], Generator[None, Any, Any]]


def traverse_with_path(fn: TraverseWithPathFn, structure: Any) -> Any:
  """Traverses `structure`, applying `fn` at each node.

  The given traversal function `fn` is applied at each node of `structure`,
  where a node may be a traversable container object or a leaf (non-traversable)
  value. Traversable containers are lists, tuples, namedtuples (without
  signature-changing `__new__` methods), dicts, and `fdl.Buildable` instances.

  `fn` should be a function taking two parameters, `path` and `value`, where
  `path` is a `Path` instance (tuple of `PathElement`s) and `value` provides the
  corresponding value.

  Additionally, `fn` must contain a single `yield` statement, which relinquishes
  control back to `traverse_with_path` to continue the traversal. The result of
  the traversal is then provided back to `fn` as the value of the yield
  statement (this communication between `fn` and `traverse_with_path` makes `fn`
  a "coroutine").

  For example, a minimal traversal function might just be:

      def traverse(path, value):
        new_value = yield
        return new_value

  For leaf values, the `new_value` provided by the `yield` statement will be
  exactly `value`. For containers, `new_value` will have the same type as
  `value`, but its elements will be the result of the traversal function applied
  to the elements of `value`. (Note that this common pattern of simply returning
  the result of `yield` can be further simplified to just `return (yield)`.)

  If `fn` returns before reaching its `yield` statement, further traversal into
  `value` does not take place. In all cases, the return value of `fn` is used
  directly in place of `value` in the output structure.

  Note: The output of a traversal that returns the post-traversal value (e.g.,
  `return (yield)`) will not maintain object identity relationships between
  different containers in `structure`. In other words, if (for example) the same
  list instance appears twice in `structure`, it will be replaced by two
  separate list instances in the traversal output. If maintaining object
  identity is important, see `memoized_traverse`.

  The following is a more involved example that replaces all tuples with a
  string (returning before `yield` to prevent traversal into the tuple), and
  also replaces all 2s with `None`, before then replacing `None` values in list
  containers with a string after traversal:

      structure = {
          "a": [1, 2],
          "b": (1, 2, 3),
      }

      def replace_twos_and_tuples(unused_path, value):
        if value == 2:
          return None
        elif isinstance(value, tuple):
          return "used to be a tuple..."

        # Provides the post-traversal value! Here, any value of 2 has
        # already been mapped to None.
        new_value = yield

        if isinstance(new_value, list):
          return ["used to be a two..." if x is None else x for x in new_value]
        else:
          return new_value

      output = daglish_legacy.traverse_with_path(
          replace_twos_and_tuples, structure)

      assert output == {
          "a": [1, "used to be a two..."],
          "b": "used to be a tuple...",
      }

  Args:
    fn: The function to apply to each node of `structure`. The function should
      be a coroutine with a single yield statement taking two parameters, `path`
      and `value`.
    structure: The structure to traverse.

  Raises:
    ValueError: If `fn` is not a generator function (i.e., does not contain a
      yield statement).
    RuntimeError: If `fn` yields a non-None value, or yields more than once.

  Returns:
    The structured output from the traversal.
  """
  if not inspect.isgeneratorfunction(fn):
    raise ValueError("`fn` should contain a yield statement.")

  def traverse(path: daglish.Path, structure: Any) -> Any:
    """Recursive traversal implementation."""

    generator = fn(path, structure)
    try:
      if next(generator) is not None:
        raise RuntimeError("The traversal function yielded a non-None value.")
      traverser = daglish.find_node_traverser(type(structure))
      if traverser:
        path_elements = traverser.path_elements(structure)
        values, metadata = traverser.flatten(structure)
        new_values = (
            traverse(path + (path_element,), subtree)
            for path_element, subtree in zip(path_elements, values)
        )  # pyformat: disable
        new_structure = traverser.unflatten(new_values, metadata)
      else:
        new_structure = structure
      generator.send(new_structure)
    except StopIteration as e:
      return e.value  # pytype: disable=attribute-error
    else:
      raise RuntimeError("Does the traversal function have two yields?")

  return traverse((), structure)


def collect_paths_by_id(structure: Any,
                        memoizable_only: bool) -> Dict[int, List[daglish.Path]]:
  """Returns a dict mapping id(v)->paths for all `v` traversable from structure.

  I.e., if `result = collect_paths_by_id(structure)`, then `result[id(v)]` is
  the list of every path `p` such that `follow_path(structure, p) is v`.

  This dict only includes values `v` for which `is_memoizable(v)` is true.

  Args:
    structure: The structure for which the id->paths mapping should be created.
    memoizable_only: If true, then only include values `v` for which
      `is_memoizable(v)` is true.  Currently required to be True, to avoid bugs
      that can result from Python's interning optimizations.
  """
  if not memoizable_only:
    raise ValueError(
        "Including non-memoizable objects when collecting paths by id may "
        "cause problems, because of Python's interning optimizations.  If you "
        "are sure this is what you need, contact the Fiddle team, and we can "
        "look into enabling this flag.")
  paths_by_id = {}

  def collect_paths(path: daglish.Path, value: Any):
    if not memoizable_only or daglish.is_memoizable(value):
      paths_by_id.setdefault(id(value), []).append(path)
    return (yield)

  traverse_with_path(collect_paths, structure)
  return paths_by_id


def collect_value_by_id(structure: Any,
                        memoizable_only: bool) -> Dict[int, Any]:
  """Returns a dict mapping id(v)->v for all `v` traversable from `structure`.

  Args:
    structure: The structure for which the id->paths mapping should be created.
    memoizable_only: If true, then only include values `v` for which
      `is_memoizable(v)` is true.
  """
  value_by_id = {}

  def collect_value(path: daglish.Path, value: Any):
    del path  # Unused.
    if not memoizable_only or daglish.is_memoizable(value):
      value_by_id[id(value)] = value
    return (yield)

  traverse_with_path(collect_value, structure)
  return value_by_id


def collect_value_by_path(structure: Any,
                          memoizable_only: bool) -> Dict[daglish.Path, Any]:
  """Returns a dict mapping `path->v` for all `v` traversable from `structure`.

  Args:
    structure: The structure for which the `path->value` map should be created.
    memoizable_only: If true, then only include values `v` for which
      `is_memoizable(v)` is true.
  """
  value_by_path = {}

  def collect_value(path: daglish.Path, value: Any):
    if not memoizable_only or daglish.is_memoizable(value):
      value_by_path[path] = value
    return (yield)

  traverse_with_path(collect_value, structure)
  return value_by_path


TraverseWithAllPathsFn = Callable[[daglish.Paths, daglish.Path, Any],
                                  Generator[None, Any, Any]]


def traverse_with_all_paths(fn: TraverseWithAllPathsFn, structure):
  """Traverses `structure`, providing all paths to each element.

  Unlike `traverse_with_path`, this function performs an initial traversal and
  collects all paths to each object in the provided `structure`. These paths
  (in addition to the current path) are then provided to `fn` during the main
  traversal.

  See `traverse_with_path` for additional details on how `fn` can control the
  traversal via `return` and `yield` statements.

  Args:
    fn: The function to apply to each node of `structure`. The function should
      be a coroutine with a single yield statement taking three parameters
      `all_paths`, `current_path`, and `value`.
    structure: The structure to traverse.

  Returns:
    The structured output from the traversal.
  """
  if not inspect.isgeneratorfunction(fn):
    raise ValueError("`fn` should contain a yield statement.")

  paths_memo = collect_paths_by_id(structure, memoizable_only=True)

  def wrap_with_paths(current_path: daglish.Path, value: Any):
    all_paths = ((),)
    if daglish.is_memoizable(value):
      all_paths = paths_memo[id(value)]
    elif current_path:
      # Non-memoizable values may have still have multiple paths via a shared
      # parent container. Here we augment all paths to the parent container.
      parent = daglish.follow_path(structure, current_path[:-1])
      parent_paths = paths_memo[id(parent)]
      all_paths = daglish.add_path_element(parent_paths, current_path[-1])
    return (yield from fn(all_paths, current_path, value))

  return traverse_with_path(wrap_with_paths, structure)


MemoizedTraverseFn = Callable[[daglish.Paths, Any], Generator[None, Any, Any]]


def memoized_traverse(fn: MemoizedTraverseFn, structure):
  """Traverses `structure`, memoizing outputs.

  This simplifies the case where the traversal function `fn` should only be
  applied once to each instance of an object in `structure`. During the
  traversal, this function memoizes `fn`'s output for a given input object, and
  reuses it if the same object is encountered again. This behavior preserves
  instance relationships present in `structure`, and is most commonly the
  desired behavior when transforming a graph of `fdl.Buildable` objects. For
  example, `fdl.build()` should only build each `fdl.Buildable` instance once
  (and then reuse the result if an instance is encountered again).

  Like `traverse_with_all_paths`, an initial traversal collects all paths to
  each object in `structure`. However, since each object is encountered only
  once, no `current_path` parameter is supplied to `fn`, only the tuple of all
  paths.

  Note that immutable types that can't contain references (including most
  primitive types like strings and ints) are excluded from memoization. This
  avoids surprising interactions with Python's interning optimizations. See
  `is_memoizable` for additional details.

  See `traverse_with_path` for additional details on how `fn` can control the
  traversal via `return` and `yield` statements.

  Args:
    fn: The function to apply to each node of `structure`. The function should
      be a coroutine with a single yield statement taking two parameters,
      `paths`, and `value`.
    structure: The structure to traverse.

  Returns:
    The structured output from the traversal.
  """
  if not inspect.isgeneratorfunction(fn):
    raise TypeError("`fn` should contain a yield statement.")

  memo = {}

  def wrap_with_memo(all_paths: daglish.Paths, current_path: daglish.Path,
                     value: Any):
    del current_path
    key = id(value)
    if key in memo:
      return memo[key]
    else:
      output = yield from fn(all_paths, value)
      if daglish.is_memoizable(value):
        memo[key] = output
      return output

  return traverse_with_all_paths(wrap_with_memo, structure)
