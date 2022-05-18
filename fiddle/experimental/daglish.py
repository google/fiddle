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

import abc
import copy
import dataclasses
import functools
import inspect
import typing
from typing import Any, Callable, Dict, Generator, Iterable, List, Tuple, Union

from fiddle import config


class PathElement(metaclass=abc.ABCMeta):
  """Element of a path."""

  @property
  @abc.abstractmethod
  def code(self) -> str:
    """Generates code for accessing this path."""
    raise NotImplementedError()

  @abc.abstractmethod
  def follow(self, container) -> Any:
    """Returns the element of `container` specified by this path element."""


@dataclasses.dataclass(frozen=True)
class Index(PathElement):
  """An index into a sequence (list or tuple)."""
  index: int

  @property
  def code(self) -> str:
    return f"[{self.index}]"

  def follow(self, container: Union[List[Any], Tuple[Any, ...]]) -> Any:
    return container[self.index]


@dataclasses.dataclass(frozen=True)
class Key(PathElement):
  """A key of a mapping (e.g., dict)."""
  key: Any

  @property
  def code(self) -> str:
    return f"[{self.key!r}]"

  def follow(self, container: Dict[Any, Any]) -> Any:
    return container[self.key]


@dataclasses.dataclass(frozen=True)
class Attr(PathElement):
  """An attribute of an object."""
  name: str

  @property
  def code(self) -> str:
    return f".{self.name}"

  def follow(self, container: Any) -> Any:
    return getattr(container, self.name)


class BuildableAttr(Attr):
  """An attribute of a Buildable."""


@dataclasses.dataclass(frozen=True)
class BuildableFnOrCls(Attr):
  """The callable (__fn_or_cls__) for a fdl.Buildable."""

  def __init__(self):
    super().__init__("__fn_or_cls__")


Path = Tuple[PathElement, ...]
Paths = Tuple[Path, ...]

_CONTAINER_TYPES = (List, Tuple, Dict, config.Buildable)
TraversableContainer = Union[_CONTAINER_TYPES]

# The following types are assembled based on the standard builtin types listed
# at https://docs.python.org/3/library/stdtypes.html, and the builtin constants
# listed at https://docs.python.org/3/library/constants.html.
_IMMUTABLE_NONCONTAINER_TYPES = (
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    type(None),
    type(NotImplemented),
    type(Ellipsis),
)


def path_str(path: Path) -> str:
  return "".join(x.code for x in path)


def follow_path(root: Any, path: Path):
  """Follows the path from a root item to a contained item, and returns it.

  Equivalent to `functools.reduce(lambda v, p: p.follow(v), root, path)`,
  but gives better error messages.

  Args:
    root: The starting point for the path.
    path: A sequence of `PathElement`s, indicating how to get from `root` to the
      contained item.

  Returns:
    The contained item identified by `path`.

  Raises:
    ValueError: If `path` is not compatible with `root`.
  """
  value = root
  for i, path_elt in enumerate(path):
    try:
      value = path_elt.follow(value)
    except (KeyError, IndexError, TypeError, AttributeError) as e:
      raise ValueError(f"{path_elt} is not compatible with "
                       f"root{path_str(path[:i])}={value!r}: {e}") from e
  return value


def _add_path_element(paths: Paths, elt: PathElement) -> Paths:
  return tuple(path + (elt,) for path in paths)


def is_traversable(value: Any) -> bool:
  """Determines what container types can be traversed into."""
  return isinstance(value, _CONTAINER_TYPES)


def is_memoizable(value: Any) -> bool:
  """Determines what values can be memoized.

  A primary concern is whether `value` may be subject to Python's interning
  optimizations, which could lead to confusing results under some circumstances
  if memoization is allowed. For the purposes of this function then, immutable
  types that can't contain references (including strings and ints) are excluded
  from memoization. Instances of such types are guaranteed to maintain an
  equality relationship with themselves over time.

  Args:
    value: A candidate value to check for memoizability.

  Returns:
    A bool indicating whether `value` is memoizable.
  """
  return (
      not isinstance(value, _IMMUTABLE_NONCONTAINER_TYPES) and
      value != ()  # pylint: disable=g-explicit-bool-comparison
  )  # pyformat: disable


def is_leaf(value: Any) -> bool:
  return not is_traversable(value)


def is_namedtuple(value: Any) -> bool:
  return (
      isinstance(value, tuple) and
      hasattr(value, "_asdict") and
      hasattr(value, "_fields") and
      all(isinstance(f, str) for f in value._fields)
  )  # pyformat: disable


def _container_like(container: TraversableContainer,
                    iterable: Iterable[Any]) -> TraversableContainer:
  """Returns a container like `container` but with values from `iterable`.

  If `container` is a `Dict`, a new `Dict` will be created, with keys from
  `container` and values from `iterable`. A best effort will be made to
  construct a new container of the same specific type as `container`, however
  the constructor is assumed to take a sequence of `(key, value)` pairs.

  If `container` is a `NamedTuple`, a new instance of the same `NamedTuple` type
  will be created, with field names from `container` and values from `iterable`.
  Note that this function is not compatible with `NamedTuple`s that have custom
  `__new__` methods with a different signature from the default `__new__`.

  If `container` is a `Buildable`, a new `Buildable` (of the same specific type
  as `container`) will be created, with arguments set based on argument names
  from `container.__arguments__` and values from `iterable`.

  Otherwise, if `container` is a list or tuple, a list or tuple will be
  constructed directly from `iterable`.

  Args:
    container: The container to replicate (but with values from `iterable`).
    iterable: An iterable that yields the new values to wrap in a container of
      the same type as `container`.

  Raises:
    ValueError: If `container` is an unsupported type.

  Returns:
    A new container of the same type as `container`, but with values taken
    from `iterable`.
  """
  # The rest of this function confuses pytype due to incorrect type narrowing...
  # disable type checking for `container` here by casting to Any.
  container = typing.cast(Any, container)
  # TODO: Extend to support a broader range of container types,
  # including at least defaultdict and maybe more general Mappings.
  if isinstance(container, Dict):
    return type(container)(zip(container, iterable))
  elif is_namedtuple(container):
    # If this namedtuple has a custom __new__ method that changes the signature
    # to have different numbers/types of arguments than the underlying
    # namedtuple fields, then the construction below will fail...
    return type(container)(*iterable)
  elif isinstance(container, config.Buildable):
    # TODO: Make sure argument history is handled gracefully.
    buildable = copy.copy(container)
    for name, value in zip(container.__arguments__, iterable):
      setattr(buildable, name, value)
    return buildable
  elif isinstance(container, (List, Tuple)):
    return type(container)(iterable)
  else:
    raise ValueError(f"Not a supported container type: {container}.")


def _yield_items(
    container: TraversableContainer) -> Iterable[Tuple[PathElement, Any]]:
  """Yields (key, value) pairs from `container`.

  The value component of the output of this function is intended to be
  compatible with `_container_like` above, in other words:

      values = [value for path_element, value in _yield_items(container)]
      container == _container_like(container, values)

  Args:
    container: A compatible Python container (List, Tuple, NamedTuple, Dict), or
      a `fdl.Buildable` instance.

  Raises:
    ValueError: If `container` is an unsupported type.

  Yields:
    The container's (key, value) pairs.
  """
  # The rest of this function confuses pytype due to incorrect type narrowing...
  # disable type checking for `container` here by casting to Any.
  container = typing.cast(Any, container)
  if isinstance(container, Dict):
    for key, value in container.items():
      yield Key(key), value
  elif is_namedtuple(container):
    for name, value in container._asdict().items():
      yield Attr(name), value
  elif isinstance(container, config.Buildable):
    for name, value in container.__arguments__.items():
      yield Attr(name), value
  elif isinstance(container, (List, Tuple)):
    for index, value in enumerate(container):
      yield Index(index), value
  else:
    raise ValueError(f"Not a supported container type: {container}.")


def _validate_traversal_fn(fn):
  """Raises an exception if fn is not a valid traversal function."""
  if not (inspect.isgeneratorfunction(fn) or
          (isinstance(fn, functools.partial) and
           inspect.isgeneratorfunction(fn.func)) or
          inspect.isgeneratorfunction(getattr(fn, "__call__", None))):
    raise ValueError("`fn` should contain a yield statement.")


TraverseWithPathFn = Callable[[Path, Any], Generator[None, Any, Any]]


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

      output = daglish.traverse_with_path(replace_twos_and_tuples, structure)

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
  _validate_traversal_fn(fn)

  def traverse_impl(path: Path, structure: Any) -> Any:
    """Recursive traversal implementation."""

    def subtree_fn(item: Tuple[Path, Any]):
      subtree_path, subtree = item
      return traverse_impl(path + (subtree_path,), subtree)

    def traverse_subtrees() -> Any:
      if is_traversable(structure):
        items = _yield_items(structure)
        return _container_like(structure, map(subtree_fn, items))
      else:
        return structure

    generator = fn(path, structure)
    try:
      if next(generator) is not None:
        raise RuntimeError("The traversal function yielded a non-None value.")
      generator.send(traverse_subtrees())
    except StopIteration as e:
      return e.value  # pytype: disable=attribute-error
    else:
      raise RuntimeError("Does the traversal function have two yields?")

  return traverse_impl((), structure)


def collect_paths_by_id(structure: Any,
                        memoizable_only: bool) -> Dict[int, List[Path]]:
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

  def collect_paths(path: Path, value: Any):
    if not memoizable_only or is_memoizable(value):
      paths_by_id.setdefault(id(value), []).append(path)
    yield

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

  def collect_value(path: Path, value: Any):
    del path  # Unused.
    if not memoizable_only or is_memoizable(value):
      value_by_id[id(value)] = value
    yield

  traverse_with_path(collect_value, structure)
  return value_by_id


def collect_value_by_path(structure: Any,
                          memoizable_only: bool) -> Dict[Path, Any]:
  """Returns a dict mapping `path->v` for all `v` traversable from `structure`.

  Args:
    structure: The structure for which the `path->value` map should be created.
    memoizable_only: If true, then only include values `v` for which
      `is_memoizable(v)` is true.
  """
  value_by_path = {}

  def collect_value(path: Path, value: Any):
    if not memoizable_only or is_memoizable(value):
      value_by_path[path] = value
    yield

  traverse_with_path(collect_value, structure)
  return value_by_path


TraverseWithAllPathsFn = Callable[[Paths, Path, Any], Generator[None, Any, Any]]


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
  _validate_traversal_fn(fn)

  paths_memo = collect_paths_by_id(structure, memoizable_only=True)

  def wrap_with_paths(current_path: Path, value: Any):
    all_paths = ((),)
    if is_memoizable(value):
      all_paths = paths_memo[id(value)]
    elif current_path:
      # Non-memoizable values may have still have multiple paths via a shared
      # parent container. Here we augment all paths to the parent container.
      parent = follow_path(structure, current_path[:-1])
      parent_paths = paths_memo[id(parent)]
      all_paths = tuple(path + current_path[-1:] for path in parent_paths)
    return (yield from fn(all_paths, current_path, value))

  return traverse_with_path(wrap_with_paths, structure)


MemoizedTraverseFn = Callable[[Paths, Any], Generator[None, Any, Any]]


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
  _validate_traversal_fn(fn)

  memo = {}

  def wrap_with_memo(all_paths: Paths, current_path: Path, value: Any):
    del current_path
    key = id(value)
    if key in memo:
      return memo[key]
    else:
      output = yield from fn(all_paths, value)
      if is_memoizable(value):
        memo[key] = output
      return output

  return traverse_with_all_paths(wrap_with_memo, structure)
