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
import collections
import dataclasses
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union


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


class NamedTupleType:
  pass


# This has the same type as Path, but different semantic meaning.
PathElements = Tuple[PathElement, ...]
PathElementsFn = Callable[[Any], PathElements]
FlattenFn = Callable[[Any], Tuple[Tuple[Any, ...], Any]]
UnflattenFn = Callable[[Iterable[Any], Any], Any]

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class NodeTraverser:
  """Contains information required to traverse a given node type."""
  flatten: FlattenFn
  unflatten: UnflattenFn
  path_elements: PathElementsFn


class NodeTraverserRegistry:
  """A registry of `NodeTraverser`s."""

  def __init__(self):
    self._node_traversers: Dict[Type[Any], NodeTraverser] = {}

  def register_node_traverser(
      self,
      node_type: Type[Any],
      flatten_fn: FlattenFn,
      unflatten_fn: UnflattenFn,
      path_elements_fn: PathElementsFn,
  ) -> None:
    """Registers a node traverser for `node_type` in the default registry.

    Args:
      node_type: The node type to regiser a traverser for. The traverser will be
        used *only* for nodes of this type, not subclasses (with the exception
        of the special-cased `daglish.NamedTupleType`).
      flatten_fn: A function that flattens values for traversal. This should
        accept an instance of `node_type`, and return a tuple of `(values,
        metadata)`, where `values` is a sequence of values and `metadata` is
        arbitrary traverser-specific data.
      unflatten_fn: A function that unflattens values, which should accept
        `values` and `metadata` and return a new instance of `node_type`.
      path_elements_fn: A function that returns `PathElement` instances for the
        flattened values returned by `flatten_fn`. This should accept an
        instance of `node_type`, and return a sequence of `PathElement`s aligned
        with the values returned by `flatten_fn`.
    """
    if not isinstance(node_type, type):
      raise TypeError(f"`node_type` ({node_type}) must be a type.")
    if node_type in self._node_traversers:
      raise ValueError(
          f"A node traverser for {node_type} has already been registered.")
    self._node_traversers[node_type] = NodeTraverser(
        flatten=flatten_fn,
        unflatten=unflatten_fn,
        path_elements=path_elements_fn,
    )

  def find_node_traverser(
      self,
      node_type: Type[Any],
  ) -> Optional[NodeTraverser]:
    """Finds a `NodeTraverser` for the given `node_type`.

    This simply looks up `node_type`, with one special case: if `node_type` is a
    `NamedTuple` (as determined by the `is_namedtuple_subclass` function), then
    `daglish.NamedTupleType` is looked up in `registry` instead.

    Args:
      node_type: The node type to find a traverser for.

    Returns:
      A `NodeTraverser` instance for `node_type`, if it exists, else `None`.
    """
    if not isinstance(node_type, type):
      raise TypeError(f"`node_type` ({node_type}) must be a type.")
    if is_namedtuple_subclass(node_type):
      node_type = NamedTupleType
    return self._node_traversers.get(node_type)

  def map_children(self, fn: Callable[[Any], Any], value: Any) -> Any:
    """Maps `fn` over the immediate children of `value`.

    Args:
      fn: A single-argument callable to apply to the children of `value`.
      value: The value to map `fn` over.

    Returns:
      A value of the same type as `value`, but with each child replaced by the
      return value of `fn(child)`.

    Raises:
      ValueError: If `value` is not traversable.
    """
    traverser = self.find_node_traverser(type(value))
    if traverser is None:
      raise ValueError(f"{value} is not traversable.")
    values, metadata = traverser.flatten(value)
    return traverser.unflatten((fn(value) for value in values), metadata)

  def is_traversable_type(self, node_type: Type[Any]) -> bool:
    """Returns whether `node_type` can be traversed."""
    return self.find_node_traverser(node_type) is not None


# The default registry of node traversers.
_default_traverser_registry = NodeTraverserRegistry()

# Forward functions from the module level to the default registry.
register_node_traverser = _default_traverser_registry.register_node_traverser
find_node_traverser = _default_traverser_registry.find_node_traverser
map_children = _default_traverser_registry.map_children
is_traversable_type = _default_traverser_registry.is_traversable_type

register_node_traverser(
    dict,
    flatten_fn=lambda x: (tuple(x.values()), tuple(x.keys())),
    unflatten_fn=lambda values, keys: dict(zip(keys, values)),
    path_elements_fn=lambda x: [Key(key) for key in x.keys()])


def flatten_defaultdict(node):
  return tuple(node.values()), (node.default_factory, tuple(node.keys()))


def unflatten_defaultdict(values, metadata):
  default_factory, keys = metadata
  return collections.defaultdict(default_factory, zip(keys, values))


register_node_traverser(
    collections.defaultdict,
    flatten_fn=flatten_defaultdict,
    unflatten_fn=unflatten_defaultdict,
    path_elements_fn=lambda x: tuple(Key(key) for key in x.keys()))

register_node_traverser(
    tuple,
    flatten_fn=lambda x: (x, None),
    unflatten_fn=lambda x, _: tuple(x),
    path_elements_fn=lambda x: tuple(Index(i) for i in range(len(x))))

register_node_traverser(
    NamedTupleType,
    flatten_fn=lambda x: (tuple(x), type(x)),
    unflatten_fn=lambda values, node_type: node_type(*values),
    path_elements_fn=lambda x: tuple(Attr(name) for name in x._asdict().keys()))

register_node_traverser(
    list,
    flatten_fn=lambda x: (tuple(x), None),
    unflatten_fn=lambda x, _: list(x),
    path_elements_fn=lambda x: tuple(Index(i) for i in range(len(x))))


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


def add_path_element(paths: Iterable[Path], element: PathElement) -> Paths:
  return tuple(path + (element,) for path in paths)


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


def is_namedtuple_subclass(type_: Type[Any]) -> bool:
  return (
      issubclass(type_, tuple) and
      hasattr(type_, "_asdict") and
      hasattr(type_, "_fields") and
      all(isinstance(f, str) for f in type_._fields)
  )  # pyformat: disable
