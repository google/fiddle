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

from __future__ import annotations

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

  def is_traversable_type(self, node_type: Type[Any]) -> bool:
    """Returns whether `node_type` can be traversed."""
    return self.find_node_traverser(node_type) is not None


# The default registry of node traversers.
_default_traverser_registry = NodeTraverserRegistry()

# Forward functions from the module level to the default registry.
register_node_traverser = _default_traverser_registry.register_node_traverser
find_node_traverser = _default_traverser_registry.find_node_traverser
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


def is_prefix(prefix_path: Path, containing_path: Path):
  """Returns `True` if `prefix_path` is a prefix of `containing_path`.

  Args:
    prefix_path: the `Path` that may be a prefix of `containing_path`.
    containing_path: the `Path` that may be prefixed by `prefix_path`.

  Returns:
    `True` if `prefix_path` is a prefix of `containing_path`.
  """
  return prefix_path == containing_path[:len(prefix_path)]


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


def is_internable(value: Any) -> bool:
  """Returns true if Python can apply an interning optimization to `value`.

  If this is false, then `x is y` is only true if they point to the same object,
  created at the same place.

  If this is true, then `x is y` may be true for unrelated but equal values
  (i.e., values that were created at different places).

  The most common examples of values that the interning optimization can
  apply to are constants, such as booleans, strings, and small integers.

  The interning optimization may be applied to (nested) tuples whose
  values are constants.

  Args:
    value: any value, it can be a Fiddle buildable or a regular Python value.
  """
  return not is_memoizable(value) or (
      # pylint: disable-next=unidiomatic-typecheck
      # We want tuples only and not things like NamedTuples which are not
      # interned by Python.
      type(value) is tuple and all(is_internable(e) for e in value))


def is_namedtuple_subclass(type_: Type[Any]) -> bool:
  return (
      issubclass(type_, tuple) and
      hasattr(type_, "_asdict") and
      hasattr(type_, "_fields") and
      all(isinstance(f, str) for f in type_._fields)
  )  # pyformat: disable


@dataclasses.dataclass
class Traversal(metaclass=abc.ABCMeta):
  """Defines an API that traversers must implement.

  Please note that users of a traverser will mostly interact with the State
  API, which bundles a Traverser with the current paths in the traversal.
  """

  traversal_fn: Callable[..., Any]
  root_obj: Any
  registry: NodeTraverserRegistry = _default_traverser_registry

  def find_node_traverser(self,
                          node_type: Type[Any]) -> Optional[NodeTraverser]:
    """Uses the configured registry to find a NodeTraverser."""
    return self.registry.find_node_traverser(node_type)

  @abc.abstractmethod
  def apply(self, value: Any, state: State) -> Any:
    """Calls the underlying function bound to the traversal."""
    raise NotImplementedError()

  @abc.abstractmethod
  def all_paths_to_object(self, object_id: int,
                          allow_caching: bool) -> List[Path]:
    """Returns all paths to a container."""
    raise NotImplementedError()

  def initial_state(self) -> State:
    """Returns an initial state for this traversal.

    The initial state has a reference to the traversal, the empty path, and no
    parent states.
    """
    return State(self, (), self.root_obj, _parent=None)  # pytype: disable=attribute-error

  @classmethod
  def begin(cls, fn: Callable[..., Any], root_obj: Any) -> State:
    """Creates a new traversal and returns the initial state.

    Args:
      fn: Function which is applied at each node during the traversal.
      root_obj: Root object being traversed.

    Returns:
      The initial state (from `initial_state`) of a new traversal instance.
    """
    return cls(traversal_fn=fn, root_obj=root_obj).initial_state()  # pytype: disable=not-instantiable

  @classmethod
  def run(cls, fn: Callable[..., Any], root_obj: Any) -> Any:
    """Creates a traversal and state, and then calls/returns `fn` on this."""
    state = cls.begin(fn=fn, root_obj=root_obj)
    return fn(root_obj, state)


_T = TypeVar("_T")


@dataclasses.dataclass(frozen=True)
class SubTraversalResult:
  __slots__ = ("node_traverser", "values", "metadata", "path_elements")
  node_traverser: NodeTraverser
  values: List[Any]
  metadata: Any
  path_elements: PathElements

  def unflatten(self):
    return self.node_traverser.unflatten(self.values, self.metadata)


@dataclasses.dataclass(frozen=True)
class State:
  """Contains a current traversal state.

  Attributes:
    traversal: Reference to main traversal object.
    current_path: A path that can be followed to the current object. In the case
      of shared objects, there will be other paths to the current object, and
      often these are determined by a somewhat arbitrary DAG traversal order.
  """
  __slots__ = ("traversal", "current_path", "_value", "_parent")

  traversal: Traversal
  current_path: Path

  # Implementation note: Please don't use _value outside of the @property
  # accessors.
  _value: Any  # pylint: disable=invalid-name

  _parent: Optional[State]

  @property
  def _object_id(self) -> int:
    return id(self._value)

  @property
  def _is_memoizable(self) -> bool:
    return is_memoizable(self._value)

  @property
  def ancestors_inclusive(self) -> Iterable[State]:
    """Gets ancestors, including the current state."""
    yield self
    if self._parent is not None:
      yield from self._parent.ancestors_inclusive

  def get_all_paths(self, allow_caching: bool = True) -> List[Path]:
    """Gets all paths to the current value.

    Args:
      allow_caching: Whether paths can be cached. Most traverers will compute
        paths to all objects from the root config object. But this can fail to
        reflect objects that are being added by the traversal, so occasionally
        one may want to take a big performance hit to reflect any newly added
        objects.

    Returns:
      List of paths.
    """
    base_paths: List[Path] = [()]
    suffix_len = 0  # number of elements from `self.current_path` to append
    # pylint: disable=protected-access
    for state in self.ancestors_inclusive:
      if state._is_memoizable:
        base_paths = self.traversal.all_paths_to_object(
            state._object_id, allow_caching=allow_caching)
        break
      else:
        suffix_len += 1
    # pylint: enable=protected-access
    path_suffix = self.current_path[-suffix_len:] if suffix_len > 0 else ()
    return [path + path_suffix for path in base_paths]

  def is_traversable(self, value: Any) -> bool:
    """Returns whether a value is traversable."""
    return self.traversal.find_node_traverser(type(value)) is not None

  def _flattened_map_children(
      self, value: Any, node_traverser: NodeTraverser) -> SubTraversalResult:
    """Shared function for map_children / flattened_map_children."""
    subvalues, metadata = node_traverser.flatten(value)
    path_elements = node_traverser.path_elements(value)
    new_subvalues = [
        self.call(subvalue, path_element)
        for subvalue, path_element in zip(subvalues, path_elements)
    ]
    return SubTraversalResult(
        node_traverser=node_traverser,
        values=new_subvalues,
        metadata=metadata,
        path_elements=path_elements)

  def map_children(self, value: _T) -> _T:
    """Maps over children for traversable values, otherwise returns it.

    Args:
      value: Value to map over. Non-traversable values are returned unmodified.

    Returns:
      A mapped value of the same type.
    """
    node_traverser = self.traversal.find_node_traverser(type(value))
    if node_traverser is None:
      return value
    result = self._flattened_map_children(value, node_traverser)
    return node_traverser.unflatten(result.values, result.metadata)

  def flattened_map_children(self, value: Any) -> SubTraversalResult:
    """Maps over children for traversable values, but doesn't unflatten results.

    Unlike `map_children`, we've found that there's not an easy natural choice
    for handling the case where `value` is not a traversable type. Generally
    calling code expects a SubTraversalResult, which contains an iterable of
    values and metadata. The transformation of these values and choice of
    metadata is non-trivial, and client code would probably have to branch on it
    anyway. Therefore, we throw an error for non-traversable values; please test
    whether `value` is traversable beforehand.

    Args:
      value: Value to map over.

    Returns:
      Sub-traversal results.

    Raises:
      ValueError: If `value` is not traversable. Please test beforehand by
      calling `state.is_traversable()`.
    """
    node_traverser = self.traversal.find_node_traverser(type(value))
    if node_traverser is None:
      raise ValueError("Please handle non-traversable values yourself.")
    return self._flattened_map_children(value, node_traverser)

  def call(self, value, *additional_path: PathElement):
    """Low-level function to execute a sub-traversal.

    This creates a new state, and then applies the function bound to the
    traverser.

    Args:
      value: Sub-value to run the traversal on.
      *additional_path: Additional path elements relating the sub-value to the
        value referred to by `self`.

    Returns:
      Result of sub-traversal (whatever `self.traversal.traversal_fn` returns).
    """
    new_state = State(self.traversal, (*self.current_path, *additional_path),
                      value, self)
    return self.traversal.apply(value, new_state)


@dataclasses.dataclass
class BasicTraversal(Traversal):
  """Basic traversal.

  This traversal will go through shared objects multiple times, for each path
  that they are reachable from.
  """

  paths_cache: Dict[int, List[Path]] = dataclasses.field(default_factory=dict)

  def apply(self, value, state):
    return self.traversal_fn(value, state)

  def all_paths_to_object(self, object_id: int,
                          allow_caching: bool) -> List[Path]:
    if allow_caching and object_id in self.paths_cache:
      return self.paths_cache[object_id]
    else:
      all_paths = self.paths_cache = collect_paths_by_id(
          self.root_obj, memoizable_only=True)
      return all_paths[object_id]


@dataclasses.dataclass
class MemoizedTraversal(BasicTraversal):
  """Traversal that memoizes results."""

  memoize_internables: bool = True
  memo: Dict[int, Any] = dataclasses.field(default_factory=dict)

  @classmethod
  def begin(cls,
            fn: Callable[..., Any],
            root_obj: Any,
            memoize_internables: bool = True) -> State:
    """Creates a new traversal and returns the initial state.

    Args:
      fn: Function which is applied at each node during the traversal.
      root_obj: Root object being traversed.
      memoize_internables: `True` if internables should be memoized. Something
        is internable if it's declaration in separate code locations still cause
        references to it to use the same instance when the value is the same
        (for example, primitives are internable). This is a result of Python's
        internal optimization. See also `daglish.is_internable`.

    Returns:
      The initial state (from `initial_state`) of a new traversal instance.
    """
    return cls(
        traversal_fn=fn,
        root_obj=root_obj,
        memoize_internables=memoize_internables).initial_state()

  def apply(self, value, state):
    if not self.memoize_internables and is_internable(value):
      return self.traversal_fn(value, state)
    elif id(value) in self.memo:
      return self.memo[id(value)]
    else:
      result = self.memo[id(value)] = self.traversal_fn(value, state)
      return result


def collect_paths_by_id(structure,
                        memoizable_only=False) -> Dict[int, List[Path]]:
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

  def traverse(value, state: State):
    if not memoizable_only or is_memoizable(value):
      paths_by_id.setdefault(id(value), []).append(state.current_path)
    if state.is_traversable(value):
      state.flattened_map_children(value)

  BasicTraversal.run(traverse, structure)
  return paths_by_id
