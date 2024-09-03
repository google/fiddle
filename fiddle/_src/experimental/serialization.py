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

"""Functions to help serialize configuration.

This module provides functions that help ensure pickle-compatibility
(`clear_argument_history`), as well as functions and associated helper classes
used to serialize Fiddle structures into a custom JSON-based representation
(`dump_json`, `load_json`).
"""

import abc
import collections
import copy
import dataclasses
import enum
import functools
import importlib
import inspect
import itertools
import json
import operator
import re
import sys
import types
from typing import Any, Dict, Iterable, List, Optional, Type

from fiddle import daglish
from fiddle._src import config as config_lib
from fiddle._src import reraised_exception
from fiddle._src import special_overrides
from fiddle._src.experimental import daglish_legacy
from fiddle._src.experimental import lazy_imports


_VERSION = '0.0.1'


def clear_argument_history(buildable: config_lib.Buildable):
  """Creates a copy of a buildable, clearing its history.

  This can be useful when a Config's history contains a non-picklable value.

  Args:
    buildable: A Fiddle configuration DAG whose history will be removed.

  Returns:
    A copy of `buildable` with `__argument_history__` set to empty.
  """

  def traverse(value: Any, state: daglish.State) -> Any:
    if state.is_traversable(value):
      if isinstance(value, config_lib.Buildable):
        sub_results = state.flattened_map_children(value)
        metadata = sub_results.metadata.without_history()
        return sub_results.node_traverser.unflatten(sub_results.values,
                                                    metadata)
      return state.map_children(value)
    else:
      return copy.deepcopy(value)

  return daglish.MemoizedTraversal.run(traverse, buildable)


@dataclasses.dataclass(frozen=True)
class UnreachableElement(daglish.PathElement):
  """Represents an unreachable element.

  These are added during the serialization traversal below to provide more
  informative error messages.
  """

  def follow(self, container: Any):
    raise NotImplementedError("Unreachable elements can't be followed.")


@dataclasses.dataclass(frozen=True)
class SetElement(UnreachableElement):
  """Represents an element of a set."""

  @property
  def code(self):
    return '[<set element>]'


@dataclasses.dataclass(frozen=True)
class MetadataElement(UnreachableElement):
  """Represents traversal into a value's metadata."""

  @property
  def code(self):
    return '.<metadata>'


@dataclasses.dataclass(frozen=True)
class IdentityElement(daglish.PathElement):
  """Represents a path element that is its container."""

  @property
  def code(self):
    return ''

  def follow(self, container: Any):
    return container


# A serialization-specific traverser registry. This can be used to register
# traversal for types that should be serializable but shouldn't be traversed
# during other Fiddle operations (e.g., `fdl.build()`).
_traverser_registry = daglish.NodeTraverserRegistry(use_fallback=True)
register_node_traverser = _traverser_registry.register_node_traverser
find_node_traverser = _traverser_registry.find_node_traverser


def _flatten_standard_object(instance):
  keys = tuple(instance.__dict__.keys())
  return instance.__dict__.values(), (type(instance), keys)


def _unflatten_standard_object(values, metadata):
  object_type, dict_keys = metadata
  instance = object.__new__(object_type)
  instance.__dict__.update(zip(dict_keys, values))
  return instance


def register_dict_based_object(object_type: Type[Any]):
  """Registers serialization support for the given (dict-based) `object_type`.

  This adds serialization support for "dict-based" Python objects, where here
  dict-based means that we can simply serialize the `object_type.__dict__`
  attribute, and on deserialization create a new empty instance of `object_type`
  and restore its `__dict__` contents. If this behavior is insufficient or
  incorrect for a given type, more explicit handling using
  `register_node_traverser` will be required to support serialization.

  Args:
    object_type: The type to register serialization support for.
  """
  register_node_traverser(
      object_type,
      flatten_fn=_flatten_standard_object,
      unflatten_fn=_unflatten_standard_object,
      path_elements_fn=lambda x: tuple(daglish.Attr(k) for k in x.__dict__))


for set_type in (set, frozenset):
  register_node_traverser(
      set_type,
      flatten_fn=lambda x: (tuple(x), None),
      unflatten_fn=lambda values, _, set_type=set_type: set_type(values),
      path_elements_fn=lambda x: tuple(SetElement() for _ in x))

register_node_traverser(
    bytes,
    flatten_fn=lambda x: ((x.decode('raw_unicode_escape'),), None),
    unflatten_fn=lambda values, _: values[0].encode('raw_unicode_escape'),
    path_elements_fn=lambda x: (IdentityElement(),),
)

register_node_traverser(
    config_lib.NoValue,
    flatten_fn=lambda _: ((), None),
    unflatten_fn=lambda values, metadata: config_lib.NO_VALUE,
    path_elements_fn=lambda _: ())


def _is_leaf_type(value_type):
  """Returns whether `value_type` is a JSON-representable primitive type."""
  return value_type in (int, float, bool, str, type(None))


def path_str(path: daglish.Path) -> str:
  """Returns a path string for `path`."""
  return '<root>' + daglish.path_str(path)


def path_strs(paths: Iterable[daglish.Path]) -> List[str]:
  """Returns a list of path strings corresponding to `paths`."""
  return [path_str(path) for path in paths]


class UnserializableValueError(Exception):
  """An exception thrown when an unserializable value is encountered."""


class PyrefPolicy(metaclass=abc.ABCMeta):
  """Represents a policy for importing references to Python objects.

  A `PyrefPolicy` provides control over what kind of Python symbols can be
  loaded during deserialization.
  """

  @abc.abstractmethod
  def allows_import(self, module: str, symbol: str) -> bool:
    """Returns whether this policy allows importing `symbol` from `module`.

    This will be called by `import_symbol` before any imports take place. If
    this returns `True`, `symbol` will be imported from `module`.

    Args:
      module: The module to import `symbol` from.
      symbol: The symbol to import from `module`.
    """

  def allows_value(self, value: Any) -> bool:
    """Returns whether this policy allows an imported `value` to be used.

    This is called after `value` has already been imported, but before it is
    returned by `import_symbol`.

    Args:
      value: An imported value.
    """


class PyrefPolicyError(Exception):
  """An error arising when an import is disallowed by a `PyrefPolicy`."""

  PRE_IMPORT = object()

  def __init__(self, policy: PyrefPolicy, module: str, symbol: str, value: Any):
    self.module = module
    self.symbol = symbol
    self.value = value

    if value is not PyrefPolicyError.PRE_IMPORT:
      value_str = f' (with value {value!r})'
    else:
      value_str = ''
    msg = (
        f'Importing {symbol} from {module}{value_str} is not permitted by the '
        f'active Python reference policy ({policy}).')
    super().__init__(msg)


class DefaultPyrefPolicy(PyrefPolicy):
  """Provides a default policy for loading pyref values.

  This policy excludes builtins that are not a supported serializable type
  (e.g., `list`, `tuple`, `dict`, etc are allowed). This policy is subject to
  change and may become stricter in the future.
  """

  def allows_import(self, module, symbol):
    return True

  def allows_value(self, value) -> bool:
    is_serializable_type = (
        isinstance(value, type) and find_node_traverser(value) is not None)
    module_name = getattr(inspect.getmodule(value), '__name__', None)
    is_builtin = module_name in sys.builtin_module_names
    return is_serializable_type or not is_builtin


def _fiddle_pyref_context(module: str, symbol: str) -> str:
  return (
      f'\nFiddle context: Error occurred while importing pyref to {symbol!r} '
      f'from {module!r}.')


def import_symbol(policy: PyrefPolicy, module: str, symbol: str):
  """Returns the value obtained from importing `symbol` from `module`.

  Args:
    policy: The `PyrefPolicy` governing which symbols can be imported.
    module: The module to import `symbol` from.
    symbol: The symbol to import from `module`.

  Raises:
    ModuleNotFoundError: If `module` can't be found.
    AttributeError: If `symbol` can't be accessed on `module`.
    PyrefPolicyError: If importing `symbol` from `module` is disallowed by
      this `PyrefPolicy`.
  """
  value = PyrefPolicyError.PRE_IMPORT
  if policy.allows_import(module, symbol):
    module = special_overrides.maybe_get_module_override_for_migrated_serialization_symbol(
        module, symbol
    )
    make_message = functools.partial(_fiddle_pyref_context, module, symbol)
    with reraised_exception.try_with_lazy_message(make_message):
      value = importlib.import_module(module)
      for attr_name in symbol.split('.'):
        value = getattr(value, attr_name)

    if policy.allows_value(value):
      return value

  raise PyrefPolicyError(policy, module, symbol, value)


def _to_snake_case(name: str) -> str:
  """Converts a camel or studly-caps name to a snake_case name."""
  return re.sub(r'(?<=[^_A-Z])(?=[A-Z])', '_', name).lower()


@dataclasses.dataclass
class SerializationConstant:
  module: str
  symbol: str

  def to_pyref(self):
    return {
        _TYPE_KEY: _PYREF_TYPE,
        _MODULE_KEY: self.module,
        _NAME_KEY: self.symbol
    }


_serialization_constants_by_id: Dict[int, SerializationConstant] = {}
_serialization_constants_by_value: Dict[Any, SerializationConstant] = {}


def register_constant(module: str, symbol: str, compare_by_identity: bool):
  """Registers a module-level constant value for serialization.

  The specified value will be serialized as a pyref (python reference), meaning
  that its value will be deserialized by reading the specified symbol from the
  specified module.

  Args:
    module: The name of the module containing the constant symbol.
    symbol: The symbol in `module` that contains a constant value.
    compare_by_identity: If True, then only use a pyref to serialize a value `x`
      if `x is <module>.<symbol>`.  If False, then use a pyref to serialize any
      value `x` where `x == <module>.<symbol>`.  If `compare_by_identity` is
      False, then the value must be hashable.

  Raises:
    ModuleNotFoundError: If `module` can't be found.
    AttributeError If 'symbol' can't be accessed on `module`.
    PyrefPolicyError: If importing `symbol` from `module` is disallowed by the
      `DefaultPyrefPolicy`.
    ValueError: If registration is unnecessary for `<module>.<symbol>` (e.g.,
      because it is a `type` or `function`).
  """
  value = import_symbol(DefaultPyrefPolicy(), module, symbol)

  # Perform some sanity checks.
  if isinstance(value, (type, types.FunctionType)):
    raise ValueError(
        f'Can not register {module}.{symbol} as a serialization constant '
        f'because it is a type or function (so registration is unnecessary).')
  if _is_leaf_type(type(value)):
    raise ValueError(
        f'Can not register {module}.{symbol} as a serialization constant '
        f'because it is a JSON-representable primitive type (so registration '
        'is unnecessary).')
  if find_node_traverser(type(value)) is not None:
    raise ValueError(
        f'Can not register {module}.{symbol} as a serialization constant '
        f'because it is a daglish-traversable type (so registration '
        'is unnecessary).')

  serialization_constant = SerializationConstant(module, symbol)
  if compare_by_identity:
    _serialization_constants_by_id[id(value)] = serialization_constant
  else:
    _serialization_constants_by_value[value] = serialization_constant


# TODO(b/273321868): Remove this soon.
def register_enum(enum_type: Type[enum.Enum]) -> None:
  """Registers all members of the given enum using `register_constant`.

  Args:
    enum_type: Any class that inherits from `enum.Enum`.
  """
  for name in enum_type.__members__:
    register_constant(
        enum_type.__module__,
        f'{enum_type.__qualname__}.{name}',
        compare_by_identity=True,
    )

# The following constants define the keys (and some values) in the
# JSON-serializable object representation produced as an intermediate step in
# the overall JSON serialization.

# Keys present in "root" dictionary.
_ROOT_KEY = 'root'
_OBJECTS_KEY = 'objects'
_REFCOUNTS_KEY = 'refcounts'
_VERSION_KEY = 'version'

# Keys used for non-primitive serialized values.
_TYPE_KEY = 'type'
_PATHS_KEY = 'paths'  # Only present for non-metadata values.

# Keys present for serialized traversable and custom objects.
_ITEMS_KEY = 'items'
_METADATA_KEY = 'metadata'

# Type identifier and keys used for "leaf" values.
_LEAF_TYPE = 'leaf'
_VALUE_KEY = 'value'

# Type identifier and keys used for "reference" values.
_REF_TYPE = 'ref'
_KEY_KEY = 'key'

# Type identifier and keys used for Python reference values.
_PYREF_TYPE = 'pyref'
_MODULE_KEY = 'module'
_NAME_KEY = 'name'


def _serialize_lazy_imports(proxy_object):
  """Serialize proxy object for lazy imports."""
  qualname = proxy_object.__qualname__
  parts = qualname.split(':')
  if len(parts) == 2:
    module_name, symbol_name = parts
  elif len(parts) == 1:
    module_name, symbol_name = qualname.rsplit('.', maxsplit=1)
  else:
    raise ValueError(f'Invalid qualname {qualname} for ProxyObject.')
  output = {
      _TYPE_KEY: _PYREF_TYPE,
      _MODULE_KEY: module_name,
      _NAME_KEY: symbol_name,
  }
  return output


class Serialization:
  """Maintains state for an in-progress serialization.

  This class is used to recursively assemble a JSON-serializable dictionary
  representation of a given value. This class manages the state maintained
  during this process. The resulting dictionary representation can be obtained
  from the result property once an instance has been created:

      dict_representation = Serialization(value, pyref_policy).result

  A new instance of the class should be created per serialization.

  The resulting dictionary representation has the following keys at the
  top-level:

    * 'root': A dictionary representation of the root object.
    * 'objects': A dictionary mapping object names to their dictionary
        representations.
    * 'refcounts': A dictionary mapping object names to the number of times they
        are referenced (e.g., via other objects).
    * 'version': A version string, indicating what version of the serialization
        format produced this result.

  Individual object representations (as contained in 'root' and 'objects') may
  either be a JSON primitive value or a dictionary representation. Dictionary
  representations will always have a 'type' key, and additional fields which
  depend on this type:

    * Traversable structures (those that have either a serialization-specific
      or daglish traverser available) will have their type information,
      flattened items, and traversal metadata recursively serialized into a
      dictionary with keys 'type', 'items', and 'metadata'. The items will be
      a list of lists, where each sublist contains both a string representation
      of the item's `daglish.PathElement` (only used for debugging/visualization
      purposes) as well as the serialized item itself.
    * Python types and functions are serialized with a special 'pyref' type
      value, along with fields 'module' and 'name' that are used when importing
      the type or function during deserialization.
    * References to other objects have a special 'ref' type, and a 'key' field
      that identifies an object in the top-level 'objects' dictionary.
    * Leaf values that need path information will have a 'leaf' type, with an
      additional 'value' field that is always a JSON primitive value.

  Any object's dictionary representation may also have a 'paths' field that
  stores a list of all paths to that object from the root.
  """

  def __init__(
      self,
      value: Any,
      pyref_policy: Optional[PyrefPolicy] = None,
  ):
    """Initializes the serialization.

    Args:
      value: The value to serialize.
      pyref_policy: The `PyrefPolicy` instance that should govern whether a
        given Python reference is permissible. Providing a policy here can help
        catch potential errors early, however this will only guarantee
        successful deserialization if the same policy is passed when creating a
        `Deserialization` instance.

    Raises:
      UnserializableValueError: If an unserializable value is encountered.
      ModuleNotFoundError: If a module can't be imported in the course of
        creating a Python reference.
      AttributeError: If an attribute or symbol can't be accessed in the course
        of creating a Python reference.
      PyrefPolicyError: If a Python reference (e.g., function or type) is
        disallowed by `pyref_policy`.
    """
    # Maps object ids to their key in the _objects map.
    self._memo: Dict[int, str] = {}
    # Maps object key to the object's serialized representation.
    self._objects: Dict[str, Any] = {}
    # Maintains references to (unserialized) referenced values. These are only
    # retained to avoid id reuse when creating temporary metadata objects.
    self._ref_values: List[Any] = []
    # Counts the number of times each object in _objects is referenced.
    self._refcounts = collections.defaultdict(int)
    # The root value being serialized.
    self._root = value
    # Maps (memoizable) object ids to all paths from `value` that reach them.
    self._paths_by_id = daglish_legacy.collect_paths_by_id(
        value, memoizable_only=True)
    # The active PyrefPolicy.
    self._pyref_policy = pyref_policy or DefaultPyrefPolicy()
    # The result of the serialization.
    self._result = {
        _ROOT_KEY: self._serialize(self._root, (), all_paths=((),)),
        _OBJECTS_KEY: self._objects,
        _REFCOUNTS_KEY: self._refcounts,
        _VERSION_KEY: _VERSION,
    }

  @property
  def result(self) -> Dict[str, Any]:
    """Returns the serialization result (JSON-compatible dictionary)."""
    return self._result

  def _unique_name(self, value: Any) -> str:
    """Creates a unique and informative name for `value`."""
    try:
      hint_type = type(value)
      if isinstance(value, config_lib.Buildable):
        hint_type = config_lib.get_callable(value)
      if isinstance(hint_type, lazy_imports.ProxyObject):
        hint = hint_type.name
      else:
        hint = hint_type.__name__
    except AttributeError:
      hint = re.sub(r"[<>()\[\] '\"]", '', str(type(value)))

    hint = _to_snake_case(hint)
    for i in itertools.count(1):
      name = f'{hint}_{i}'
      if name not in self._objects:
        break

    return name

  def _ref(self, key):
    """Creates a reference to another serialized object."""
    self._refcounts[key] += 1
    return {_TYPE_KEY: _REF_TYPE, _KEY_KEY: key}

  def _pyref(self, value, current_path: daglish.Path):
    """Creates a reference to an importable Python value."""
    module = inspect.getmodule(value)
    if module is None:
      msg = (
          f"Couldn't find the module for {value!r}; Are you wrapping the"
          ' function or using a test mock?'
      )
      raise UnserializableValueError(msg)
    module_name = module.__name__
    if isinstance(value, enum.Enum):
      symbol = value.__class__.__qualname__ + '.' + value.name
    else:
      symbol = value.__qualname__
    # Check to make sure we can import the symbol.
    imported_value = import_symbol(self._pyref_policy, module_name, symbol)
    # Class methods compare equal but do not preserve identity when imported.
    matches = (
        operator.eq if isinstance(value, types.MethodType) else operator.is_
    )
    if not matches(value, imported_value):
      msg = (
          f"Couldn't create a pyref for {value!r}; the same value was not"
          f' obtained by importing {symbol!r} from {module_name!r}. Error'
          f' occurred at path {path_str(current_path)!r}.'
      )
      raise UnserializableValueError(msg)
    return {
        _TYPE_KEY: _PYREF_TYPE,
        _MODULE_KEY: module_name,
        _NAME_KEY: symbol,
    }

  def _leaf(self, value):
    """Creates a leaf value."""
    return {
        _TYPE_KEY: _LEAF_TYPE,
        _VALUE_KEY: value,
    }

  def _serialize(
      self,
      value: Any,
      current_path: daglish.Path,
      all_paths: Optional[daglish.Paths] = None,
  ) -> Dict[str, Any]:
    """Returns a JSON-serializable dict representation of `value`.

    This function traverses `value`, recursively building up a JSON-serializable
    dictionary capturing values and associated metadata, debugging, and
    visualization info.

    Args:
      value: The value to serialize.
      current_path: The traversal's current path.
      all_paths: An optional list of all paths that reach `value`. If `None`, no
        path information will be added to `value`'s representation.

    Raises:
      UnserializableValueError: If an unserializable value is encountered.
    """
    # If we've already serialized this value, just return a reference to it.
    if id(value) in self._memo:
      return self._ref(self._memo[id(value)])

    if all_paths is not None:
      # If we should add paths (all_paths is not None) and we have an entry for
      # value in self._paths_by_id, use that, since it may contain additional
      # paths not available via the parent.
      all_paths = self._paths_by_id.get(id(value), all_paths)

    traverser = find_node_traverser(type(value))
    if traverser is None:
      # If there is no traverser, handle two special cases below: primitive
      # values and types/functions. Otherwise we don't know how to serialize
      # the value.
      if _is_leaf_type(type(value)):
        if all_paths is None:
          return value  # If we don't need to add paths, just return the value.
        output = self._leaf(value)
      elif isinstance(
          value, (type, types.FunctionType, types.MethodType, enum.Enum)
      ):
        output = self._pyref(value, current_path)
      elif isinstance(value, lazy_imports.ProxyObject):
        output = _serialize_lazy_imports(value)
      elif id(value) in _serialization_constants_by_id:
        output = _serialization_constants_by_id[id(value)].to_pyref()
      elif (isinstance(value, collections.abc.Hashable) and
            value in _serialization_constants_by_value):
        output = _serialization_constants_by_value[value].to_pyref()
      else:
        msg = (f'Unserializable value {value} of type {type(value)}. Error '
               f'occurred at path {path_str(current_path)!r}.")')
        raise UnserializableValueError(msg)
    else:  # We have a traverser; serialize value's flattened elements.
      values, metadata = traverser.flatten(value)
      path_elements = traverser.path_elements(value)
      assert len(values) == len(path_elements)

      serialized_items = []
      for path_element, child_value in zip(path_elements, values):
        child_paths = (
            daglish.add_path_element(all_paths, path_element)
            if all_paths is not None else None)
        serialized_value = self._serialize(
            child_value, current_path + (path_element,), all_paths=child_paths)
        # The serialized item is a two-element list with a string representation
        # of the path element available for debugging/visualization purposes.
        serialized_item = (f'{path_element!r}', serialized_value)
        serialized_items.append(serialized_item)

      if isinstance(metadata, config_lib.BuildableTraverserMetadata):
        metadata = metadata.without_history()
      serialized_metadata = self._serialize(
          metadata, current_path + (MetadataElement(),), all_paths=None)

      output = {
          _TYPE_KEY: self._pyref(type(value), current_path),
          _ITEMS_KEY: serialized_items,
          _METADATA_KEY: serialized_metadata
      }

    if all_paths is not None:
      output[_PATHS_KEY] = path_strs(all_paths)

    if daglish.is_memoizable(value) and output[_TYPE_KEY] != _PYREF_TYPE:
      name = self._unique_name(value)
      assert name not in self._objects
      self._memo[id(value)] = name
      self._objects[name] = output
      self._ref_values.append(value)
      return self._ref(name)
    else:
      return output


class DeserializationError(Exception):
  """Represents an error that occurs during deserialization."""
  pass


class Deserialization:
  """Maintains state for an in-progress deserialization.

  This class is used to recursively recreate a Python object from its
  JSON-serializable dictionary representation (as created by the `Serialization`
  class above). This class manages the state maintained during this proces. The
  resulting deserialized value can be obtained from the result() method once an
  instance has been created:

      value = Deserialization(dict_representation, pyref_policy).result

  A new instance of the class should be created per deserialization.
  """

  def __init__(
      self,
      serialized_value: Dict[str, Any],
      pyref_policy: Optional[PyrefPolicy] = None,
  ):
    """Initializes the deserialization.

    Args:
      serialized_value: The dictionary representation of the value to
        deserialize.
      pyref_policy: The `PyrefPolicy` instance that should govern whether a
        given Python reference is permissible. Defaults to `DefaultPyrefPolicy`
        if not specified.
    """
    # Maps object keys to the object's representation.
    self._serialized_objects: Dict[str, Any] = serialized_value[_OBJECTS_KEY]
    # Maps object keys to their deserialized values.
    self._deserialized_objects: Dict[str, Any] = {}
    # The root object to deserialize. Deserialization starts here.
    self._root = serialized_value[_ROOT_KEY]
    # The active PyrefPolicy.
    self._pyref_policy = pyref_policy or DefaultPyrefPolicy()
    # The deserialized result.
    self._result = self._deserialize(self._root)

  @property
  def result(self) -> Any:
    """Returns the result of the deserialization."""
    return self._result

  def _deserialize_ref(self, ref):
    assert ref[_TYPE_KEY] == _REF_TYPE
    key = ref[_KEY_KEY]
    if key in self._deserialized_objects:
      return self._deserialized_objects[key]

    deserialized_object = self._deserialize(self._serialized_objects[key])
    self._deserialized_objects[key] = deserialized_object
    return deserialized_object

  def _deserialize_pyref(self, pyref):
    assert pyref[_TYPE_KEY] == _PYREF_TYPE
    return import_symbol(self._pyref_policy, pyref[_MODULE_KEY],
                         pyref[_NAME_KEY])

  def _deserialize_leaf(self, leaf):
    assert leaf[_TYPE_KEY] == _LEAF_TYPE
    return leaf[_VALUE_KEY]

  def _deserialize(self, serialized_object: Any):
    """Recursively deserializes the given `serialized_object`."""
    # Lists are always recursively deserialized.
    if isinstance(serialized_object, list):
      return [self._deserialize(x) for x in serialized_object]

    # Anything that isn't a dict must be a primitive value, return as is.
    if not isinstance(serialized_object, dict):
      return serialized_object

    # Now we should be guaranteed that serialized_object has a 'type' field. Get
    # the type of the object and handle the special cases ref, pyref, and leaf.
    object_type = serialized_object[_TYPE_KEY]
    if object_type == _REF_TYPE:
      return self._deserialize_ref(serialized_object)
    elif object_type == _PYREF_TYPE:
      return self._deserialize_pyref(serialized_object)
    elif object_type == _LEAF_TYPE:
      return self._deserialize_leaf(serialized_object)

    # At this point, object_type should itself be a pyref that we can load
    # in order to find a corresponding traverser.
    if (not isinstance(object_type, dict) or
        object_type.get(_TYPE_KEY) != _PYREF_TYPE):
      raise DeserializationError(f'Invalid object type: {object_type}.')

    object_type = self._deserialize(object_type)
    traverser = find_node_traverser(object_type)

    if traverser is None:
      raise DeserializationError(
          f'No traverser available for object type {object_type}.')

    serialized_items = serialized_object[_ITEMS_KEY]
    # Ignore the path element string and get the flattened values.
    values = [value for _, value in self._deserialize(serialized_items)]
    metadata = self._deserialize(serialized_object[_METADATA_KEY])
    return traverser.unflatten(values, metadata)


def dump_json(
    value: Any,
    pyref_policy: Optional[PyrefPolicy] = None,
    indent: Optional[int] = None,
) -> str:
  """Returns the JSON serialization of `value`.

  Args:
    value: The value to serialize.
    pyref_policy: A `PyrefPolicy` instance that governs which Python references
      are permissible. Providing a policy here can help catch potential errors
      early, however this will only guarantee successful deserialization if the
      same policy is passed when calling `load_json`. Defaults to
      `DefaultPyrefPolicy`.
    indent: An optional indentation to use when formatting the JSON. If given,
      the resulting JSON will be formatted with each new indentation level
      indented by `indent` spaces from the last for readability. If not
      provided, the resulting JSON will be compact with minimal whitespace.

  Raises:
    UnserializableValueError: If an unserializable value is encountered.
    PyrefPolicyError: If a Python reference (e.g., function or type) is
      disallowed by `pyref_policy`.
  """
  return json.dumps(Serialization(value, pyref_policy).result, indent=indent)


def load_json(
    serialized_value: str,
    pyref_policy: Optional[PyrefPolicy] = None,
) -> Any:
  """Returns a Python object deserialized from `serialized_value`.

  Args:
    serialized_value: A JSON-serialized value.
    pyref_policy: A `PyrefPolicy` instance that governs which Python references
      are permissible.

  Raises:
    DeserializationError: If an unknown type or other serialization format error
      is encountered  during deserialization.
    ModuleNotFoundError: If a referenced module can't be imported.
    AttributeError: If a referenced attribute or symbol can't be accessed.
    PyrefPolicyError: If a Python reference (e.g., function or type) is
      disallowed by `pyref_policy`.
    json.decoder.JSONDecodeError: If Python's `json.loads` encounters an error
      while decoding the provided JSON.
  """
  return Deserialization(json.loads(serialized_value), pyref_policy).result
