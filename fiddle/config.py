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

"""Defines the `Config` class and associated subclasses and functions."""
from __future__ import annotations

import abc
import collections
import copy
import dataclasses
import functools
import inspect
import types
from typing import Any, Callable, Collection, Dict, FrozenSet, Generic, Iterable, List, Mapping, NamedTuple, Set, Tuple, Type, TypeVar, Union
from fiddle import daglish
from fiddle import history
from fiddle import tag_type
from fiddle._src import field_metadata

T = TypeVar('T')
TypeOrCallableProducingT = Union[Callable[..., T], Type[T]]

# Unique object instance that should never be used by end-users, and can thus
# be used to differentiate between unset values and user-set values that are
# None or other commonly-used sentinel.
_UNSET_SENTINEL = object()


class BuildableTraverserMetadata(NamedTuple):
  """Metadata for a Buildable.

  This separate class is used for DAG traversals.
  """
  fn_or_cls: Callable[..., Any]
  argument_names: Tuple[str, ...]
  argument_tags: Dict[str, FrozenSet[tag_type.TagType]]
  argument_history: Mapping[str, Tuple[history.HistoryEntry, ...]] = (
      types.MappingProxyType({}))

  def without_history(self) -> BuildableTraverserMetadata:
    return self._replace(argument_history={})

  def arguments(self, values: Iterable[Any]) -> Dict[str, Any]:
    """Returns a dictionary combining `self.argument_names` with `values`."""
    return dict(zip(self.argument_names, values))

  def tags(self) -> Dict[str, set[tag_type.TagType]]:
    return collections.defaultdict(
        set, {name: set(tags) for name, tags in self.argument_tags.items()})

  def history(self) -> Dict[str, List[history.HistoryEntry]]:
    return collections.defaultdict(list, {
        name: list(entries) for name, entries in self.argument_history.items()
    })


class Buildable(Generic[T], metaclass=abc.ABCMeta):
  """Base class for buildable types (`Config` and `Partial`).

  Buildable types implement a `__build__` method that is called during
  `fdl.build()` with arguments set on the `Buildable` to get output for the
  corresponding instance.
  """

  # "Dunder"-style names were chosen here to reduce the possibility of naming
  # conflicts while still making the fields accessible.
  __fn_or_cls__: TypeOrCallableProducingT
  __signature__: inspect.Signature
  __arguments__: Dict[str, Any]
  __argument_history__: Dict[str, List[history.HistoryEntry]]
  __argument_tags__: Dict[str, Set[tag_type.TagType]]
  _has_var_keyword: bool

  def __init__(self, fn_or_cls: Union['Buildable', TypeOrCallableProducingT], /,
               *args, **kwargs):
    """Initialize for `fn_or_cls`, optionally specifying parameters.

    Args:
      fn_or_cls: The function or class to configure, or a `Buildable` to copy.
      *args: Any positional arguments to configure for `fn_or_cls`.
      **kwargs: Any keyword arguments to configure for `fn_or_cls`.
    """
    # Using `super().__setattr__` here because assigning directly would trigger
    # our `__setattr__` override. Using `super().__setattr__` instead of special
    # casing these attribute names in `__setattr__` also has the effect of
    # making them easily gettable but not as easily settable by user code.
    if isinstance(fn_or_cls, Buildable):
      initial_arguments = fn_or_cls.__arguments__.copy()
      fn_or_cls = fn_or_cls.__fn_or_cls__
    else:
      initial_arguments = {}
    super().__setattr__('__fn_or_cls__', fn_or_cls)
    super().__setattr__('__arguments__', initial_arguments)
    signature = inspect.signature(fn_or_cls)
    super().__setattr__('__signature__', signature)
    has_var_keyword = any(param.kind == param.VAR_KEYWORD
                          for param in signature.parameters.values())
    arg_history = collections.defaultdict(list)
    arg_history['__fn_or_cls__'].append(
        history.new_value('__fn_or_cls__', fn_or_cls))
    super().__setattr__('__argument_history__', arg_history)
    super().__setattr__('__argument_tags__', collections.defaultdict(set))
    super().__setattr__('_has_var_keyword', has_var_keyword)

    arguments = signature.bind_partial(*args, **kwargs).arguments
    for name in list(arguments.keys()):  # Make a copy in case we mutate.
      param = signature.parameters[name]
      if param.kind == param.VAR_POSITIONAL:
        # TODO: Add *args support.
        err_msg = 'Variable positional arguments (aka `*args`) not supported.'
        raise NotImplementedError(err_msg)
      elif param.kind == param.VAR_KEYWORD:
        arguments.update(arguments.pop(param.name))

    if hasattr(fn_or_cls, '__fiddle_init__'):
      fn_or_cls.__fiddle_init__(self)

    if dataclasses.is_dataclass(self.__fn_or_cls__):
      fields = dataclasses.fields(self.__fn_or_cls__)
      for field in fields:
        metadata = field_metadata.field_metadata(field)
        if metadata:
          for tag in metadata.tags:
            add_tag(self, field.name, tag)
          if (metadata.buildable_initializer and field.name not in arguments):
            setattr(self, field.name, metadata.buildable_initializer())

    for name, value in arguments.items():
      setattr(self, name, value)

  def __init_subclass__(cls):
    daglish.register_node_traverser(
        cls,
        flatten_fn=lambda x: x.__flatten__(),
        unflatten_fn=cls.__unflatten__,
        path_elements_fn=lambda x: x.__path_elements__(),
    )

  @abc.abstractmethod
  def __build__(self, /, *args, **kwargs):
    """Builds output for this instance; see subclasses for details."""
    raise NotImplementedError()

  def __flatten__(self) -> Tuple[Tuple[Any, ...], BuildableTraverserMetadata]:
    arguments = ordered_arguments(self)
    keys = tuple(arguments.keys())
    values = tuple(arguments.values())
    tags = {
        name: frozenset(tags) for name, tags in self.__argument_tags__.items()
    }
    argument_history = {
        name: tuple(entries)
        for name, entries in self.__argument_history__.items()
    }
    metadata = BuildableTraverserMetadata(
        fn_or_cls=self.__fn_or_cls__,
        argument_names=keys,
        argument_tags=tags,
        argument_history=argument_history)
    return values, metadata

  @classmethod
  def __unflatten__(cls, values: Iterable[Any],
                    metadata: BuildableTraverserMetadata):

    rebuilt = cls(metadata.fn_or_cls, **metadata.arguments(values))  # pytype: disable=not-instantiable
    object.__setattr__(rebuilt, '__argument_tags__', metadata.tags())
    object.__setattr__(rebuilt, '__argument_history__', metadata.history())
    return rebuilt

  def __path_elements__(self):
    return tuple(daglish.Attr(name) for name in ordered_arguments(self).keys())

  def __getattr__(self, name: str):
    """Get parameter with given `name`."""
    value = self.__arguments__.get(name, _UNSET_SENTINEL)
    if value is not _UNSET_SENTINEL:
      return value
    if (dataclasses.is_dataclass(self.__fn_or_cls__) and
        _field_uses_default_factory(self.__fn_or_cls__, name)):
      raise ValueError("Can't get default value for dataclass field " +
                       f'{self.__fn_or_cls__.__qualname__}.{name} ' +
                       'since it uses a default_factory.')
    param = self.__signature__.parameters.get(name)
    if param is not None and param.default is not param.empty:
      return param.default
    msg = f"No parameter '{name}' has been set on {self!r}."
    # TODO: Implement an edit distance function and display valid
    # attributes that are close to `name`.
    if hasattr(self.__fn_or_cls__, name):
      msg += (f' Note: {self.__fn_or_cls__.__module__}.'
              f'{self.__fn_or_cls__.__qualname__} has an attribute/method with '
              'this name, so this could be caused by using a '
              f'fdl.{self.__class__.__qualname__} in '
              'place of the actual function or class being configured. Did you '
              'forget to call `fdl.build(...)`?')
    raise AttributeError(msg)

  def __validate_param_name__(self, name) -> None:
    """Raises an error if `name` is not a valid parameter name."""
    param = self.__signature__.parameters.get(name)

    if param is not None:
      if param.kind == param.POSITIONAL_ONLY:
        # TODO: Add positional-only arg support.
        raise NotImplementedError(
            'Positional only arguments not supported. '
            f'Tried to set {name!r} on {self.__fn_or_cls__}')
      elif param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
        # Just pretend it doesn't correspond to a valid parameter name... below
        # a TypeError will be thrown unless there is a **kwargs parameter.
        param = None

    if param is None and not self._has_var_keyword:
      if name in self.__signature__.parameters:
        err_msg = (f'Variadic arguments (e.g. *{name}) are not supported.')
      else:
        valid_parameter_names = (
            name for name, param in self.__signature__.parameters.items()
            if param.kind not in (param.POSITIONAL_ONLY, param.VAR_POSITIONAL))
        err_msg = (f"No parameter named '{name}' exists for "
                   f'{self.__fn_or_cls__}; valid parameter names: '
                   f"{', '.join(valid_parameter_names)}.")
      raise TypeError(err_msg)

  def __setattr__(self, name: str, value: Any):
    """Sets parameter `name` to `value`."""

    self.__validate_param_name__(name)
    self.__arguments__[name] = value
    self.__argument_history__[name].append(history.new_value(name, value))

  def __delattr__(self, name):
    """Unsets parameter `name`."""
    try:
      del self.__arguments__[name]
      entry = history.deleted_value(name)
      self.__argument_history__[name].append(entry)
    except KeyError:
      err = AttributeError(f"No parameter '{name}' has been set on {self!r}")
      raise err from None

  def __dir__(self) -> Collection[str]:
    """Provide a useful list of attribute names, optimized for Jupyter/Colab.

    `__dir__` is often implicitly called by tooling such as Jupyter/Colab to
    provide autocomplete suggestions. This implementation of `__dir__` makes it
    easy to see what are valid attributes to get, set, or delete.

    Returns:
      A list of useful attribute names corresponding to set or unset parameters.
    """
    valid_param_names = {
        name for name, param in self.__signature__.parameters.items()
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }
    set_argument_names = self.__arguments__.keys()
    all_names = valid_param_names.union(set_argument_names)
    return all_names

  def __repr__(self):
    if hasattr(self.__fn_or_cls__, '__qualname__'):
      formatted_fn_or_cls = self.__fn_or_cls__.__qualname__
    else:
      formatted_fn_or_cls = str(self.__fn_or_cls__)
    formatted_params = []
    # Show parameters from signature first (in signature order) followed by
    # **varkwarg parameters (in the order they were set).
    param_names = (
        list(self.__signature__.parameters) + [
            name for name in self.__arguments__
            if name not in self.__signature__.parameters
        ])
    for name in param_names:
      tags = self.__argument_tags__.get(name, ())
      value = self.__arguments__.get(name, _UNSET_SENTINEL)
      if tags or (value is not _UNSET_SENTINEL):
        param_str = name
        if tags:
          param_str += f"[{', '.join(sorted(str(tag) for tag in tags))}]"
        if value is not _UNSET_SENTINEL:
          param_str += f'={value!r}'
        formatted_params.append(param_str)
    name = type(self).__name__
    return f"<{name}[{formatted_fn_or_cls}({', '.join(formatted_params)})]>"

  def __copy__(self):
    """Shallowly copies this `Buildable` instance.

    This copy implementation ensures that setting parameters on a copy of a
    `Buildable` won't affect the original instance. However, the copy is
    shallow, so parameter values will still refer to the same instances of
    objects after the copy.

    Returns:
      A shallow copy of this `Buildable`.
    """
    # TODO: Preserve argument history...
    return self.__unflatten__(*self.__flatten__())

  def __deepcopy__(self, memo):
    """Deeply copies this `Buildable` instance.

    This copy implementation ensures that setting parameters on a copy of a
    `Buildable` won't affect the original instance.

    Args:
      memo: The `deepcopy` memoization dictionary (used to avoid copying the
        same instance of an object more than once).

    Returns:
      A deep copy of this `Buildable`.
    """
    values, metadata = self.__flatten__()
    deepcopied_values = copy.deepcopy(values, memo)
    deepcopied_metadata = copy.deepcopy(metadata, memo)
    return self.__unflatten__(deepcopied_values, deepcopied_metadata)

  def __eq__(self, other):
    """Returns true iff self and other contain the same argument values.

    This compares the specific types of `self` and `other`, the function or
    class being configured, and then checks for equality in the configured
    arguments.

    Default argument values are considered in this comparison: If one
    `Buildable` has an argument explicitly set to its default value while
    another does not, the two will still be considered equal (motivated by the
    fact that calls to the function or class being configured will be the same).

    Argument history is not compared (i.e., it doesn't matter how the
    `Buildable`s being compared reached their current state).

    Args:
      other: The other value to compare `self` to.

    Returns:
      `True` if `self` equals `other`, `False` if not.
    """
    if type(self) is not type(other):
      return False
    if self.__fn_or_cls__ != other.__fn_or_cls__:
      return False
    assert self._has_var_keyword == other._has_var_keyword, (
        'Internal invariant violated: has_var_keyword should be the same if '
        "__fn_or_cls__'s are the same.")

    missing = object()
    for key in set(self.__arguments__) | set(other.__arguments__):
      v1 = getattr(self, key, missing)
      v2 = getattr(other, key, missing)
      if (v1 is not missing or v2 is not missing) and v1 != v2:
        return False
    return True

  def __getstate__(self):
    """Gets pickle serialization state, removing some fields.

    For now, we discard the __signature__ (which can be recalculated) and
    __argument_history__, because these tend to contain values which cannot be
    serialized.

    Returns:
      Dict of serialized state.
    """
    result = dict(self.__dict__)
    result.pop('__signature__', None)
    return result

  def __setstate__(self, state) -> None:
    """Loads pickle serialization state.

    This re-derives the signature if not present, and adds an empty
    __argument_history__, if it was removed.

    Args:
      state: State dictionary, from __getstate__.
    """
    self.__dict__.update(state)  # Support unpickle.
    if '__signature__' not in self.__dict__:
      object.__setattr__(self, '__signature__',
                         inspect.signature(self.__fn_or_cls__))


class Config(Generic[T], Buildable[T]):
  """A mutable representation of a function or class's parameters.

  This class represents the configuration for a given function or class,
  exposing configured parameters as mutable attributes. For example, given a
  class

      class SampleClass:
        '''Example class for demonstration purposes.'''

        def __init__(self, arg, kwarg=None):
          self.arg = arg
          self.kwarg = kwarg

  a configuration may (for instance) be accomplished via

      class_config = Config(SampleClass, 1, kwarg='kwarg')

  or via

      class_config = Config(SampleClass)
      class_config.arg = 1
      class_config.kwarg = 'kwarg'

  A function can be configured in the same ways:

      def test_function(arg, kwarg=None):
        return arg, kwarg

      fn_config = Config(test_function, 1)
      fn_config.kwarg = 'kwarg'

  A `Config` instance may be transformd into instances and function outputs by
  passing it to the `build` function. The `build` function invokes each function
  or class in the configuration tree (appropriately propagating the built
  outputs from nested `Config`s). For example, using the `SampleClass` config
  from above:

      instance = build(class_config)
      assert instance.arg == 'arg'
      assert instance.kwarg == 'kwarg'

  If the same `Config` instance is used in multiple places within the
  configuration tree, its function or class is invoked only once during `build`,
  and the result shared across all occurences of the `Config` instance. (See
  `build` documentation for further details.) To create a new instance of a
  `Config` with the same parameter settings that will yield a separate instance
  during `build`, `copy.copy()` or `copy.deepcopy()` may be used.

  A class or function can customize the Config instance by defining a
  `__fiddle_init__` property. For example:

      class MyClass:
        def __init__(self, x, y, z):
          ...

        @staticmethod
        def __fiddle_init__(cfg):
          cfg.y = 42
          cfg.z = Config(MyOtherClass)
  """

  # NOTE: We currently need to repeat these annotations for pytype.
  __fn_or_cls__: TypeOrCallableProducingT
  __signature__: inspect.Signature

  def __build__(self, /, *args, **kwargs):
    """Builds this `Config` for the given `args` and `kwargs`.

    This method is called during `build` to get the output for this `Config`.

    Args:
      *args: Positional arguments to pass to `self.__fn_or_cls__`.
      **kwargs: Keyword arguments to pass to `self.__fn_or_cls__`.

    Returns:
      The result of calling `self.__fn_or_cls__` with the given `args` and
      `kwargs`.
    """
    return self.__fn_or_cls__(*args, **kwargs)


class Partial(Generic[T], Buildable[T]):
  """A `Partial` config creates a partial function or class when built.

  In some cases, it may be desired to leave a function or class uninvoked, and
  instead output a corresponding `functools.partial` object. Where the `Config`
  base class calls its underlying `__fn_or_cls__` when built, this `Partial`
  instead results in a partially bound function or class:

      partial_config = Partial(SampleClass)
      partial_config.arg = 1
      partial_config.kwarg = 'kwarg'
      partial_class = build(partial_config)
      instance = partial_class(arg=2)  # Keyword arguments can be overridden.
      assert instance.arg == 2
      assert instance.kwarg == 'kwarg'

  A `Partial` can also be created from an existing `Config`, by passing it to
  the `Partial` constructor. This results in a shallow copy that is decoupled
  from the `Config` used to create it. In the example below, any further changes
  to `partial_config` are not reflected by `class_config` (and vice versa):

      partial_config = Partial(class_config)
      class_config.arg = 'new value'  # Further modification to `class_config`.
      partial_class = build(partial_config)
      instance = partial_class()
      assert instance.arg == 'arg'  # The instance config is still 'arg'.
  """

  # NOTE: We currently need to repeat this annotation for pytype.
  __fn_or_cls__: TypeOrCallableProducingT

  def __build__(self, /, *args, **kwargs):
    """Builds this `Partial` for the given `args` and `kwargs`.

    This method is called during `build` to get the output for this `Partial`.

    Args:
      *args: Positional arguments to partially bind to `self.__fn_or_cls__`.
      **kwargs: Keyword arguments to partially bind to `self.__fn_or_cls__`.

    Returns:
      A `functools.partial` or instance with `args` and `kwargs` bound to
      `self.__fn_or_cls__`.
    """
    return functools.partial(self.__fn_or_cls__, *args, **kwargs)


def _field_uses_default_factory(dataclass_type: Type[Any], field_name: str):
  """Returns true if <dataclass_type>.<field_name> uses a default_factory."""
  for field in dataclasses.fields(dataclass_type):
    if field.name == field_name:
      return field.default_factory != dataclasses.MISSING
  return False


def update_callable(buildable: Buildable,
                    new_callable: TypeOrCallableProducingT):
  """Updates `config` to build `new_callable` instead.

  When extending a base configuration, it can often be useful to swap one class
  for another. For example, an experiment may want to swap in a subclass that
  has augmented functionality.

  `update_callable` updates `config` in-place (preserving argument history).

  Args:
    buildable: A `Buildable` (e.g. a `fdl.Config`) to mutate.
    new_callable: The new callable `config` should call when built.

  Raises:
    TypeError: if `new_callable` has varargs, or if there are arguments set on
      `config` that are invalid to pass to `new_callable`.
  """
  # TODO: Consider adding a "drop_invalid_args: bool = False" argument.

  # Note: can't just call config.__init__(new_callable, **config.__arguments__)
  # to preserve history.
  #
  # Note: can't call `setattr` on all the args to validate them, because that
  # will result in duplicate history entries.

  original_args = buildable.__arguments__
  signature = inspect.signature(new_callable)
  if any(param.kind == param.VAR_POSITIONAL
         for param in signature.parameters.values()):
    raise NotImplementedError(
        'Variable positional arguments (aka `*args`) not supported.')
  has_var_keyword = any(param.kind == param.VAR_KEYWORD
                        for param in signature.parameters.values())
  if not has_var_keyword:
    invalid_args = [
        arg for arg in original_args.keys() if arg not in signature.parameters
    ]
    if invalid_args:
      raise TypeError(f'Cannot switch to {new_callable} (from '
                      f'{buildable.__fn_or_cls__}) because the Buildable would '
                      f'have invalid arguments {invalid_args}.')

  object.__setattr__(buildable, '__fn_or_cls__', new_callable)
  object.__setattr__(buildable, '__signature__', signature)
  object.__setattr__(buildable, '_has_var_keyword', has_var_keyword)
  buildable.__argument_history__['__fn_or_cls__'].append(
      history.new_value('__fn_or_cls__', new_callable))


def assign(buildable: Buildable, /, **kwargs):
  """Assigns multiple arguments to cfg.

  Although this function does not enable a caller to do something they can't
  already do with other syntax, this helper function can be useful when
  manipulating deeply nested configs. Example:

  ```py
  cfg = # ...
  fdl.assign(cfg.my.deeply.nested.child.object, arg_a=1, arg_b='b')
  ```

  The above code snippet is equivalent to:

  ```py
  cfg = # ...
  cfg.my.deeply.nested.child.object.arg_a = 1
  cfg.my.deeply.nested.child.object.arg_b = 'b'
  ```

  Args:
    buildable: A `Buildable` (e.g. a `fdl.Config`) to set values upon.
    **kwargs: The arguments and values to assign.
  """
  for name, value in kwargs.items():
    setattr(buildable, name, value)


def ordered_arguments(buildable: Buildable) -> Dict[str, Any]:
  """Returns arguments of a Buildable, ordered by the signature."""
  result = {}
  for name, param in buildable.__signature__.parameters.items():
    if param.kind != param.VAR_KEYWORD and name in buildable.__arguments__:
      result[name] = buildable.__arguments__[name]
  for name, value in buildable.__arguments__.items():
    result.setdefault(name, value)
  return result


def add_tag(buildable: Buildable, argument: str, tag: tag_type.TagType) -> None:
  """Tags `name` with `tag` in `buildable`."""
  buildable.__validate_param_name__(argument)
  buildable.__argument_tags__[argument].add(tag)
  buildable.__argument_history__[argument].append(
      history.update_tags(argument, buildable.__argument_tags__[argument]))


def set_tags(buildable: Buildable, argument: str,
             tags: Collection[tag_type.TagType]) -> None:
  """Sets tags for a parameter in `buildable`, overriding existing tags."""
  clear_tags(buildable, argument)
  for tag in tags:
    add_tag(buildable, argument, tag)
  buildable.__argument_history__[argument].append(
      history.update_tags(argument, buildable.__argument_tags__[argument]))


def remove_tag(buildable: Buildable, argument: str,
               tag: tag_type.TagType) -> None:
  """Removes a given tag from a named argument of a Buildable."""
  buildable.__validate_param_name__(argument)
  field_tag_set = buildable.__argument_tags__[argument]
  if tag not in field_tag_set:
    raise ValueError(
        f'{tag} not set on {argument}; current tags: {field_tag_set}.')
  # TODO: Track in history?
  field_tag_set.remove(tag)
  buildable.__argument_history__[argument].append(
      history.update_tags(argument, buildable.__argument_tags__[argument]))


def clear_tags(buildable: Buildable, argument: str) -> None:
  """Removes all tags from a named argument of a Buildable."""
  buildable.__validate_param_name__(argument)
  buildable.__argument_tags__[argument].clear()
  buildable.__argument_history__[argument].append(
      history.update_tags(argument, buildable.__argument_tags__[argument]))


def get_tags(buildable: Buildable,
             argument: str) -> FrozenSet[tag_type.TagType]:
  return frozenset(buildable.__argument_tags__[argument])
