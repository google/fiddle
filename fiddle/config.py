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
import functools
import inspect
from typing import Any, Callable, Collection, Dict, Generic, List, Tuple, Type, TypeVar, Union, overload

from fiddle import build_guard
from fiddle import history
from fiddle import partialclass

import tree

T = TypeVar('T')
TypeOrCallableProducingT = Union[Callable[..., T], Type[T]]
SubclassOfT = TypeVar('SubclassOfT', bound=T)
SubtypeOrCallableProducingT = Union[Callable[..., T], Type[SubclassOfT]]

# Unique object instance that should never be used by end-users, and can thus
# be used to differentiate between unset values and user-set values that are
# None or other commonly-used sentinel.
_UNSET_SENTINEL = object()


class BuildError(ValueError):
  """Error raised when building a Config fails."""

  def __init__(
      self,
      buildable: Buildable,
      path_from_config_root: str,
      original_error: Exception,
      args: Tuple[Any, ...],
      kwargs: Dict[str, Any],
  ) -> None:
    super().__init__(str(original_error))
    self.buildable = buildable
    self.path_from_config_root = path_from_config_root
    self.original_error = original_error
    self.args = args
    self.kwargs = kwargs

  def __str__(self):
    fn_or_cls_name = self.buildable.__fn_or_cls__.__qualname__
    return (f'Failed to construct or call {fn_or_cls_name} '
            f'(at {self.path_from_config_root}) with arguments\n'
            f'    args: {self.args}\n'
            f'    kwargs: {self.kwargs}')


# This error lives here to avoid a circular dependency. Please see
# placeholders.py.
class PlaceholderNotFilledError(ValueError):
  """A placeholder was not filled when build() was called."""


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
  _has_var_keyword: bool

  def __init__(self, fn_or_cls: Union['Buildable', TypeOrCallableProducingT],
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
    super().__setattr__('__argument_history__', collections.defaultdict(list))
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

    for name, value in arguments.items():
      setattr(self, name, value)

  @abc.abstractmethod
  def __build__(self, *args, **kwargs):
    """Builds output for this instance; see subclasses for details."""
    raise NotImplementedError()

  def __getattr__(self, name: str):
    """Get parameter with given `name`."""
    value = self.__arguments__.get(name, _UNSET_SENTINEL)
    if value is not _UNSET_SENTINEL:
      return value
    param = self.__signature__.parameters.get(name)
    if param is not None and param.default is not param.empty:
      return param.default
    msg = f"No parameter '{name}' has been set on {self!r}."
    if hasattr(self.__fn_or_cls__, name):
      msg += (f' Note: {self.__fn_or_cls__.__module__}.'
              f'{self.__fn_or_cls__.__qualname__} has an attribute/method with '
              'this name, so this could be caused by using a '
              f'fdl.{self.__class__.__qualname__} in '
              'place of the actual function or class being configured. Did you '
              'forget to call `fdl.build(...)`?')
    raise AttributeError(msg)

  def __setattr__(self, name: str, value: Any):
    """Sets parameter `name` to `value`."""
    param = self.__signature__.parameters.get(name)

    if param is not None:
      if param.kind == param.POSITIONAL_ONLY:
        # TODO: Add positional-only arg support.
        raise NotImplementedError('Positional only arguments not supported.')
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

    self.__arguments__[name] = value
    self.__argument_history__[name].append(history.entry(name, value))

  def __delattr__(self, name):
    """Unsets parameter `name`."""
    try:
      del self.__arguments__[name]
      entry = history.entry(name, history.DELETED)
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
    formatted_params = [f'{k}={v!r}' for k, v in self.__arguments__.items()]
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
    return type(self)(self)  # pytype: disable=not-instantiable

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
    config_copy = type(self)(self.__fn_or_cls__)  # pytype: disable=not-instantiable
    deepcopied_arguments = copy.deepcopy(self.__arguments__, memo)
    object.__setattr__(config_copy, '__arguments__', deepcopied_arguments)
    return config_copy

  def __eq__(self, other):
    """Returns true iff self and other contain the same argument values."""
    if not isinstance(other, self.__class__):
      return False
    if self.__fn_or_cls__ is not other.__fn_or_cls__:
      return False
    assert self._has_var_keyword == other._has_var_keyword, (
        'Internal invariant violated: has_var_keyword should be the same if '
        "__fn_or_cls__'s are the same.")

    return self.__arguments__ == other.__arguments__

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

      TestClass:
        '''Example class for demonstration purposes.'''

        def __init__(self, arg, kwarg=None):
          self.arg = arg
          self.kwarg = kwarg

  a configuration may (for instance) be accomplished via

      class_config = Config(TestClass, 1, kwarg='kwarg')

  or via

      class_config = Config(TestClass)
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
  outputs from nested `Config`s). For example, using the `TestClass` config from
  above:

      instance = build(class_config)
      assert instance.arg == 'arg'
      assert instance.kwarg == 'kwarg'

  If the same `Config` instance is used in multiple places within the
  configuration tree, its function or class is invoked only once during `build`,
  and the result shared across all occurences of the `Config` instance. (See
  `build` documentation for further details.) To create a new instance of a
  `Config` with the same parameter settings that will yield a separate instance
  during `build`, `copy.copy()` or `copy.deepcopy()` may be used.
  """

  # NOTE: We currently need to repeat these annotations for pytype.
  __fn_or_cls__: TypeOrCallableProducingT
  __signature__: inspect.Signature

  def __build__(self, *args, **kwargs):
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
  instead output a corresponding `functools.partial` or `partialclass` object.
  Where the `Config` base class calls its underlying `__fn_or_cls__` when built,
  this `Partial` instead results in a partially bound function or class:

      partial_config = Partial(TestClass)
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

  def __build__(self, *args, **kwargs):
    """Builds this `Partial` for the given `args` and `kwargs`.

    This method is called during `build` to get the output for this `Partial`.

    Args:
      *args: Positional arguments to partially bind to `self.__fn_or_cls__`.
      **kwargs: Keyword arguments to partially bind to `self.__fn_or_cls__`.

    Returns:
      A `functools.partial` or `partialclass` instance with `args` and `kwargs`
      bound to `self.__fn_or_cls__`.
    """
    is_class = inspect.isclass(self.__fn_or_cls__)
    partial_type = partialclass.partialclass if is_class else functools.partial
    return partial_type(self.__fn_or_cls__, *args, **kwargs)

  def __call__(self, *args, **kwargs):
    """Converts this `Partial` into a `Config`, with optional overrides.

    Args:
      *args: Positional arguments to configure for `self.__fn_or_cls__`.
      **kwargs: Keyword arguments to configure for `self.__fn_or_cls__`.

    Returns:
      A `Config` instance copied from this `Partial`. All arguments are copied
      from this `Partial` instance, and arguments may additionally be
      (re)configured by passing `args` or `kwargs`.
    """
    return Config(self, *args, **kwargs)


# Define Pytypes for `build(Partial)`
@overload
def build(config: Partial[T],) -> SubtypeOrCallableProducingT:
  pass


# Define Pytypes for `build(Config)`
@overload
def build(config: Config[T],) -> T:
  pass


# This is a free function instead of a method on the `Config` object in order
# to avoid potential naming collisions (e.g., if a function or class has a
# parameter named `build`).
def build(config):
  """Builds `config`, recursively building any nested `Buildable` instances.

  This is the core function for turning a `Config` into a usable object. It
  recursively walks through `config`'s parameters, building any nested `Config`
  instances. Depending on the specific `Config` type passed (`Config` or
  `Partial`), the result is either the result of calling `config.__fn_or_cls__`
  with the configured parameters, or a partial function or class with those
  parameters bound.

  If the same `Config` instance is seen multiple times during traversal of the
  configuration tree, `build` is called only once (for the first instance
  encountered), and the result is reused for subsequent copies of the instance.
  This is achieved via the `memo` dictionary (similar to `deepcopy`). This has
  the effect that for configured class instances, each separate config instance
  is in one-to-one correspondence with an actual instance of the configured
  class after calling `build` (shared config instances <=> shared class
  instances).

  Args:
    config: A `Config` instance to build.

  Returns:
    The built version of `config`.
  """
  memo = {}

  def _build(config, path_str: str):

    def map_fn(map_path: List[Any], leaf):
      attr, *rest = map_path
      leaf_path = f'{path_str}.{attr}' + ''.join(f'[{x!r}]' for x in rest)
      return _build(leaf, leaf_path) if isinstance(leaf, Buildable) else leaf

    if id(config) not in memo:
      kwargs = tree.map_structure_with_path(map_fn, config.__arguments__)
      try:
        memo[id(config)] = config.__build__(**kwargs)
      except PlaceholderNotFilledError:
        raise
      except Exception as e:
        raise BuildError(config, path_str, e, (), kwargs) from e

    return memo[id(config)]

  with build_guard.in_build():
    return _build(config, '<root>')
