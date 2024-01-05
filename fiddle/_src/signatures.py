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

"""Module for getting (and caching) inspect.Signature objects."""

import dataclasses
import inspect
import sys
from typing import Any, Callable, Dict, Generic, List, Mapping, Optional, Tuple, Type, Union
import weakref
import typing_extensions


# Unique object instance that represents the index where variadic positional
# arguments start for a Buildable.
VARARGS = object()


class NoValue:
  """Sentinel class for arguments with no value."""

  def __repr__(self):
    return 'fdl.NO_VALUE'

  def __deepcopy__(self, memo):
    """Override for deepcopy that does not copy this sentinel object."""
    del memo
    return self

  def __copy__(self):
    """Override for `copy.copy()` that does not copy this sentinel object."""
    return self


NO_VALUE = NoValue()

_signature_cache = weakref.WeakKeyDictionary()
_type_hints_cache = weakref.WeakKeyDictionary()


def _get_signature_uncached(fn_or_cls) -> inspect.Signature:
  """Returns the signature for a function or class.

  This mostly calls inspect.signature, but handles generics and some builtin
  types better.

  Args:
    fn_or_cls: Function or class callable.
  """
  # TODO(b/285387519): Switch to typing.get_origin after dropping Python 3.8
  # support.
  origin = typing_extensions.get_origin(fn_or_cls)
  fn_or_cls = origin if origin is not None else fn_or_cls
  # Work around a bug in inspect.signature
  # https://github.com/python/cpython/issues/85074
  if sys.version_info < (3, 9):
    if isinstance(fn_or_cls, type) and issubclass(fn_or_cls, Generic):
      fn_or_cls = fn_or_cls.__init__

  try:
    return inspect.signature(fn_or_cls)
  except ValueError:
    if isinstance(fn_or_cls, type) and hasattr(fn_or_cls, '__call__'):
      try:
        return inspect.signature(fn_or_cls.__call__)
      except ValueError:
        pass
    raise


def get_signature(fn_or_cls) -> inspect.Signature:
  """Returns (and maybe caches) the signature for `fn_or_cls`.

  If a signature is not available for `fn_or_cls` and `fn_or_cls` is a type,
  this function will instead try to get a signature for `fn_or_cls.__call__`.
  This behavior allows a reasonable signature to be inferred for certain builtin
  types such as `dict`.

  Args:
    fn_or_cls: The function or class to return a signature for.
  """
  try:
    return _signature_cache[fn_or_cls]
  except TypeError:  # Unhashable value...
    return _get_signature_uncached(fn_or_cls)
  except KeyError:
    signature = _get_signature_uncached(fn_or_cls)
    _signature_cache[fn_or_cls] = signature
    return signature


def has_signature(fn_or_cls) -> bool:
  """Returns whether `fn_or_cls` has an available signature.

  Note that this will return `True` even if a signature for `fn_or_cls` is not
  yet in the cache. The signature for `fn_or_cls` will be cached if it is
  available.

  Args:
    fn_or_cls: The function or class to check signature availability for.
  """
  try:
    get_signature(fn_or_cls)
  except (ValueError, TypeError):
    return False
  return True


def _find_class_construction_fn(cls: Type[Any]) -> Callable[..., Any]:
  """Find the first ``__init__`` or ``__new__`` method in the class's MRO."""
  for base in inspect.getmro(cls):
    if '__init__' in base.__dict__:
      return base.__init__
    if '__new__' in base.__dict__:
      return base.__new__
  raise RuntimeError('Could not find a class constructor.')


def _get_type_hints_uncached(
    fn_or_cls,
    *,
    include_extras,
) -> Dict[str, Any]:
  """Returns a dictionary corresponding to the annotations for `fn_or_cls`."""
  try:
    if isinstance(fn_or_cls, type) and not dataclasses.is_dataclass(fn_or_cls):
      fn_or_cls = _find_class_construction_fn(fn_or_cls)
    return typing_extensions.get_type_hints(
        fn_or_cls, include_extras=include_extras
    )
  except (TypeError, NameError):
    return {}


def get_type_hints(
    fn_or_cls: Any, *, include_extras: bool = False
) -> Dict[str, Any]:
  """Returns (and maybe caches) type hints for `fn_or_cls`, suppressing errors."""
  try:
    return _type_hints_cache[fn_or_cls, include_extras]
  except TypeError:  # Unhashable fn_or_cls.
    return _get_type_hints_uncached(fn_or_cls, include_extras=include_extras)
  except KeyError:
    hints = _get_type_hints_uncached(fn_or_cls, include_extras=include_extras)
    _type_hints_cache[fn_or_cls, include_extras] = hints
    return hints


@dataclasses.dataclass
class SignatureInfo:
  """To store signature related information about the callable."""

  signature: inspect.Signature
  has_var_keyword: Optional[bool] = None

  def __post_init__(self):
    self._var_positional_start = None
    for index, param in enumerate(self.signature.parameters.values()):
      if param.kind == param.VAR_POSITIONAL:
        self._var_positional_start = index
      elif param.kind == param.VAR_KEYWORD and self.has_var_keyword is None:
        self.has_var_keyword = True

  @staticmethod
  def signature_binding(
      fn_or_cls, *args, **kwargs
  ) -> Dict[Union[int, str], Any]:
    """Bind arguments and return an args dict in canonical storage format.

    Args:
      fn_or_cls: A function or a class object.
      *args: A list of positional arguments.
      **kwargs: A dict of keyword arguments.

    Returns:
      An arguments dict in canonical storage format. Please see `Buildable`
        docstring for the definition of canonical storage format.
    """
    signature = get_signature(fn_or_cls)
    try:
      arguments = signature.bind_partial(*args, **kwargs).arguments
    except TypeError as e:
      raise TypeError(f'Cannot bind arguments to {fn_or_cls}: {e}') from e
    for index, name in enumerate(list(arguments.keys())):
      param = signature.parameters[name]
      # Use the index as key for positional only arguments
      if param.kind == param.POSITIONAL_ONLY:
        value = arguments.pop(param.name)
        arguments[index] = value
      if param.kind == param.VAR_POSITIONAL:
        values = arguments.pop(param.name)
        for i, value in enumerate(values):
          arguments[index + i] = value
      if param.kind == param.VAR_KEYWORD:
        arguments.update(arguments.pop(param.name))
    return arguments

  def get_default(self, argument: Union[int, str], missing: Any) -> Any:
    """Get default value for the argument, return missing if not found.

    Args:
      argument: Specify the argument either by its name or the index for
        positional arguments.
      missing: A value to return if the argument does not have defaults.

    Returns:
      The default value for the argument if exists, return the missing object
        otherwise.
    """
    value = missing
    param = None
    if isinstance(argument, str):
      if argument in self.parameters:
        param = self.parameters[argument]
    else:
      assert isinstance(argument, int)
      if (
          self.var_positional_start is not None
          and argument < self.var_positional_start
      ):
        params = list(self.parameters.values())
        param = params[argument]
    if param and param.default is not param.empty:
      value = param.default
    return value

  def transform_to_args_kwargs(
      self,
      arguments: Dict[Any, Any],
      include_pos_or_kw_in_args: bool = False,
      include_no_value: bool = False,
  ) -> Tuple[List[Any], Dict[str, Any]]:
    """Transform arguments in dict to *args and **kwargs.

    Args:
      arguments: An arguments dict in canonical storage format (please see
        `Buildable` docstring for the precise definition).
      include_pos_or_kw_in_args: Whether to always put positional-or-keyword
        args in the *args list as output. If set to False, such args will be
        added to **kwargs output dict if the variadic positional args list is
        empty in `arguments`.
      include_no_value: Whether to include `NO_VALUE` in the output. If set as
        True, a value is set to `NO_VALUE` if no user specific value exists and
        its default value is empty.

    Returns:
      A positional args list that can be used to build the Buildable.
      A keyword args dict that can be used to build the Buildable.
    """
    # Make a copy to avoid mutating the input.
    arguments = arguments.copy()
    # Don't use inspect.BoundArguments to generate *args and **kwargs because it
    # will put positional-or-keyword arguments in the *args list. This will lead
    # to unwanted behaviors for `fdl.Partial`, where those arguments should be
    # passed in using keywords so that they be overridden by keyword calling the
    # resulting `Partial`.
    parameters = list(self.parameters.values())
    positional_values = []
    for index, param in enumerate(parameters):
      if param.kind == param.POSITIONAL_ONLY:
        if index in arguments:
          positional_values.append(arguments[index])
          del arguments[index]
        elif include_no_value:
          positional_values.append(self.get_default(index, NO_VALUE))
      if param.kind == param.POSITIONAL_OR_KEYWORD:
        if include_pos_or_kw_in_args or self.var_positional_start in arguments:
          if param.name in arguments:
            positional_values.append(arguments[param.name])
            del arguments[param.name]
          elif include_no_value:
            positional_values.append(self.get_default(index, NO_VALUE))
    if self.var_positional_start is not None:
      index = self.var_positional_start
      while index in arguments:
        positional_values.append(arguments[index])
        del arguments[index]
        index += 1
    return positional_values, arguments

  def validate_param_name(self, name, fn_or_cls) -> None:
    """Raises an error if ``name`` is not a valid parameter name."""
    param = self.signature.parameters.get(name)

    if param is not None:
      if param.kind == param.POSITIONAL_ONLY:
        raise AttributeError(
            f'Cannot access POSITIONAL_ONLY parameter {name!r} on {fn_or_cls}'
        )
      elif param.kind == param.VAR_POSITIONAL:
        raise AttributeError(
            f'Cannot access VAR_POSITIONAL parameter {name!r} on {fn_or_cls}'
        )
      elif param.kind == param.VAR_KEYWORD:
        # Just pretend it doesn't correspond to a valid parameter name... below
        # a TypeError will be thrown unless there is a **kwargs parameter.
        param = None

    if param is None and not self.has_var_keyword:
      if name in self.signature.parameters:
        err_msg = f'Variadic arguments (e.g. *{name}) are not supported.'
      else:
        err_msg = (
            f"No parameter named '{name}' exists for "
            f'{fn_or_cls}; valid parameter names: '
            f"{', '.join(self.valid_param_names)}."
        )
      raise AttributeError(err_msg)

  def replace_varargs_handle(self, key: Union[int, slice]) -> Any:
    """Replace VARARGS handle in index key if exists."""
    replace_fn = lambda x: self.var_positional_start if x is VARARGS else x
    if isinstance(key, slice):
      key = slice(replace_fn(key.start), replace_fn(key.stop), key.step)
    else:
      key = replace_fn(key)
    assert isinstance(
        key, (int, slice)
    ), f'Key must be an int or slice, got {key}.'
    return key

  def index_to_key(
      self, index: int, arguments: Dict[Any, Any]
  ) -> Union[int, str]:
    """Map index of positional arguments to the key for `__arguments__`."""
    if index < 0:
      args, _ = self.transform_to_args_kwargs(arguments, True, True)
      index += len(args)
    params = list(self.signature.parameters.values())
    if index < len(params):
      param = params[index]
      if param.kind == param.POSITIONAL_OR_KEYWORD:
        return param.name
    return index

  @property
  def parameters(self) -> Mapping[str, inspect.Parameter]:
    return self.signature.parameters

  @property
  def valid_param_names(self) -> Tuple[str]:
    return tuple(
        name
        for name, param in self.signature.parameters.items()
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    )

  @property
  def var_keyword_name(self) -> Optional[str]:
    for param in self.signature.parameters.values():
      if param.kind == param.VAR_KEYWORD:
        return param.name
    return None

  @property
  def var_positional_start(self) -> Optional[int]:
    return self._var_positional_start
