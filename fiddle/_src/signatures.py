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
from typing import Any, Callable, Dict, Generic, List, Mapping, Optional, Tuple, Type
import weakref
import typing_extensions

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
  var_positional_name: Optional[str] = None
  positional_arg_names: Optional[List[str]] = dataclasses.field(
      default_factory=list
  )

  def __post_init__(self):
    # During serilization, signature is set to None so no action is needed.
    if self.signature is None:
      return

    # If *args exists, we must pass things before it in positional format. This
    # list tracks those arguments.
    maybe_positional_args = []
    positional_only_args = []
    for param in self.signature.parameters.values():
      if param.kind == param.POSITIONAL_ONLY:
        positional_only_args.append(param.name)
      elif param.kind == param.POSITIONAL_OR_KEYWORD:
        maybe_positional_args.append(param.name)
      elif param.kind == param.VAR_POSITIONAL:
        positional_only_args.extend(maybe_positional_args)
        if not self.var_positional_name:
          self.var_positional_name = param.name
      elif param.kind == param.VAR_KEYWORD and self.has_var_keyword is None:
        self.has_var_keyword = True
    self.positional_arg_names = positional_only_args

  @staticmethod
  def signature_binding(fn_or_cls, *args, **kwargs) -> Any:
    """Bind partial for arguments and return arguments."""
    signature = get_signature(fn_or_cls)
    arguments = signature.bind_partial(*args, **kwargs).arguments
    for name in list(arguments.keys()):  # Make a copy in case we mutate.
      param = signature.parameters[name]
      if param.kind == param.VAR_KEYWORD:
        arguments.update(arguments.pop(param.name))
    return arguments

  def get_positional_names(self) -> Tuple[List[str], List[str], str]:
    """Get positional argument names."""
    positional_only = []
    keyword_or_positional = []
    for param in self.signature.parameters.values():
      if param.kind == param.POSITIONAL_ONLY:
        positional_only.append(param.name)
      elif param.kind == param.POSITIONAL_OR_KEYWORD:
        keyword_or_positional.append(param.name)
    return positional_only, keyword_or_positional, self.var_positional_name

  def validate_param_name(
      self, name, fn_or_cls, allow_postional_argument=False
  ) -> None:
    """Raises an error if ``name`` is not a valid parameter name."""
    param = self.signature.parameters.get(name)

    if param is not None:
      if param.kind == param.POSITIONAL_ONLY and not allow_postional_argument:
        raise AttributeError(
            f'Cannot access POSITIONAL_ONLY parameter {name!r} on {fn_or_cls}'
        )
      elif param.kind == param.VAR_POSITIONAL and not allow_postional_argument:
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
      raise TypeError(err_msg)

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
