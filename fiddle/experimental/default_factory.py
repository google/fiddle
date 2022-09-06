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

"""Function wrapper that defines default factories for function parameters.

This package defines a function wrapper that can be applied directly or using a
decorator, which defines default factories for the parameters of a function.

This wrapper can be applied using a decorator (`@default_factory.for_parameter`)
or by direct construction (`default_factory.CallableWithDefaultDFactory`). Each
time the decorated/wrapped function is called, any missing parameter with a
default factory has its value filled in by calling the factory.  In the
decorator form, this provides an alternative to deal with a common "sharp edge"
of Python, where mutable default arguments do not act as expected.  E.g.:

>>> @default_factory.for_parameter(seq=list)
... def append(elt: T, seq: List[T]) -> T:
...   seq.append(elt)
...   return seq
"""
# TODO: Move this file out of Fiddle, into its own library.

import inspect
from typing import Any, Callable, Dict, Generic, TypeVar

T = TypeVar('T')


class CallableWithDefaultFactory(Generic[T]):
  """A function wrapper that uses default factories for function parameters.

  Each time the wrapper is called, any missing parameter with a default factory
  has its value filled in by calling the factory.

  `default_factory.CallableWithDefaultFactory` is often created by the
  `@default_factory.for_parameter` decorator.  But it can also be useful
  to construct `CallableWithDefaultFactory` directly.  In the following example,
  `CallableWithDefaultFactory` is used to create a factory function for a
  `dataclass` that overrides the `default_factory` for one of its fields:

  >>> @dataclasses.dataclass
  ... class Engine:
  ...   capacity: float = 2.0
  >>> @dataclasses.dataclass
  ... class Car:
  ...   engine: Engine = dataclasses.field(default_factory=Engine)
  >>> @dataclasses.dataclass
  ... class HybridEngine(Engine):
  ...   torque: float = 300
  >>> HybridCar = default_factory.CallableWithDefaultFactory(
  ...     Car, engine=HybridEngine)
  >>> HybridCar()
  Car(engine=HybridEngine(capacity=2.0, torque=300))
  """
  _func: Callable[..., T]
  _default_factories: Dict[str, Callable[..., Any]]

  def __init__(self, func: Callable[..., T], /,
               **default_factories: Callable[..., Any]):
    """Constructs a `CallableWithDefaultFactory`.

    Args:
      func: The function that should be wrapped.
      **default_factories: Default factories for the wrapped function's
        parameters.  Keys may be the names of positional-only parameters,
        keyword-only parameters, or positional-or-keyword parameters.

    Raises:
      TypeError: If `func` or any values of `default_factories` are not
        callable.
      ValueError: If any key of `default_factories` is not a positional
        or keyword parameter to `func`.
    """
    if isinstance(func, CallableWithDefaultFactory):
      default_factories = {**func.default_factories, **default_factories}
      func = func.func

    # Validate constructor arguments.
    if not callable(func):
      raise TypeError(f'func={func!r} is not callable.')
    signature = inspect.signature(func)
    for name, factory in default_factories.items():
      if name not in signature.parameters:
        raise ValueError(f'{name!r} is not a parameter for {func}')
      if signature.parameters[name].kind == inspect.Parameter.VAR_POSITIONAL:
        raise ValueError(
            f'default factory not supported for varargs parameter *{name!r}')
      if signature.parameters[name].kind == inspect.Parameter.VAR_KEYWORD:
        raise ValueError(
            f'default factory not supported for varkwargs parameter **{name!r}')
      if not callable(factory):
        raise TypeError(
            f'Default factory {name!r}={factory!r} is not callable.')

    # Copy the __signature__ from `func`, but set the default value for
    # each default factory argument.
    self.__signature__ = _signature_for_callable_with_default_factories(
        signature, default_factories)

    self._func = func
    self._default_factories = default_factories

  @property
  def func(self):
    """The function wrapped by this CallableWithDefaultFactory."""
    return self._func

  @property
  def default_factories(self) -> Dict[str, Callable[..., Any]]:
    """Default factories for the wrapped function's parameters."""
    return self._default_factories

  def __repr__(self):
    func_name = getattr(self._func, '__qualname__',
                        getattr(self._func, '__name__', repr(self._func)))
    return f'<{type(self).__qualname__} {func_name}{self.__signature__}>'

  def __call__(self, /, *args, **kwargs) -> T:
    signature = self.__signature__
    bound_args = signature.bind_partial(*args, **kwargs)
    for name, factory in self._default_factories.items():
      if name not in bound_args.arguments:
        bound_args.arguments[name] = factory()
    return self._func(*bound_args.args, **bound_args.kwargs)


def _signature_for_callable_with_default_factories(
    func_signature: inspect.Signature,
    default_factories: dict[str, Callable[..., Any]]) -> inspect.Signature:
  """Returns the signature for a callable with default factories.

  * Updates Parameter.default for any parameters with default factories.
  * Changes Parameter.kind from POSITIONAL_OR_KEYWORD to KEYWORD_ONLY as
    necessary, to ensure that any parameter without a default that follows
    a parameter with a default is KEYWORD_ONLY.

  Args:
    func_signature: The signature of the wrapped function.
    default_factories: The default factories.
  """
  parameters = []

  seen_default = False
  keyword_only = False

  for param in func_signature.parameters.values():
    # Add the default factory for this parameter.
    factory = default_factories.get(param.name)
    if factory is not None:
      param = param.replace(default=_DefaultFactory(factory))

    # Ensure that parameters without a default that follow parameters with
    # a default have kind=KEYWORD_ONLY.
    has_default = param.default is not inspect.Parameter.empty
    seen_default = seen_default or has_default
    keyword_only = keyword_only or (seen_default and not has_default)
    if keyword_only and param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
      param = param.replace(kind=inspect.Parameter.KEYWORD_ONLY)
    if keyword_only and param.kind == inspect.Parameter.POSITIONAL_ONLY:
      raise ValueError(
          f'non-default argument {param.name!r} follows default argument.')

    parameters.append(param)
  return func_signature.replace(parameters=parameters)


class _DefaultFactory:
  """Wrapper for default factories in CallableWithDefaultFactory signatures."""

  def __init__(self, factory: Callable[..., Any]):
    self._factory = factory

  def __repr__(self):
    name = getattr(self._factory, '__qualname__',
                   getattr(self._factory, '__name__', repr(self._factory)))
    return f'<built by: {name}()>'


def for_parameter(
    **param_factories
) -> Callable[[Callable[..., T]], CallableWithDefaultFactory[T]]:
  """Decorator that defines default factories for a function's parameters.

  Each time the decorated function is called, any missing parameter with a
  default factory has its value filled in by calling the factory.  For example,
  this can be used to define mutable default values for functions, that are
  created each time the function is called:

  >>> @default_factory.for_parameter(seq=list)
  >>> def append(elt, seq: List[Any]):
  ...   seq.append(elt)
  ...   return seq

  This decorator can also be used in other contexts where the default value
  for a parameter should be computed each time the function is called.  E.g.:

  >>> @default_factory.for_parameter(noise=random.random)
  >>> def add_noise(value, noise):
  ...   return value + noise**2

  Args:
    **param_factories: Default factories for the wrapped function's parameters.
      Keys may be the names of positional-only parameters, keyword-only
      parameters, or positional-or-keyword parameters.

  Raises:
    TypeError: If the wrapped function or any values of `default_factories`
      are not callable.
    ValueError: If any key of `default_factories` is not a positional
      or keyword parameter to the wrapped function.

  Returns:
    A decorator that returns a `CallableWithDefaultFactory`.
  """

  def decorator(func):
    return CallableWithDefaultFactory(func, **param_factories)

  return decorator
