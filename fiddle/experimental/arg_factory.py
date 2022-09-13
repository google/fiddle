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

"""Functions that generate callable arguments each time they are called.

The `arg_factory` package provides two function wrappers that can be used
to generate parameter values each time a wrapped function is called:

  * `@arg_factory.default_factory_for` is a decorator that defines a "default
    factory" for one or more parameters.  Each time the decorated function
    is called, any missing parameter with a default factory has its value
    filled in by calling the factory.  For example, this can be used to define
    mutable default values for functions, that are created each time the
    function is called.

  * `arg_factory.partial` is a specialized version of `functools.partial`
    that binds parameters to factories rather than values.  These factories
    are used to compute the parameter values each time the partial function
    is called.

## Differences between `default_factory_for` and `partial`

These two wrappers have similar behavior for keyword arguments, but differ in
how they handle factories for positional arguments:

  * When `arg_factory.partial` binds an argument positionally, that argument is
    removed from the signature.  I.e., the wrapper function no longer expects
    that argument.

  * In contrast, `@arg_factory.default_factory_for` does not remove any
    arguments from the signature of the wrapped function.

The wrappers also differ in their restrictions for positional-only
arguments.  Consider a function with signature `f(x, y, /)`:

  * `arg_factory.partial` may be used to define a factory for
    `x` without defining one for `y` (and doing so will "consume" the argument
    `x`); but `@arg_factory.default_factory_for` can not (since non-default
    arguments may not follow default arguments).

  * Conversely, `@arg_factory.default_factory_for` may be used to define a
    factory for `y` without defining one for `x` (and doing so will make `y`
    an optional argument); but `arg_factory.partial` can not (since `y` is a
    positional-only argument, so there's no way to bind it without also binding
    `x`).

A final difference is that `@arg_factory.default_factory_for` may only be used
to provide default factories for positional and keyword parameters in the
wrapped function's signature; it may not be used to add factories for with var-
positional (`*args`) or var-keyword (**kwargs`) parameters. But
`arg_factory.partial` *can* be used to provide a default for a var-positional or
var-keyword argument.  E.g.:

>>> def f(**kwargs):
...   print(kwargs)
>>> p = arg_factory.partial(f, x=list)
>>> p(y=3)
{'x': [], y: 3}
"""
# TODO: Move this file out of Fiddle, into its own library.

import dataclasses
import functools
import inspect
from typing import Any, Callable, Dict, Generic, TypeVar

T = TypeVar('T')


def partial(func, /, *args, **kwargs):
  """A specialized version of `functools.partial` that binds args to factories.

  `arg_factory.partial` is similar to `functools.partial`, except that it binds
  arguments to *factory functions* rather than *values*.  These factory
  functions are used to construct value for an argument each time the partial is
  called.

  Example:

  Consider the following code.

  >>> def f(x, noise): return x + noise
  >>> g = functools.partial(f, noise=random.random())
  >>> g(5) == g(5)  # noise has the same value for each call to `g`.
  True

  Here, the value for the `noise` parameter is set when the partial object is
  created, so each call to `g` will use the same value.  If we want a new
  value for each call to the partial, we can use `arg_factory.partial`:

  >>> g = arg_factory.partial(f, noise=random.random)
  >>> g(5) == g(5)  # noise has a new value for each call to g.
  False

  Since we used `arg_factory.partial`, each call to `g` will call the
  `random.random`, and so will use a different value for `noise`.

  Args:
    func: The function whose arguments should be bound to factories.
    *args: Factories for positional arguments to `func`.
    **kwargs: Factories for keyword arguments to `func`.

  Returns:
    A `functools.partial` object.
  """
  args = [_ArgFactory(arg) for arg in args]
  kwargs = {name: _ArgFactory(arg) for name, arg in kwargs.items()}
  return functools.partial(_InvokeArgFactoryWrapper(func), *args, **kwargs)


def partialmethod(func, /, *args, **kwargs):
  """Version of `functools.partialmethod` that supports binds args to factories.

  `arg_factory.partialmethod` behaves like `arg_factory.partial`, except that it
  is designed to be used as a method definition rather than being directly
  callable.

  See `functools.partialmethod` and `arg_factory.partial` for more details.

  Args:
    func: The function whose arguments should be bound to factories.
    *args: Factories for positional arguments to `func`.
    **kwargs: Factories for keyword arguments to `func`.

  Returns:
    A `functools.partialmethod` object.
  """
  args = [_ArgFactory(arg) for arg in args]
  kwargs = {name: _ArgFactory(arg) for name, arg in kwargs.items()}
  return functools.partialmethod(
      _InvokeArgFactoryWrapper(func), *args, **kwargs)


class _ArgFactory:
  """A wrapper indicating that an argument should be computed with a factory.

  This wrapper is used by `arg_factory.partial` to define arguments whose
  value should should be built each time the partial is called.  It is also
  used in as the default value in the signatures for `arg_factory.partial`
  and `arg_factory.CallableWithDefaultFactory`.
  """
  _factory: Callable[[], Any]

  def __init__(self, factory: Callable[..., Any]):
    if not callable(factory):
      raise TypeError(
          'Expected arguments to `arg_factory.partial` to be callable.')
    self._factory = factory

  factory = property(lambda self: self._factory)

  def __repr__(self):
    name = getattr(self._factory, '__qualname__',
                   getattr(self._factory, '__name__', repr(self._factory)))
    return f'<built by: {name}()>'


def is_arg_factory_partial(partial_fn):
  """Returns True if `partial_fn` was constructed with `arg_factory.partial`."""
  return (isinstance(partial_fn,
                     (functools.partial, functools.partialmethod)) and
          isinstance(partial_fn.func, _InvokeArgFactoryWrapper))


@dataclasses.dataclass
class _InvokeArgFactoryWrapper:
  """Function wrapper that invokes the factories of ArgFactory args.

  When called, this wrapper transforms the arguments by replacing each
  `ArgFactory` argument `x` with `x.factory()`; and then calls the wrapped
  function with the transformed arguments.
  """

  func: Callable[..., Any]

  def __call__(self, /, *args, **kwargs):
    args = tuple(_arg_factory_value(arg) for arg in args)
    kwargs = {key: _arg_factory_value(arg) for (key, arg) in kwargs.items()}
    return self.func(*args, **kwargs)

  @property
  def __signature__(self):
    return inspect.signature(self.func)


def _arg_factory_value(value):
  return value.factory() if isinstance(value, _ArgFactory) else value


class CallableWithDefaultFactory(Generic[T]):
  """A function wrapper that uses default factories for function parameters.

  Each time the wrapper is called, any missing parameter with a default factory
  has its value filled in by calling the factory.

  `arg_factory.CallableWithDefaultFactory` is often created by the
  `@arg_factory.default_factory` decorator.  But it can also be useful
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
  >>> HybridCar = arg_factory.CallableWithDefaultFactory(
  ...     Car, engine=HybridEngine)
  >>> HybridCar()
  Car(engine=HybridEngine(capacity=2.0, torque=300))
  """
  _func: Callable[..., T]
  _default_factories: Dict[str, Callable[..., Any]]
  __signature__: inspect.Signature

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

  # If we see a parameter with a default, followed by a parameter without a
  # default, then the parameter without a default and all following parameters
  # must be keyword-only.  `seen_default` is the name of the most recent
  # parameter we've seen with a default value.
  seen_default = None
  keyword_only = False

  for param in func_signature.parameters.values():
    # Add the default factory for this parameter.
    factory = default_factories.get(param.name)
    if factory is not None:
      param = param.replace(default=_ArgFactory(factory))

    # Ensure that parameters without a default that follow parameters with
    # a default have kind=KEYWORD_ONLY.
    has_default = param.default is not inspect.Parameter.empty
    if has_default:
      seen_default = param.name
    elif seen_default is not None:
      keyword_only = True
    if keyword_only and param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
      param = param.replace(kind=inspect.Parameter.KEYWORD_ONLY)
    if keyword_only and param.kind == inspect.Parameter.POSITIONAL_ONLY:
      raise ValueError(
          f'After adding default factories for {set(default_factories)}, '
          f'the resulting function has a non-default parameter {param.name!r} '
          f'that follows a default parameter {seen_default!r}.  Please either '
          f'provide a default for {param.name!r} as well, or change '
          f'{param.name!r} to a keyword parameter.')

    parameters.append(param)
  return func_signature.replace(parameters=parameters)


def default_factory_for(
    **param_factories
) -> Callable[[Callable[..., T]], CallableWithDefaultFactory[T]]:
  """Decorator that defines default factories for a function's parameters.

  Each time the decorated function is called, any missing parameter with a
  default factory has its value filled in by calling the factory.  For example,
  this can be used to define mutable default values for functions, that are
  created each time the function is called:

  >>> @arg_factory.default_factory_for(seq=list)
  >>> def append(elt, seq: List[Any]):
  ...   seq.append(elt)
  ...   return seq

  This decorator can also be used in other contexts where the default value
  for a parameter should be computed each time the function is called.  E.g.:

  >>> @arg_factory.default_factory_for(noise=random.random)
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
