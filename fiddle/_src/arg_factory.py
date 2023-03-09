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

  * `@arg_factory.supply_defaults` is a decorator that defines a "default
    factory" for one or more parameters.  Each time the decorated function
    is called, any missing parameter with a default factory has its value
    filled in by calling the factory.  For example, this can be used to define
    mutable default values for functions, that are created each time the
    function is called.

  * `arg_factory.partial` is a specialized version of `functools.partial`
    that binds parameters to factories rather than values.  These factories
    are used to compute the parameter values each time the partial function
    is called.

## Differences between `supply_defaults` and `partial`

These two wrappers have similar behavior for keyword arguments, but differ in
how they handle factories for positional arguments:

  * When `arg_factory.partial` binds an argument positionally, that argument is
    removed from the signature.  I.e., the wrapper function no longer expects
    that argument.

  * In contrast, `@arg_factory.supply_defaults` does not remove any
    arguments from the signature of the wrapped function.

The wrappers also differ in their restrictions for positional-only
arguments.  Consider a function with signature `f(x, y, /)`:

  * `arg_factory.partial` may be used to define a factory for
    `x` without defining one for `y` (and doing so will "consume" the argument
    `x`); but `@arg_factory.supply_defaults` can not (since non-default
    arguments may not follow default arguments).

  * Conversely, `@arg_factory.supply_defaults` may be used to define a
    factory for `y` without defining one for `x` (and doing so will make `y`
    an optional argument); but `arg_factory.partial` can not (since `y` is a
    positional-only argument, so there's no way to bind it without also binding
    `x`).

A final difference is that `@arg_factory.supply_defaults` may only be used
to provide default factories for positional and keyword parameters in the
wrapped function's signature; it may not be used to add factories for functions
with var-positional (`*args`) or var-keyword (**kwargs`) parameters. But
`arg_factory.partial` *can* be used to provide a default for a var-positional or
var-keyword argument.  E.g.:

>>> def f(**kwargs):
...   print(kwargs)
>>> p = arg_factory.partial(f, x=list)
>>> p(y=3)
{'x': [], y: 3}
"""

from __future__ import annotations

# TODO(b/272074307): Move this file out of Fiddle, into its own library.

import dataclasses
import functools
import inspect
from typing import Any, Callable, TypeVar, overload

T = TypeVar('T')


def partial(*args, **kwargs):
  """A specialized version of `functools.partial` that binds args to factories.

  `arg_factory.partial` is similar to `functools.partial`, except that it binds
  arguments to *factory functions* rather than *values*.  These factory
  functions are used to construct a value for an argument each time the partial
  is called.

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

  If you need to define a partial function that specifies some parameters using
  values and other parameters using factories, then you should do so by
  composing `functools.partial` with `arg_factory.partial`.  E.g.:

  >>> h = functools.partial(arg_factory.partial(f, noise=random.random), x=4)

  Args:
    *args: The function whose arguments should be bound to factories, followed
      by factories for positional arguments to `func`.
    **kwargs: Factories for keyword arguments to `func`.

  Returns:
    A `functools.partial` object.
  """
  func, *args = args
  args = [ArgFactory(arg) for arg in args]
  kwargs = {name: ArgFactory(arg) for name, arg in kwargs.items()}
  return functools.partial(_InvokeArgFactoryWrapper(func), *args, **kwargs)


def partialmethod(*args, **kwargs):
  """Version of `functools.partialmethod` that supports binds args to factories.

  `arg_factory.partialmethod` behaves like `arg_factory.partial`, except that it
  is designed to be used as a method definition rather than being directly
  callable.

  See `functools.partialmethod` and `arg_factory.partial` for more details.

  Args:
    *args: The function whose arguments should be bound to factories, followed
      by factories for positional arguments to `func`.
    **kwargs: Factories for keyword arguments to `func`.

  Returns:
    A `functools.partialmethod` object.
  """
  func, *args = args
  args = [ArgFactory(arg) for arg in args]
  kwargs = {name: ArgFactory(arg) for name, arg in kwargs.items()}
  return functools.partialmethod(
      _InvokeArgFactoryWrapper(func), *args, **kwargs)


def _raise_unsupported_op_error(factory, op_type, error_type=ValueError):
  """Raises a ValueError for an ArgFactory operation that is not supported."""
  name = getattr(
      factory, '__qualname__', getattr(factory, '__name__', repr(factory))
  )
  raise error_type(
      f'arg_factory.default_factory({name}) does not support {op_type}.\n'
      '`arg_factory.default_factory(...)` should only be used as a default '
      'value for a function wrapped with `arg_factory.supply_defaults`; or as '
      'an argument to `arg_factory.partial`.\n'
      'Did you forget to apply the `@arg_factory.supply_defaults` decorator?'
  )


def _unsupported(op_type):
  """Returns a function that calls _raise_unsupported_op_error."""

  def unsupported_op_handler(arg_factory, *args, **kwargs):
    del args, kwargs  # unused
    factory = arg_factory._factory  # pylint: disable=protected-access
    _raise_unsupported_op_error(factory, op_type)

  return unsupported_op_handler


class ArgFactory:
  """A wrapper indicating that an argument should be computed with a factory.

  This wrapper is used by `arg_factory.partial` to define arguments whose
  value should should be built each time the partial is called.  It is also
  used in as the default value in the signatures for `arg_factory.partial`
  and `arg_factory.supply_defaults`.
  """

  _factory: Callable[..., Any]

  # Type overload to make sure pytype is happy for expressions such as
  # `def fn(x: list = ArgFactory(list)): ...`, where the actual return type
  # of `ArgFactory(list)` is `ArgFactory`, but the declared type is `list`.
  # (Without the overload, we'd get an annotation-type-mismatch error.)
  @overload
  def __new__(cls, factory: Callable[..., T]) -> T:
    ...

  def __new__(cls, *args, **kwargs):
    del args, kwargs  # Unused
    return super().__new__(cls)

  def __init__(self, factory: Callable[..., Any]):
    if not callable(factory):
      raise TypeError(
          'Expected arguments to `arg_factory.partial` to be callable. '
          f'Got {factory!r}.')
    super().__setattr__('_factory', factory)

  factory = property(lambda self: self._factory)

  def __repr__(self):
    name = getattr(self._factory, '__qualname__',
                   getattr(self._factory, '__name__', repr(self._factory)))
    return f'<built by: {name}()>'

  def __eq__(self, other):
    return isinstance(other, ArgFactory) and self._factory == other._factory

  def __hash__(self):
    return hash(self._factory)

  # The following overrides are intended to provide useful error messages if
  # the user uses `default_factory` but forgets to use the `@supply_defaults`
  # decorator.  In that case, the function will be called with an `ArgFactory`
  # object (rather than the result of calling the factory).
  #
  # We override math operations, attribute operations, container operations,
  # and the calling operation to raise an exception with a helpful error
  # message. We intentionally do *not* override __eq__ or __hash__, because we
  # don't want to break code that handles function signatures generically.
  __ge__ = _unsupported('math operations')
  __gt__ = _unsupported('math operations')
  __le__ = _unsupported('math operations')
  __lt__ = _unsupported('math operations')
  __add__ = _unsupported('math operations')
  __and__ = _unsupported('math operations')
  __divmod__ = _unsupported('math operations')
  __floordiv__ = _unsupported('math operations')
  __lshift__ = _unsupported('math operations')
  __matmul__ = _unsupported('math operations')
  __mod__ = _unsupported('math operations')
  __mul__ = _unsupported('math operations')
  __or__ = _unsupported('math operations')
  __pow__ = _unsupported('math operations')
  __rshift__ = _unsupported('math operations')
  __sub__ = _unsupported('math operations')
  __truediv__ = _unsupported('math operations')
  __xor__ = _unsupported('math operations')
  __radd__ = _unsupported('math operations')
  __rand__ = _unsupported('math operations')
  __rdiv__ = _unsupported('math operations')
  __rdivmod__ = _unsupported('math operations')
  __rfloordiv__ = _unsupported('math operations')
  __rlshift__ = _unsupported('math operations')
  __rmatmul__ = _unsupported('math operations')
  __rmod__ = _unsupported('math operations')
  __rmul__ = _unsupported('math operations')
  __ror__ = _unsupported('math operations')
  __rpow__ = _unsupported('math operations')
  __rrshift__ = _unsupported('math operations')
  __rsub__ = _unsupported('math operations')
  __rtruediv__ = _unsupported('math operations')
  __rxor__ = _unsupported('math operations')
  __iadd__ = _unsupported('math operations')
  __iand__ = _unsupported('math operations')
  __ifloordiv__ = _unsupported('math operations')
  __ilshift__ = _unsupported('math operations')
  __imatmul__ = _unsupported('math operations')
  __imod__ = _unsupported('math operations')
  __imul__ = _unsupported('math operations')
  __ior__ = _unsupported('math operations')
  __ipow__ = _unsupported('math operations')
  __irshift__ = _unsupported('math operations')
  __isub__ = _unsupported('math operations')
  __itruediv__ = _unsupported('math operations')
  __ixor__ = _unsupported('math operations')
  __abs__ = _unsupported('math operations')
  __neg__ = _unsupported('math operations')
  __pos__ = _unsupported('math operations')
  __invert__ = _unsupported('math operations')
  __trunc__ = _unsupported('math operations')
  __floor__ = _unsupported('math operations')
  __ceil__ = _unsupported('math operations')
  __round__ = _unsupported('math operations')
  __int__ = _unsupported('math operations')
  __bool__ = _unsupported('math operations')
  __nonzero__ = _unsupported('math operations')
  __complex__ = _unsupported('math operations')
  __float__ = _unsupported('math operations')
  __iter__ = _unsupported('iteration')
  __next__ = _unsupported('iteration')
  __len__ = _unsupported('len')
  __reversed__ = _unsupported('reversed')
  __contains__ = _unsupported('contains')
  __call__ = _unsupported('calling')
  __getitem__ = _unsupported('getitem')
  __setitem__ = _unsupported('setitem')
  __delitem__ = _unsupported('delitem')

  def __getattr__(self, name):
    # self._factory may not exist yet when handling copy.deepcopy.
    factory = Ellipsis if name == '_factory' else self._factory
    _raise_unsupported_op_error(
        factory, f'the attribute {name}', AttributeError
    )

  def __setattr__(self, name, value):
    _raise_unsupported_op_error(self._factory, 'setting attributes')

  def __delattr__(self, name):
    _raise_unsupported_op_error(self._factory, f'the attribute {name}')


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

  def __call__(self, *args, **kwargs):
    args = tuple(_arg_factory_value(arg) for arg in args)
    kwargs = {key: _arg_factory_value(arg) for (key, arg) in kwargs.items()}
    return self.func(*args, **kwargs)

  @property
  def __signature__(self):
    return inspect.signature(self.func)


def _arg_factory_value(value):
  return value.factory() if isinstance(value, ArgFactory) else value


def supply_defaults(wrapped_func):
  """Decorator that defines default factories for a function's parameters.

  Each time the decorated function is called, any missing parameter with a
  default factory has its value filled in by calling the factory.  Default
  factories are specified by setting the default value of a parameter to
  `arg_factory.default_factory(<factory>)`.

  For example, this can be used to define mutable default values for functions,
  that are created each time the function is called:

  >>> @arg_factory.supply_defaults
  ... def append(elt, seq: list[Any] = arg_factory.default_factory(list)):
  ...   seq.append(elt)
  ...   return seq

  This decorator can also be used in other contexts where the default value
  for a parameter should be computed each time the function is called.  E.g.:

  >>> @arg_factory.supply_defaults
  ... def add_noise(value, noise = arg_factory.default_factory(random.random)):
  ...   return value + noise**2

  This decorator uses `functools.wraps` to copy fields such as `__name__` and
  `__doc__` from the wrapped function to the wrapper function.

  Args:
    wrapped_func: The function that should be decorated.

  Returns:
    The decorated function.
  """
  signature = inspect.signature(wrapped_func)
  factories = {
      param.name: param.default.factory
      for param in signature.parameters.values()
      if isinstance(param.default, ArgFactory)
  }
  if not factories:
    raise ValueError(
        '@supply_defaults expected at least one argument to '
        'have a default value constructed with default_factory.'
    )

  @functools.wraps(wrapped_func)
  def wrapper(*args, **kwargs):
    bound_args = signature.bind(*args, **kwargs)
    for name, factory in factories.items():
      if name not in bound_args.arguments:
        bound_args.arguments[name] = factory()
    return wrapped_func(*bound_args.args, **bound_args.kwargs)

  return wrapper


# Type overload to make sure pytype is happy for expressions such as
# `x: list = default_factory(list)`, where the actual return type of
# `default_factory` is `ArgFactory`, but the declared type is `list`.
# (Without the overload, we'd get an annotation-type-mismatch error.)
@overload
def default_factory(factory: Callable[..., T]) -> T:
  ...


def default_factory(factory: Callable[..., Any]) -> ArgFactory:
  """Returns a value used to declare the default factory for a parameter.

  `default_factory` should be used in conjuction with the `@supply_defaults`
  decorator to declare default factories for a function's parameters.  See
  the documentation for `supply_defaults` for more information.

  Args:
    factory: The factory that should be used to construct the value for a
      parameter.  This factory is called each time the function is called.
  """
  return ArgFactory(factory)
