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

"""Specialized version of `functools.partial` that supports arg factories."""
# TODO: Move this file out of Fiddle, into its own library.

import dataclasses
import functools
import inspect
from typing import Callable, Any


def partial(func, /, *args, **kwargs):
  """A specialized version of `functools.partial` that supports arg factories.

  `arg_factory.partial` is similar to `functools.partial`, in that it allows you
  to pre-bind positional and keyword arguments to values.  But
  `arg_factory.partial` also lets you pre-bind arguments to *factories*, which
  are used to construct value for an argument *each time the partial is called*.
  To pre-bind an argument to a factory, simply use the `arg_factory.ArgFactory`
  wrapper when binding the argument.

  Example:

  Consider the following code.

  >>> def f(x, noise): return x + noise
  >>> g = functools.partial(f, noise=random.random())
  >>> g(5) == g(5)  # noise has the same value for each call to `g`.
  True

  Here, the value for the `noise` parameter is set when the partial object is
  created, so each call to `g` will use the same value.  If we want a new
  value for each call to the partial, we can use `arg_factory.partial`:

  >>> g = arg_factory.partial(f, noise=ArgFactory(random.random))
  >>> g(5) == g(5)  # noise has a new value for each call to g.
  False

  Since we wrapped `random.random` in an `ArgFactory` wrapper, each call to
  `g` will call the factory, and so will use a different value for `noise`.

  Args:
    func: The function whose arguments should be pre-bound.
    *args: The positional arguments to `func`.
    **kwargs: The keyword argumetns to `func`.

  Returns:
    A partial object.
  """
  return functools.partial(_InvokeArgFactoryWrapper(func), *args, **kwargs)


def partialmethod(func, /, *args, **kwargs):
  """A version of `functools.partialmethod` that supports arg factories.

  `arg_factory.partialmethod` behaves like `arg_factory.partial`, except that it
  is designed to be used as a method definition rather than being directly
  callable.

  See `functools.partialmethod` and `arg_factory.partial` for more details.

  Args:
    func: The function whose arguments should be pre-bound.
    *args: The positional arguments to `func`.
    **kwargs: The keyword argumetns to `func`.

  Returns:
    A partialmethod object.
  """
  return functools.partialmethod(
      _InvokeArgFactoryWrapper(func), *args, **kwargs)


@dataclasses.dataclass(frozen=True)
class ArgFactory:
  """A wrapper indicating that an argument should be computed with a factory.

  This wrapper is used by `arg_factory.partial` to define arguments whose
  value should should be built each time the partial is called.
  """
  factory: Callable[[], Any]


def is_arg_factory_partial(partial_fn):
  """Returns true if `partial_fn` was constructed with `arg_factory.partial`."""
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
  return value.factory() if isinstance(value, ArgFactory) else value
