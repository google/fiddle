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

# pylint: disable=unused-import

from fiddle._src.arg_factory import ArgFactory
from fiddle._src.arg_factory import default_factory
from fiddle._src.arg_factory import is_arg_factory_partial
from fiddle._src.arg_factory import partial
from fiddle._src.arg_factory import partialmethod
from fiddle._src.arg_factory import supply_defaults
