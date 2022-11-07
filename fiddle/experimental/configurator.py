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

"""A midlevel API for creating configs, between core APIs and auto_config.

This API is a Fiddle version of
https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html .

WARNING: THIS API IS EXPERIMENTAL AND MAY BE DELETED. Please use auto_config
right now for non-experimental projects, or fork this file.
"""

import functools
import typing
from typing import TypeVar, Union

from fiddle import config as config_lib

FunctionOrClassT = TypeVar("FunctionOrClassT")
ValueT = TypeVar("ValueT")


class _Configurator:
  """Dummy class used for syntactic reasons."""

  def __call__(self, fn_or_cls: FunctionOrClassT) -> FunctionOrClassT:
    """Lift a callable to one that will create a config when called.

    Usually, this is aliased in a module calling it, e.g.

    C = configurator.configurator

    def fixture():
      foo = C(Foo)(a=1, b=4)
      bar = C(Bar)(foo=7)
      return C.cast_to_config(bar)

    Args:
      fn_or_cls: Function or class to "lift".

    Returns:
      A method that, when called with the same args `fn_or_cls` is expecting,
      will return a fdl.Config(fn_or_cls, <args>).
    """

    @functools.wraps(fn_or_cls)
    def inner(*args, **kwargs):
      return config_lib.Config(fn_or_cls, *args, **kwargs)

    return typing.cast(FunctionOrClassT, inner)

  @staticmethod
  def cast_to_config(value: ValueT) -> config_lib.Config[ValueT]:
    """Corrects the type of a value that was from a configurator call.

    As mentioned in the main call method, we lie about the type returned by
    C(Foo) a bit, in order to get the benefits of type checking, and because
    that value may be returned to other functions.

    But when a fixture returns, it may be more helpful to have the real type of
    an object, which is normally a fdl.Config.

    Args:
      value: Value to cast, which should be from a call to C(Foo), but we don't
        actually check it.

    Returns:
      Type-cast version.
    """
    if not isinstance(value, config_lib.Config):
      raise TypeError(f"Expected a Config but got a {type(value)}.")
    return typing.cast(config_lib.Config[ValueT], value)

  @staticmethod
  def cast_to_permissive_type(
      value: ValueT) -> Union[ValueT, config_lib.Config[ValueT]]:
    """Variant of cast_to_config that can be plugged into C() calls.

    For example, if you have a base fixture

    def base():
      foo = C(Foo)(a=1, b=4)
      return C.cast_to_permissive_type(foo)

    and a parameterized fixture

    def parameterized(foo: Foo):
      bar = C(Bar)(foo=foo)
      return C.cast_to_config(bar)

    then this typing allows us to call parameterized(base()) and also use
    base() when we want to deal with its Fiddle config output.

    Please also see the cast_to_config() docstring.

    Args:
      value: Value to cast.

    Returns:
      Type-cast version.
    """
    return typing.cast(Union[ValueT, config_lib.Config[ValueT]], value)


configurator: _Configurator = _Configurator()
