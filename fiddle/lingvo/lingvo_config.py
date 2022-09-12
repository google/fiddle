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

"""Fiddle support for types using Lingvo's Params types.

Lingvo's Params types use a somewhat similar pattern to Fiddle Buildables. The
lingvo_params_integration allows you to completely mix-and-match Lingvo types
with arbitrary other Fiddle types. Instead of instantiating a fdl.Config,
instead instantiate a fdl.lingvo.LingvoConfig. You can use the LingvoConfig
identically to a fdl.Config type, including passing it to fdl.build to create
an instance of the type.

Example:

```py

import fiddle as fdl

class ClassA(BaseClass):

  @classmethod
  def Params(cls):  # pylint: disable=invalid-name
    p = super().Params()
    p.Define('a', -1, 'The "a" thing!')
    return p

  def __init__(self, params):
    super().__init__(params)
    self.a = params.a

def main():
  cfg = fdl.lingvo.LingvoConfig(ClassA)
  cfg.a = 42

  obj: ClassA = fdl.build(cfg)
  print(obj.a)  # prints 42
```

You can nest fdl.Config's and fdl.lingvo.LingvoConfig's arbitrarily.
"""

import inspect
import typing
from typing import Any, Iterable, Union

from fiddle import config
from lingvo.core import hyperparams
import typing_extensions


class ParamInitable(typing_extensions.Protocol):
  """A protocol for Param-initializable classes."""

  @classmethod
  def Params(cls) -> hyperparams.Params:
    pass

  def __init__(self, params: hyperparams.Params):
    pass


class LingvoParamsAdapter:
  """Adapts a hyperparams.Params instance for use with fdl.Config.

  fdl.Config works with callables, and inspect.Signature will look for a
  `__signature__` property when reflecting on an object. This type takes an
  hyperparams.Params instance and builds an `inspect.Signature` to be stored at
  `self.__signature__`. This wrapped object can then be used with most of the
  `fdl.Config` type entirely unmodified. The wrapped object's `__call__` method
  sets a copy of the Params object with the user's set values and then returns
  the result of the `.Instantiate()` call, mimicking the normal constructor
  behavior from a regular type.

  Attributes:
    params: The wrapped Params instance.
    is_nested: `False` iff the Params instance is the root of a Lingvo config
      tree. (There may be multiple Lingvo config trees within a larger Fiddle
      configuration.)
    __signature__: The inferred signature of a call loosely corresponding to the
      parameters defined in `hyperparams.Params`.
  """
  params: hyperparams.Params
  is_nested: bool
  __signature__: inspect.Signature

  def __init__(self, cls_or_params: Union[ParamInitable, hyperparams.Params]):
    if isinstance(cls_or_params, hyperparams.Params):
      self.params = cls_or_params
      self.is_nested = True
    else:
      self.params = cls_or_params.Params()
      self.is_nested = False
    self.__signature__ = _make_signature_from_lingvo_params(self.params)

  def __call__(self, /, *args, **kwargs):
    if args:
      raise AssertionError(
          f"*args not empty, please file a bug with repro steps; args: {args}.")
    param_copy = self.params.Copy()  # Always make a copy!
    for arg in kwargs:
      setattr(param_copy, arg, kwargs[arg])
    if self.is_nested:
      return param_copy
    return param_copy.Instantiate()

  def __repr__(self):
    if hasattr(self.params, "cls"):
      return f"<LingvoParamsAdapter({self.params.cls.__qualname__})>"
    return "<LingvoParamsAdapter(<unknown class>)"


class LingvoConfig(config.Config):
  """A Config instance specialized for use with LingvoParam-based types.

  For additional context, see the docstring associated with
  `LingvoParamsAdapter`.
  """

  def __init__(self, params_or_cls: Union["LingvoConfig", LingvoParamsAdapter,
                                          ParamInitable, hyperparams.Params], /,
               *args, **kwargs):
    if (isinstance(params_or_cls, LingvoParamsAdapter) or
        isinstance(params_or_cls, LingvoConfig)):
      # `super().__init__` does the right thing.
      pass
    else:
      params_or_cls = LingvoParamsAdapter(params_or_cls)
    super().__init__(params_or_cls, *args, **kwargs)
    # Must set all param values and not leave them as defaults.
    params = typing.cast(LingvoParamsAdapter, self.__fn_or_cls__).params
    for param_name in params.GetKeys():
      if param_name not in self.__arguments__:
        setattr(self, param_name, params.Get(param_name))

  def __setattr__(self, name: str, value: Any):
    if isinstance(value, hyperparams.Params):
      value = LingvoConfig(value)
    # TODO: what about params hidden inside containers?
    super().__setattr__(name, value)

  @classmethod
  def __unflatten__(
      cls,
      values: Iterable[Any],
      metadata: config.BuildableTraverserMetadata,
  ) -> "LingvoConfig":
    params_adapter = typing.cast(LingvoParamsAdapter, metadata.fn_or_cls)
    return cls(params_adapter, **metadata.arguments(values))


def _convert_lingvo_param(
    name: str,
    param: hyperparams._Param,  # pylint: disable=protected-access
) -> inspect.Parameter:
  """Returns an inspect.Parameter that corresponds to `param`."""
  return inspect.Parameter(
      name=name,
      kind=inspect.Parameter.KEYWORD_ONLY,
      default=param.GetDefault())


def _make_signature_from_lingvo_params(
    params: hyperparams.Params) -> inspect.Signature:
  """Makes a signature from a lingvo Params instance."""
  parameters = [
      _convert_lingvo_param(name, param)
      for name, param in params._params.items()  # pylint: disable=protected-access
  ]
  return inspect.Signature(parameters)
