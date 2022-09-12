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

"""A registry-based system for constructing models.

## Skeletons and fixtures

Skeletons generally express a tree of modules; for example in an EncoderLayer,
one might construct Flax modules like so:

EncoderLayer
  attention: normalized_block()
    layer_norm: layer_norm()
    body: self_attention()
      wrapped: MultiHeadDotProductAttention()
    dropout: dropout()
  mlp: normalized_block()
    layer_norm: layer_norm()
    body: mlp()
      wrapped: MlpBlock()
    dropout: dropout()

and this becomes even more detailed when libraries are composed. In order to
mitigate some verbosity, skeletons allow these tree-like templates to be
indexed by the type of the module.

However, details of configuration, typically specific to a project like T5,
should be filled in later. We call this configuration a "fixture". Generally
"skeletons" should only make choices that are quite natural to the types
themselves; for example, if an EncoderLayer has an "attention" and "mlp" fields,
then those are almost always filled with normalized blocks containing attention
and MLP modules.

## Validators

Validators are another utility to separate concerns between module authors and
projects which may use those modules. A module author might know that their
MlpBlock() module must have a parameter `foo` specified whenever another
parameter `bar` is True, and it is convenient to express those invariants at
the level of the type (MlpBlock, in this case) rather than counting on each
project / experimenter to reproduce those checks.

## Global vs. local registries

In general, we are going to have a global type-based registry for constructing
"skeletons" of models, and then use functions that set configuration in a nested
manner. However, the underlying implementation here is rather unopinionated in
terms of how "global" its state is, so if specific projects find non-global
registries more useful, they can easily work with them.

TBD:

 * Examples with placeholders.
 * Add a function for validating configs by traversing the config tree.
"""

import collections
import dataclasses
import inspect
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union

from fiddle import config as config_lib
import typing_extensions

S = TypeVar("S")
T = TypeVar("T")

FnOrCls = Union[Type[S], Callable[..., S]]


class SkeletonFn(Generic[S], typing_extensions.Protocol):

  def __call__(self, cfg: config_lib.Config[S]) -> None:
    pass


class ValidatorFn(Generic[S], typing_extensions.Protocol):

  def __call__(self, cfg: config_lib.Config[S]) -> None:
    pass


def _is_configurable(fn_or_cls: FnOrCls) -> bool:
  """Helper method for determining wheher a function or class can be configured.

  This will return False for builtin stuff like int/tuple/etc. We generally
  don't want to configure these classes, partly because they work without
  requisite arguments, for example int() is 0.

  Args:
    fn_or_cls: Function or class.

  Returns:
    Whether `fn_or_cls` can be configured.
  """

  try:
    has_signature = inspect.signature(fn_or_cls) is not None
  except ValueError:
    has_signature = False

  return (has_signature  #
          and not inspect.isbuiltin(fn_or_cls)  #
          and not inspect.ismethod(fn_or_cls))


class DuplicateSkeletonError(ValueError):
  """Indicates that two skeletons were attempted to be registered for a type."""

  def __init__(self, fn_or_cls: FnOrCls[Any]):
    super().__init__(f"SkeletonFn for {fn_or_cls} already exists!")
    self.fn_or_cls = fn_or_cls

  def __str__(self):
    filename = inspect.getsourcefile(self.fn_or_cls)
    _, line_number = inspect.getsourcelines(self.fn_or_cls)
    return (f"SkeletonFn for {self.fn_or_cls} (defined at "
            f"{filename}:{line_number}) already exists!")


@dataclasses.dataclass
class TableEntry(Generic[T]):
  """An entry in the registry for a specific type or function.

  Attributes:
    skeleton_fn: A function which can create an initial configuration for this
      type or function.
    validators: A list of functions which can validate configuration objects of
      this type or function.
  """
  skeleton_fn: Optional[SkeletonFn[T]] = None
  validators: List[ValidatorFn[T]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Registry:
  """An AutoBuilder registry, which knows how to create a model skeleton.

  Attributes:
    table: A mapping from a type or function to configure, to a TableEntry.
  """
  table: Dict[FnOrCls[Any], TableEntry[Any]] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(TableEntry))

  def config(self,
             fn_or_cls: FnOrCls[T],
             require_skeleton: bool = True) -> config_lib.Config[T]:
    """Creates a configuration instance of a given function or class.

    Args:
      fn_or_cls: Function or class to get a skeleton fdl.Config instance for.
      require_skeleton: If True, a KeyError will be raised if a skeleton of the
        desired function/type is not available. If False, an empty fdl.Config of
        that function/class will be returned.

    Returns:
      Configuration instance.

    Raises:
      KeyError: If there is no skeleton registered for the given class, and
        require_skeleton is True (the default).
    """
    base_config = config_lib.Config(fn_or_cls)
    entry = self.table[fn_or_cls]
    if entry.skeleton_fn is None:
      if require_skeleton:
        raise KeyError(f"No skeleton registered for {fn_or_cls}.")
      else:
        return base_config
    entry.skeleton_fn(base_config)
    return base_config

  def skeleton(
      self, fn_or_cls: FnOrCls[T]) -> Callable[[SkeletonFn[T]], SkeletonFn[T]]:
    """Registers a function as a skeleton for a given type.

    Example:

    @dataclasses.dataclass
    class MyDense:
      activation: Callable[..., Any]

    @ab.skeleton(MyDense):
    def my_dense_skeleton(config: fdl.Config):
      config.activation = fdl.Config(nn.gelu, approximate=True)

    Often the skeleton definition (`my_dense_skeleton`) can be kept separately
    from the modeling library code, to make the modeling library code agnostic
    to the choice of configuration library.

    Args:
      fn_or_cls: Function that the skeleton is configuring.

    Returns:
      Decorator that, when applied to a function definition, will register that
      function as a skeleton.
    """

    def inner(skeleton_fn):
      entry = self.table[fn_or_cls]
      if entry.skeleton_fn is not None:
        raise DuplicateSkeletonError(fn_or_cls)
      entry.skeleton_fn = skeleton_fn
      return skeleton_fn

    return inner

  def auto_skeleton(self,
                    fn_or_cls: FnOrCls[Any],
                    /,
                    attributes_require_skeletons: bool = False,
                    **kwargs) -> None:
    """Registers a skeleton based on arg types.

    Sometimes functions are written to take a more generic class, for example
    one might have an encoder layer in Flax expressed like,

    class EncoderLayer(nn.Module):
      dropout: nn.Module

    but in practice, `dropout` is almost always `nn.Dropout`, the type signature
    is just left to be the most general thing in case of experimental overrides.

    In this case, we can specify that the dropout is `nn.Dropout` as follows:

    ab.auto_skeleton(EncoderLayer, dropout=nn.Dropout)

    This is equivalent to writing:

    @ab.skeleton(EncoderLayer)
    def encoder_layer_skeleton(config: fdl.Config[EncoderLayer]):
      config.dropout = ab.config(nn.Dropout, require_skeleton=False)

    By default, `auto_skeleton` doesn't require the attributes to have defined
    skeletons, but would use them if available. In this case, the `dropout`
    class attribute can be filled with a blank `fdl.Config(nn.Dropout)`, if
    there is not a skeleton for `nn.Dropout`.

    If you require more advanced skeleton setups, then please use the `skeleton`
    method.

    Args:
      fn_or_cls: Function or class to generate the skeleton for.
      attributes_require_skeletons: Whether skeletons are required for any class
        attributes or function arguments to `fn_or_cls`. By default, skeletons
        are not required, whereupon empty fdl.Config instances are created.
      **kwargs: Any override types to use for configuration.

    Raises:
      ValueError: If there is no annotation for a parameter not mentioned in
        **kwargs.
      TypeError: If one of the parameters not mentioned in **kwargs is not a
        configurable type.
    """
    arg_to_type = {}
    for name, parameter in inspect.signature(fn_or_cls).parameters.items():
      if name in kwargs:
        # None is a special value that indicates no type should be configured
        # for the parameter.
        if kwargs[name] is not None:
          arg_to_type[name] = kwargs[name]
      elif parameter.annotation == inspect.Signature.empty:
        raise ValueError(
            f"Parameter {name!r} of {fn_or_cls} doesn't have an annotation, "
            "and no override type was set in the auto_skeleton decorator.")
      elif not _is_configurable(parameter.annotation):
        raise TypeError(
            f"Parameter {name!r} of {fn_or_cls} is of type "
            f"{parameter.annotation}, which cannot be configured. Please pass "
            f"{name}=None to your auto-skeleton invocation to avoid "
            "automatically adding a Config for this parameter.")
      else:
        arg_to_type[name] = parameter.annotation

    def arg_type_skeleton(cfg: config_lib.Config):
      for name, type_to_configure in arg_to_type.items():
        setattr(
            cfg, name,
            self.config(
                type_to_configure,
                require_skeleton=attributes_require_skeletons))

    self.skeleton(fn_or_cls)(arg_type_skeleton)

  def validator(
      self,
      fn_or_cls: FnOrCls[T]) -> Callable[[ValidatorFn[T]], ValidatorFn[T]]:
    """Decorator to register a function as a validator for a given type.

    Args:
      fn_or_cls: Function that the validator is checking.

    Returns:
      Decorator that, when applied to a function definition, will register that
      function as a validator.
    """

    def inner(validator_fn: ValidatorFn[T]) -> ValidatorFn[T]:
      self.table[fn_or_cls].validators.append(validator_fn)
      return validator_fn

    return inner


# By default, the global instance should be used. Its methods are aliased to the
# module level.
_default_registry = Registry()
config = _default_registry.config
skeleton = _default_registry.skeleton
auto_skeleton = _default_registry.auto_skeleton
validator = _default_registry.validator
