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
from typing import Any, Callable, Dict, List, Optional, Type, Union

import fiddle as fdl

FnOrCls = Union[Type[Any], Callable[..., Any]]
Skeleton = Callable[[fdl.Config], None]  # Add generics vars when Fiddle does.
Validator = Callable[[fdl.Config], None]


class DuplicateSkeletonError(ValueError):
  """Indicates that two skeletons were attempted to be registered for a type."""

  def __init__(self, fn_or_cls: FnOrCls):
    super().__init__(f"Skeleton for {fn_or_cls} already exists!")
    self.fn_or_cls = fn_or_cls

  def __str__(self):
    filename = inspect.getsourcefile(self.fn_or_cls)
    _, line_number = inspect.getsourcelines(self.fn_or_cls)
    return (f"Skeleton for {self.fn_or_cls} (defined at "
            f"{filename}:{line_number}) already exists!")


@dataclasses.dataclass
class TableEntry:
  """An entry in the registry for a specific type or function.

  Attributes:
    skeleton_fn: A function which can create an initial configuration for this
      type or function.
    validators: A list of functions which can validate configuration objects of
      this type or function.
  """
  skeleton_fn: Optional[Skeleton] = None
  validators: List[Validator] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Registry:
  """An AutoBuilder registry, which knows how to create a model skeleton.

  Attributes:
    table: A mapping from a type or function to configure, to a TableEntry.
  """
  table: Dict[FnOrCls, TableEntry] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(TableEntry))

  def config(self, fn_or_cls: FnOrCls) -> fdl.Config:
    """Creates a configuration instance of a given function or class.

    Args:
      fn_or_cls: Function or class to get a skeleton fdl.Config instance for.

    Returns:
      Configuration instance.

    Raises:
      KeyError: If there is no skeleton registered for the given class.
    """
    base_config = fdl.Config(fn_or_cls)
    entry = self.table[fn_or_cls]
    if entry.skeleton_fn is None:
      raise KeyError(f"No skeleton registered for {fn_or_cls}.")
    entry.skeleton_fn(base_config)
    return base_config

  def skeleton(self, fn_or_cls) -> Callable[[Skeleton], Skeleton]:
    """Decorator to register a function as a skeleton for a given type.

    Example:

    @dataclasses.dataclass
    class MyDense:
      activation: Callable[..., Any]

    @ab.skeleton(MyDense):
    def my_dense_skeleton(config: fdl.Config):
      config.activation = fdl.Config(nn.gelu, approximate=True)

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

  def validator(self, fn_or_cls) -> Callable[[Validator], Validator]:
    """Decorator to register a function as a validator for a given type.

    Args:
      fn_or_cls: Function that the validator is checking.

    Returns:
      Decorator that, when applied to a function definition, will register that
      function as a validator.
    """

    def inner(validator_fn: Validator) -> Validator:
      self.table[fn_or_cls].validators.append(validator_fn)
      return validator_fn

    return inner


# By default, the global instance should be used. Its methods are aliased to the
# module level.
_default_registry = Registry()
config = _default_registry.config
skeleton = _default_registry.skeleton
validator = _default_registry.validator
