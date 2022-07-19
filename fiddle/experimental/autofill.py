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

"""Provides `autofill` to instantiate child configs for configurable arguments.

When configuring a type that depends on many other child types that are
themselves configurable, nested `fdl.Config` instances must be constructed for
each of them. This is can involve large amounts of mechanical boilerplate.

When a function or class has type annotations for its arguments, `autofill`
will recursively instantiate `fdl.Config`s for them.

`autofill` has the added benefit of further incentivizing type annotations, too!

By convention, `autofill` is often imported `as af` (which can also be
pronounced "auto-Fiddle").
"""

import dataclasses
import typing
from typing import Any, Type

from fiddle import config


def _is_autofillable(annotation: Type[Any]) -> bool:
  """A heuristic for when `fdl.Config`'s should be instantiated.

  This heuristic guesses that a nested `fdl.Config` should be instantiated
  if the type annotation corresponds to a dataclass or if the type has a
  `__init_fiddle__` defined.

  Args:
    annotation: The type annotated to an argument to a Buildable.

  Returns:
    True if the argument corresponding to this annotation should be autofilled,
    False otherwise.
  """
  return (dataclasses.is_dataclass(annotation) or
          hasattr(annotation, '__fiddle_init__'))


def autofill(cfg: config.Buildable):
  """Recursively instantiates `fdl.Config`'s for arguments.

  For any argument that does not already have an assigned value, an autofilled
  `fdl.Config` will be assigned if the argument's type annotation is either a
  dataclass or has a `__init_fiddle__` method. Arguments without a type
  annotation are never autofilled.

  Args:
    cfg: The root of the configuration to fill out. This Buildable is modified
      in-place.
  """
  # TODO: Explore adding a configurable policy argument.
  type_hints = typing.get_type_hints(cfg.__fn_or_cls__)
  for param in cfg.__signature__.parameters.values():

    if param.name in cfg.__arguments__:
      # Param already has a value.
      continue

    if param.annotation is param.empty:
      # No annotation; can't instantiate a config if you don't know the type!
      continue

    if param.default is not param.empty:
      # Param has a default; will just take that.
      continue

    if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
      # If the parameter is variable (vararg, varkwarg) or is positional-only,
      # it is not supported.
      continue

    if isinstance(param.annotation, str):
      annotation = type_hints[param.name]
    else:
      annotation = param.annotation
    if not _is_autofillable(annotation):
      # Policy doesn't allow for autobuilding.
      continue

    child = config.Config(annotation)
    autofill(child)
    setattr(cfg, param.name, child)
