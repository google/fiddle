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

"""Unconstrained buildables."""

import dataclasses
from typing import Type, Any, Dict, Generic, TypeVar
from fiddle import config
from fiddle.experimental import daglish

T = TypeVar('T')


def memoized_post_traverse_without_path(fn, structure):
  """A version of memoized_traverse that doesn't pass in paths."""

  def wrap_fn(paths, value):
    del paths, value  # Unused.
    post_traversal_value = yield
    return fn(post_traversal_value)

  return daglish.memoized_traverse(wrap_fn, structure)


class UnconstrainedBuildable(Generic[T], config.Buildable[T]):
  """Buildable that acts as a generic container, with no constraints on args.

  Subclasses of `Buildable` are permitted to add restrictions to what values
  the callables' arguments may take.  For example, the `TaggedValue` type
  requires that the `tags` arguments be a `Collection[TagType]`.

  `UnconstrainedBuildable` can be used to temporarily relax these restrictions.
  In particular, the `to_unconstrained` function may be used to convert all
  buildables in a structure to `UnconstrainedBuildable` objects, which do not
  place any restrictions on the arguments.  The `from_unconstrained` function
  may then be used to convert `UnconstrainedBuildable`s back into `Buildable`s
  with the appropriate types.

  In addition to the `__fn_or_cls__` and `__arguments__` shared by all
  buildables, `UnconstrainedBuildable` also stores a `__buildable_type__` field,
  which records the type of the original Buildable.
  """
  __buildable_type__: Type[Any]

  def __init__(self, buildable_type: Type[config.Buildable],
               fn_or_cls: config.TypeOrCallableProducingT,
               arguments: Dict[str, Any]):
    if buildable_type.__flatten__ is not config.Buildable.__flatten__:
      raise ValueError(f'{buildable_type} is not supported because it has '
                       'a custom __flatten__ method.')
    object.__setattr__(self, '__buildable_type__', buildable_type)
    super().__init__(fn_or_cls, **arguments)

  def __build__(self, *args, **kwargs):
    raise ValueError('UnconstrainedBuildables may not be built.')

  def __flatten__(self):
    keys = tuple(self.__arguments__.keys())
    values = tuple(self.__arguments__.values())
    # pylint: disable=unexpected-keyword-arg
    metadata = UnconstrainedBuildableTraverserMetadata(
        fn_or_cls=self.__fn_or_cls__,
        argument_names=keys,
        buildable_type=self.__buildable_type__)
    return values, metadata

  def __path_elements__(self):
    return tuple(daglish.Attr(name) for name in self.__arguments__.keys())

  @classmethod
  def __unflatten__(cls, values, metadata):
    return cls(metadata.buildable_type, metadata.fn_or_cls,
               metadata.arguments(values))

  def __repr__(self):
    if hasattr(self.__fn_or_cls__, '__qualname__'):
      formatted_fn_or_cls = self.__fn_or_cls__.__qualname__
    else:
      formatted_fn_or_cls = str(self.__fn_or_cls__)
    formatted_params = [f'{k}={v!r}' for k, v in self.__arguments__.items()]
    name = self.__buildable_type__.__name__
    return (f'<Unconstrained {name}[{formatted_fn_or_cls}' +
            f"({', '.join(formatted_params)})]>")


def unconstrain_value(value):
  """Daglish traversal function that transforms Buildable->Unconstrained."""
  if isinstance(value, config.Buildable):
    return UnconstrainedBuildable(
        type(value), value.__fn_or_cls__, value.__arguments__)
  else:
    return value


def constrain_value(value):
  """Daglish traversal function that transforms Unconstrained->Buildable."""
  if isinstance(value, UnconstrainedBuildable):
    values, metadata = value.__flatten__()
    return value.__buildable_type__.__unflatten__(values, metadata)
  else:
    return value


def unconstrain_structure(structure):
  """Replaces any Buildable in structure with an UnconstrainedBuildable."""
  return memoized_post_traverse_without_path(unconstrain_value, structure)


def constrain_structure(structure):
  """Replaces any UnconstrainedBuildable in structure with a Buildable."""
  return memoized_post_traverse_without_path(constrain_value, structure)


@dataclasses.dataclass(frozen=True)
class UnconstrainedBuildableTraverserMetadata(config.BuildableTraverserMetadata
                                             ):
  """NodeTraverser metadata for UnconstrainedBuildable."""
  buildable_type: Type[config.Buildable]
