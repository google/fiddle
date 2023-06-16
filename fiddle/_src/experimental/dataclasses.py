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

"""Leverage Python's dataclasses' extensibility for lightweight Fiddle control.

While Fiddle is carefully designed to be used with libraries that are completely
ignorant of Fiddle's existance, Fiddle also allows library authors (if they so
choose) to customize Fiddle's behavior to make great user experiences for the
library's users. Fiddle leverages extension mechanisms within Python's
dataclasses system to add some additional metadata to control behavior in the
rest of Fiddle.
"""

import dataclasses
import types
from typing import Any, Collection, Mapping, Optional, Type, Union

from fiddle._src import config
from fiddle._src import daglish
from fiddle._src import field_metadata
from fiddle._src import signatures
from fiddle._src import tag_type
from fiddle._src.experimental import auto_config


TagOrTags = Union[tag_type.TagType, Collection[tag_type.TagType]]

FieldMetadata = field_metadata.FieldMetadata


def field(
    *,
    default_factory: Any = dataclasses.MISSING,
    tags: TagOrTags = (),
    metadata: Optional[Mapping[Any, Any]] = None,
    configurable_factory: bool = False,
    **kwargs,
) -> Union[dataclasses.Field, Any]:  # pylint: disable=g-bare-generic
  """A wrapper around dataclasses.field to add optional Fiddle metadata.

  Args:
    default_factory: This has the same meaning as
      `dataclasses.fields.default_factory`, with the addition that if it's an
      `@auto_config`'d function, then the `as_buildable` will be used to
      initalize this field when creating a `fdl.Buildable` for the enclosing
      type.
    tags: One or more tags to attach to the `fdl.Buildable`'s argument
      corresponding to the field.
    metadata: Any additional metadata to include.
    configurable_factory: If true, then set this field to
      `fdl.Config(default_factory)` when creating a `fdl.Buildable` for the
      enclosing type.  For example, if `default_factory` is a dataclass, then
      this will make it possible to configure default values for the fields of
      that dataclass.  This should not be set to True if `default_factory` is an
      `auto_config`'ed function; see above for handling of `auto_config'ed
      `default_factory`.
    **kwargs: All other kwargs are passed to `dataclasses.field`; see the
      documentation on `dataclasses.field` for valid arguments.

  Returns:
    The result of calling dataclasses.field. Note: the return type is marked as
    a union with Any to pass static typechecking at usage sites.
  """
  # TODO(b/272374473): Make a function to return a metadata object to users to
  # enable them to call `dataclasses.field` themselves.
  if isinstance(tags, tag_type.TagType):
    tags = (tags,)

  if auto_config.is_auto_config(default_factory):
    if configurable_factory:
      raise ValueError("configurable_factory should not be used with "
                       "auto_config'ed functions.")
    buildable_initializer = default_factory.as_buildable
  elif configurable_factory:
    if not (default_factory and signatures.has_signature(default_factory)):
      raise ValueError("configurable_factory requires that default_factory "
                       "be set to a function or class with a signature.")
    buildable_initializer = lambda: config.Config(default_factory)
  else:
    buildable_initializer = None

  metadata: Mapping[Any, Any] = types.MappingProxyType(metadata or {})
  # pylint: disable=protected-access
  metadata = {
      **metadata, field_metadata._FIDDLE_DATACLASS_METADATA_KEY:
          FieldMetadata(tags=tags, buildable_initializer=buildable_initializer)
  }
  # pylint: enable=protected-access
  return dataclasses.field(
      default_factory=default_factory, metadata=metadata, **kwargs)


def field_has_tag(
    dc_field: dataclasses.Field,  # pylint: disable=g-bare-generic
    tag: tag_type.TagType,
) -> bool:
  """Returns True if buildables will attach `tag` to the corresponding arg.

  In particular, `field_has_tag(field(..., tags=tags), tag)` is True if
  `tag in tags`.

  Args:
    dc_field: A dataclass field, describing an argument for a dataclass.
    tag: The tag that should be checked.
  """
  metadata = field_metadata.field_metadata(dc_field)
  return metadata is not None and tag in metadata.tags


def _names(value_or_type):
  return [field.name for field in dataclasses.fields(value_or_type)]


def _flatten_dataclass(value):
  names = _names(value)
  return [getattr(value, name) for name in names], (type(value), names)


def _unflatten_dataclass(values, metadata):
  typ, names = metadata
  assert len(names) == len(values)
  values_dict = {name: value for name, value in zip(names, values)}
  return typ(**values_dict)


def _path_elements_fn(value):
  return tuple(daglish.Attr(name) for name in _names(value))


dataclass_traverser = daglish.NodeTraverser(
    flatten=_flatten_dataclass,
    unflatten=_unflatten_dataclass,
    path_elements=_path_elements_fn,
)


class DataclassTraverserRegistry(daglish.NodeTraverserRegistry):

  def find_node_traverser(
      self, node_type: Type[Any]
  ) -> Optional[daglish.NodeTraverser]:
    traverser = self.fallback_registry.find_node_traverser(node_type)
    if traverser is None and dataclasses.is_dataclass(node_type):
      traverser = dataclass_traverser
    return traverser


daglish_dataclass_registry = DataclassTraverserRegistry(use_fallback=True)


def convert_dataclasses_to_configs(
    root: Any, allow_post_init: bool = False
) -> Any:
  """Converts a dataclass or nested structure of dataclasses to Fiddle configs.

  Args:
    root: Root dataclass instance or structure (lists, tuples, dicts, etc.) of
      dataclasses.
    allow_post_init: Allow conversion of dataclasses with a __post_init__
      method.

  Returns:
    Fiddle config that, when built, should return an object equal to `root`
    (presuming `root` does not already contain Fiddle configs).
  """

  def traverse(value, state: daglish.State):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
      if hasattr(type(value), "__post_init__") and not allow_post_init:
        raise ValueError(
            "Dataclasses with __post_init__ are not allowed, "
            "unless allow_post_init is set to True."
        )
      value = config.Config(
          type(value),
          **{
              field.name: getattr(value, field.name)
              for field in dataclasses.fields(value)
              if field.init
          },
      )
    return state.map_children(value)

  state = daglish.MemoizedTraversal(
      traverse, root, daglish_dataclass_registry
  ).initial_state()
  return traverse(root, state)
