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
import inspect
import types
from typing import Any, Collection, Mapping, Optional, Union

from fiddle import config
from fiddle import tag_type
from fiddle._src import field_metadata
from fiddle.experimental import auto_config

TagOrTags = Union[tag_type.TagType, Collection[tag_type.TagType]]

FieldMetadata = field_metadata.FieldMetadata


def _has_signature(value):
  """Returns true if inspect.signature(value) succeeds."""
  try:
    inspect.signature(value)
  except ValueError:
    return False
  return True


def field(*,
          default_factory: Any = dataclasses.MISSING,
          tags: Optional[TagOrTags] = tuple(),
          metadata: Optional[Mapping[Any, Any]] = None,
          configurable_factory: bool = False,
          **kwargs) -> Union[dataclasses.Field[Any], Any]:
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
      this will make it possible to configure default values for the fields
      of that dataclass.  This should not be set to True if `default_factory`
      is an `auto_config`'ed function; see above for handling of `auto_config'ed
      `default_factory`.
    **kwargs: All other kwargs are passed to `dataclasses.field`; see the
      documentation on `dataclasses.field` for valid arguments.

  Returns:
    The result of calling dataclasses.field. Note: the return type is marked as
    a union with Any to pass static typechecking at usage sites.
  """
  # TODO: Make a function to return a metadata object to users to enable
  # them to call `dataclasses.field` themselves.
  if isinstance(tags, tag_type.TagType):
    tags = (tags,)

  if auto_config.is_auto_config(default_factory):
    if configurable_factory:
      raise ValueError("configurable_factory should not be used with "
                       "auto_config'ed functions.")
    buildable_initializer = default_factory.as_buildable
  elif configurable_factory:
    if not (default_factory and _has_signature(default_factory)):
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
