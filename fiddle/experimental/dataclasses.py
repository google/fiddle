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
from typing import Any, Collection, Mapping, Optional, Union

from fiddle import tag_type

TagOrTags = Union[tag_type.TagType, Collection[tag_type.TagType]]
_FIDDLE_DATACLASS_METADATA_KEY = object()


# TODO: Add kw_only=True when available.
@dataclasses.dataclass(frozen=True)
class FieldMetadata:
  """Fiddle-specific metadata that can be attached to each dataclasses.Field.

  Attributes:
    tags: A collection of tags to attach to the field.
  """
  tags: Collection[tag_type.TagType] = ()
  # TODO: Add additional metadata types here (value validation rules,
  # autofill / auto_config settings, etc).


def field(*,
          tags: Optional[TagOrTags] = None,
          metadata: Optional[Mapping[Any, Any]] = None,
          **kwargs) -> Union[dataclasses.Field[Any], Any]:
  """A wrapper around dataclasses.field to add optional Fiddle metadata.

  Args:
    tags: One or more tags to attach to the `fdl.Buildable`'s argument
      corresponding to the field.
    metadata: Any additional metadata to include.
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

  metadata: Mapping[Any, Any] = types.MappingProxyType(metadata or {})
  metadata = {
      **metadata, _FIDDLE_DATACLASS_METADATA_KEY: FieldMetadata(tags=tags)
  }
  return dataclasses.field(metadata=metadata, **kwargs)


def field_metadata(
    field_object: dataclasses.Field[Any]) -> Optional[FieldMetadata]:
  """Retrieves the Fiddle-specific metadata (if present) on `field`."""
  # Note: `field_object` is named as such, and not `field` to avoid shadowing
  # the `field` symbol (function) defined above.
  return field_object.metadata.get(_FIDDLE_DATACLASS_METADATA_KEY)
