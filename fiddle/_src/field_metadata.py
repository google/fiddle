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

"""Fiddle-specific metadata associated with dataclass fields."""

import dataclasses
from typing import Any, Callable, Collection, Optional

from fiddle import tag_type

_FIDDLE_DATACLASS_METADATA_KEY = object()


# TODO: Add kw_only=True when available.
@dataclasses.dataclass(frozen=True)
class FieldMetadata:
  """Fiddle-specific metadata that can be attached to each dataclasses.Field.

  Attributes:
    tags: A collection of tags to attach to the field.
    buildable_initializer: An optional callable to initialize the field's value
      when creating a `fdl.Buildable` of the enclosing type.
  """
  tags: Collection[tag_type.TagType]
  buildable_initializer: Optional[Callable[[], Any]]
  # TODO: Add additional metadata here (e.g. value validation rules).


def field_metadata(
    field_object: dataclasses.Field[Any]) -> Optional[FieldMetadata]:
  """Retrieves the Fiddle-specific metadata (if present) on `field`."""
  # Note: `field_object` is named as such, and not `field` to avoid shadowing
  # the `field` symbol (function) defined above.
  return field_object.metadata.get(_FIDDLE_DATACLASS_METADATA_KEY)
