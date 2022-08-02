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

"""Allows adding tags within `auto_config` functions.

Beware: `adding_tags` only works within `auto_config` functions.
"""

import typing
from typing import Collection, TypeVar, Union

from fiddle import tag_type
from fiddle.experimental import auto_config

T = TypeVar('T')


def adding_tags(*,
                tags: Union[tag_type.TagType, Collection[tag_type.TagType]],
                value: T = tag_type.UnsetValue) -> T:
  del tags  # Tags aren't set in regular Python execution.
  return value


if not typing.TYPE_CHECKING:

  def _adding_tags_auto_config(
      *,
      tags: Union[tag_type.TagType, Collection[tag_type.TagType]],
      value: T = tag_type.UnsetValue) -> tag_type.AddingTags:
    if isinstance(tags, tag_type.TagType):
      tags = (tags,)
    return tag_type.AddingTags(tags=tags, value=value)

  adding_tags = auto_config.AutoConfig(
      func=adding_tags, buildable_func=_adding_tags_auto_config)
