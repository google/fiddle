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

"""A function that sets tags within auto_config."""
from typing import Collection, TypeVar, Union

from fiddle._src import tag_type
from fiddle._src import tagging
from fiddle._src.experimental import auto_config


T = TypeVar('T')


def _with_tags_buildable_path(
    value: T,
    tags: Union[tag_type.TagType, Collection[tag_type.TagType]],
) -> tagging.TaggedValueCls[T]:
  """Make a `buildable_func` for `AutoConfig`."""
  if isinstance(tags, tag_type.TagType):
    return tagging.TaggedValue([tags], value)
  else:
    return tagging.TaggedValue(tags, value)


@auto_config.with_buildable_func(_with_tags_buildable_path)
def with_tags(
    value: T,
    tags: Union[tag_type.TagType, Collection[tag_type.TagType]],
) -> T:
  """Set tags for a parameter within auto_config.

  This method can tag parameters within auto_config while keeping the
  behavior of the original function intact.

  A parameter can also be associated with multiple tags.

  Example:

    class ActivationDTypeTag(fdl.Tag):
      "Data type for intermediate values."

    class DenseLayerTag(fdl.Tag):
      "Params of dense layer."


    @auto_config.auto_config
    def build_dense():
        return nn.Dense(
            features=auto_config.with_tags(10, DenseLayerTag)
            # A parameter can also have multiple tags.
            param_dtype=auto_config.with_tags(
                jnp.float32, [DenseLayerTag, ActivationDTypeTag])
        )

  Args:
    value: Default value for the paramter.
    tags: Tags to apply. It can be one tag or a list of tags.

  Returns:
    When called within `auto_config`ed function, return a tagged value.
      Otherwise, aka in normal Python mode, return the default value.
  """
  del tags
  return value
