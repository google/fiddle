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

# pylint: disable=unused-import
from fiddle._src.experimental.dataclasses import convert_dataclasses_to_configs
from fiddle._src.experimental.dataclasses import daglish_dataclass_registry
from fiddle._src.experimental.dataclasses import field
from fiddle._src.experimental.dataclasses import field_has_tag
from fiddle._src.experimental.dataclasses import FieldMetadata
from fiddle._src.experimental.dataclasses import TagOrTags
