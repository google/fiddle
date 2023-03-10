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

"""Helper functions for visualization of Fiddle configurations.

By default, you should be able to call APIs in `graphviz` and `printing` with no
special handling. But some real-world configurations become gigantic, and so
having a few helper functions for trimming them for interactive demos and such
can be valuable.
"""

# pylint: disable=unused-import
from fiddle._src.experimental.visualize import depth_over
from fiddle._src.experimental.visualize import structure
from fiddle._src.experimental.visualize import trim_fields_to
from fiddle._src.experimental.visualize import trim_long_fields
from fiddle._src.experimental.visualize import trimmed
from fiddle._src.experimental.visualize import with_defaults_trimmed
