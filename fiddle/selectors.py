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

"""Library for manipulating selections of a Buildable DAG.

A common need for configuration libraries is to override settings in some kind
of base configuration, and these APIs allow such overrides to take place
imperatively.
"""

# pylint: disable=unused-import
from fiddle._src.selectors import NodeSelection
from fiddle._src.selectors import select
from fiddle._src.selectors import Selection
from fiddle._src.selectors import TagSelection
