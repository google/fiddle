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

"""A policy of functions to skip representing as `fdl.Config`'s.

When running `auto_config` on a function, some Python functions cannot or, in
practice, should never be mapped into `fdl.Config`-space. This file defines
policies (mostly implemented as lists) that enumerates them.

Because either adding or removing a function from this list can cause config
incompatibilities in end-user config, this file is structured around policy
versions.
"""

# pylint: disable=unused-import
from fiddle._src.experimental.auto_config_policy import latest
from fiddle._src.experimental.auto_config_policy import v1
