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

"""Provides utilities for transforming builder functions into `fdl.Config`s.

This module defines the `auto_config` function (and associated helpers), which
can be used to convert an existing function that creates an object graph into a
function that creates a graph of `Config` and `Partial` objects. When the
resulting graph of `Config` and `Partial` objects is built via `fdl.build()`, it
will yield same object graph as the original function.
"""

# pylint: disable=unused-import
from fiddle._src.experimental.auto_config import auto_config
from fiddle._src.experimental.auto_config import auto_config_policy
from fiddle._src.experimental.auto_config import auto_unconfig
from fiddle._src.experimental.auto_config import AutoConfig
from fiddle._src.experimental.auto_config import ConfigTypes
from fiddle._src.experimental.auto_config import exempt
from fiddle._src.experimental.auto_config import inline
from fiddle._src.experimental.auto_config import is_auto_config
from fiddle._src.experimental.auto_config import UnsupportedLanguageConstructError
from fiddle._src.experimental.with_tags import with_tags
