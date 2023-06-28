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

"""Experimental high-level API for auto_config codegen.

Do NOT depend on these interfaces for non-experimental code.
"""

# pylint: disable=unused-import
from fiddle._src.codegen.auto_config.experimental_top_level_api import auto_config_codegen
from fiddle._src.codegen.auto_config.experimental_top_level_api import code_generator
from fiddle._src.codegen.auto_config.experimental_top_level_api import CodegenPass
from fiddle._src.codegen.auto_config.experimental_top_level_api import ImportSymbols
from fiddle._src.codegen.auto_config.experimental_top_level_api import InitTask
from fiddle._src.codegen.auto_config.experimental_top_level_api import IrToCst
from fiddle._src.codegen.auto_config.experimental_top_level_api import LowerArgFactories
from fiddle._src.codegen.auto_config.experimental_top_level_api import MakeSymbolicReferences
from fiddle._src.codegen.auto_config.experimental_top_level_api import MoveComplexNodesToVariables
from fiddle._src.codegen.auto_config.experimental_top_level_api import MoveSharedNodesToVariables
from fiddle._src.codegen.auto_config.experimental_top_level_api import MutationCodegenPass
from fiddle._src.codegen.auto_config.experimental_top_level_api import TransformSubFixtures
