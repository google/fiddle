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

"""Library for generating code from a Config or Partial object."""

# pylint: disable=unused-import
from fiddle._src.codegen.auto_config.experimental_top_level_api import auto_config_codegen
from fiddle._src.codegen.auto_config.experimental_top_level_api import code_generator as auto_config_code_generator
from fiddle._src.codegen.import_manager import register_import_alias
from fiddle._src.codegen.legacy_codegen import assignment_path
from fiddle._src.codegen.legacy_codegen import codegen_dot_syntax
from fiddle._src.codegen.legacy_codegen import mini_ast
from fiddle._src.codegen.legacy_codegen import SharedBuildableManager
from fiddle._src.codegen.new_codegen import code_generator
from fiddle._src.codegen.new_codegen import new_codegen
