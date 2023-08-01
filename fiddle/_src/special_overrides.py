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

"""Utilities for special overrides for codegen and serialization."""

import dataclasses
from typing import Dict, Optional


@dataclasses.dataclass
class SpecialOverrides:
  """Overrides for codegen and serialization."""

  # Name of the module that may have overrides.
  module_name: str
  # Import alias, if any, for the module in `module_name`.
  module_import_alias: Optional[str] = None
  # Overridden fully qualified module name to be replaced with for the symbols
  # that have been migrated to different modules.
  migrated_symbol_destination_modules: Dict[str, str] = dataclasses.field(
      default_factory=dict
  )


SPECIAL_OVERRIDES_MAP = {
    "fiddle.config": SpecialOverrides(
        module_name="fiddle.config",
        module_import_alias="import fiddle as fdl",
    ),
    "fiddle": SpecialOverrides(
        module_name="fiddle", module_import_alias="import fiddle as fdl"
    ),
    "fiddle._src.config": SpecialOverrides(
        module_name="fiddle._src.config",
        module_import_alias="import fiddle as fdl",
    ),
    "fiddle._src.experimental.auto_config": SpecialOverrides(
        module_name="fiddle._src.experimental.auto_config",
        module_import_alias="from fiddle.experimental import auto_config",
    ),
    "fiddle._src.arg_factory": SpecialOverrides(
        module_name="fiddle._src.arg_factory",
        module_import_alias="from fiddle import arg_factory",
    ),
    "fiddle._src.codegen": SpecialOverrides(
        module_name="fiddle._src.codegen",
        module_import_alias="from fiddle import codegen",
    ),
    "fiddle._src.codegen.py_val_to_cst_converter": SpecialOverrides(
        module_name="fiddle._src.codegen.py_val_to_cst_converter",
        module_import_alias="from fiddle.codgen import py_val_to_cst_converter",
    ),
    "fiddle._src.daglish": SpecialOverrides(
        module_name="fiddle._src.daglish",
        module_import_alias="from fiddle import daglish",
    ),
}


def register_special_override(
    module_name: str, special_overrides: SpecialOverrides
) -> None:
  SPECIAL_OVERRIDES_MAP[module_name] = special_overrides
