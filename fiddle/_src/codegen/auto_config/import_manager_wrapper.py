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

"""Small helper functions around the ImportManager.

This is a bit of cruft and should eventually be cleaned up.

Context: The import manager predates modern (auto_config) codegen, and is used
by legacy codegen and diff codegen. The latter is still pretty important and
needs to be supported.
"""

import logging
import typing
from typing import Any

from fiddle._src.codegen import import_manager as import_manager_lib
from fiddle._src.codegen.auto_config import code_ir


def _name_to_attribute_expression(name: str) -> code_ir.CodegenNode:
  """Converts a fully-qualified name to a code_ir node.

  Args:
    name: Output from the import manager.

  Returns:
    Codegen node.
  """
  if "." not in name:
    logging.warning(
        "Expected to find a module in %s, but found none. This might be because"
        " your module is from __main__, so we'll still emit code, but you might"
        " need to fix imports for this symbol.",
        name,
    )
    return code_ir.BaseNameReference(code_ir.Name(name))
  base, *parts = name.split(".")
  value = code_ir.ModuleReference(code_ir.Name(base))
  for part in parts:
    value = code_ir.AttributeExpression(value, part)
  return typing.cast(code_ir.AttributeExpression, value)


def add(
    value: Any, import_manager: import_manager_lib.ImportManager
) -> code_ir.CodegenNode:
  return _name_to_attribute_expression(import_manager.add(value))
