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

"""Adds type signatures to modules.

For now, we only populate return types.
"""

import inspect

from fiddle._src import config as config_lib
from fiddle._src import signatures
from fiddle._src.codegen import import_manager as import_manager_lib
from fiddle._src.codegen.auto_config import code_ir
from fiddle._src.codegen.auto_config import import_manager_wrapper


_BUILTIN_TYPE_MAP = {
    type(None): "None",
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
}


def _get_annotation_from_type(typ) -> code_ir.CodegenNode:
  if typ in _BUILTIN_TYPE_MAP:
    return code_ir.BuiltinReference(code_ir.Name(_BUILTIN_TYPE_MAP[typ]))
  else:
    # TODO(b/293352960): import typing.Any correctly.
    # TODO(b/293509806): Handle more types, especially from function return
    # signatures.
    return code_ir.BuiltinReference(code_ir.Name("Any"))


def get_type_annotation(
    value, import_manager: import_manager_lib.ImportManager
) -> code_ir.CodegenNode:
  """Gets the type annotation for a given value."""
  if isinstance(value, config_lib.Buildable):
    buildable_type = import_manager_wrapper.add(type(value), import_manager)
    fn_or_cls = config_lib.get_callable(value)
    if isinstance(fn_or_cls, type):
      sub_type = import_manager_wrapper.add(fn_or_cls, import_manager)
    else:
      signature = signatures.get_signature(fn_or_cls)
      if isinstance(signature.return_annotation, type) and (
          signature.return_annotation is not inspect.Signature.empty
      ):
        sub_type = _get_annotation_from_type(signature.return_annotation)
      else:
        return buildable_type
    return code_ir.ParameterizedTypeExpression(buildable_type, [sub_type])
  elif isinstance(value, (list, tuple)):
    base_expression = code_ir.BuiltinReference(
        code_ir.Name("list" if isinstance(value, list) else "tuple")
    )
    sub_value_annotations = [
        get_type_annotation(item, import_manager) for item in value
    ]
    if sub_value_annotations and all(
        annotation == sub_value_annotations[0]
        for annotation in sub_value_annotations
    ):
      return code_ir.ParameterizedTypeExpression(
          base_expression, [sub_value_annotations[0]]
      )
    else:
      return base_expression
  elif isinstance(value, dict):
    base_expression = code_ir.BuiltinReference(code_ir.Name("dict"))
    key_annotations = [
        get_type_annotation(item, import_manager) for item in value.keys()
    ]
    value_annotations = [
        get_type_annotation(item, import_manager) for item in value.values()
    ]
    if key_annotations and all(
        annotation == key_annotations[0] for annotation in key_annotations
    ):
      key_annotation = key_annotations[0]
    else:
      # TODO(b/293352960): import typing.Any correctly.
      key_annotation = code_ir.BuiltinReference(code_ir.Name("Any"))
    if value_annotations and all(
        annotation == value_annotations[0] for annotation in value_annotations
    ):
      value_annotation = value_annotations[0]
    else:
      value_annotation = code_ir.BuiltinReference(code_ir.Name("Any"))
    return code_ir.ParameterizedTypeExpression(
        base_expression, [key_annotation, value_annotation]
    )
  else:
    return _get_annotation_from_type(type(value))


def add_return_types(task: code_ir.CodegenTask) -> None:
  """Adds return type signatures.

  This is normally based on config types, so for `auto_config`, it would reflect
  the as_buildable() path. Hence, we don't add it by default yet.

  Args:
    task: Codegen task.
  """
  for fn in task.top_level_call.all_fixture_functions():
    fn.return_type_annotation = get_type_annotation(
        fn.output_value, task.import_manager
    )
