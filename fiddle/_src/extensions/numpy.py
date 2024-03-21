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

"""Fiddle extensions to handle numpy code more elegantly.

Currently this affects codegen, graphviz, and other debugging functions.
"""

from fiddle._src import daglish_extensions
from fiddle._src.codegen import import_manager
from fiddle._src.codegen import py_val_to_cst_converter
from fiddle._src.codegen import special_value_codegen
import libcst as cst
import numpy as np


def _make_np_importable(name: str) -> special_value_codegen.Importable:
  return special_value_codegen.SingleImportable(
      "numpy", lambda np_name: f"{np_name}.{name}"
  )


_np_type_importables = (
    (np.int8, _make_np_importable("int8")),
    (np.complex128, _make_np_importable("complex128")),
    (np.clongdouble, _make_np_importable("clongdouble")),
    (np.complex64, _make_np_importable("complex64")),
    (np.float64, _make_np_importable("float64")),
    (np.longdouble, _make_np_importable("longdouble")),
    (np.float16, _make_np_importable("float16")),
    (np.float32, _make_np_importable("float32")),
    (np.int16, _make_np_importable("int16")),
    (np.int32, _make_np_importable("int32")),
    (np.int64, _make_np_importable("int64")),
    (np.timedelta64, _make_np_importable("timedelta64")),
    (np.uint8, _make_np_importable("uint8")),
    (np.uint64, _make_np_importable("uint64")),
    (np.uint16, _make_np_importable("uint16")),
    (np.uint32, _make_np_importable("uint32")),
)


_import_aliases = (
    # Rewrite internal import for numpy.
    ("numpy", "import numpy as np"),
)


def is_plain_dtype(value):
  """Returns True if `value` is a Numpy dtype instance.

  Dtype instances are created via np.dtype(), and can be simple ones like
  np.dtype('float32'), but also complex ones like,

  np.dtype([('name', 'S10'), ('age', 'i4'), ('height', 'f8')])

  Args:
    value: Arbitrary value to check.
  """
  return isinstance(value, np.dtype)


def convert_np_type_to_cst(value, convert_child):
  if value.fields:
    raise NotImplementedError("Struct dtypes are not supported yet.")
  name = value.name
  if name.startswith("void"):
    raise NotImplementedError(
        f'Dtype object {value} had a name that started with "void", this'
        " usually indicates that it could not be serialized."
    )
  return cst.Call(
      func=cst.Attribute(value=convert_child(np), attr=cst.Name("dtype")),
      args=[cst.Arg(convert_child(name))],
  )


def enable():
  """Registers Numpy fiddle extensions.

  This allows for things like nicer handling of numpy dtypes.
  """
  for value, importable in _np_type_importables:
    special_value_codegen.register_exact_value(value, importable)

  for module_str, import_stmt in _import_aliases:
    import_manager.register_import_alias(module_str, import_stmt)

  # The odd calling syntax here ("register(type)(handler)") comes from the fact
  # that register_converter is usually a decorator, but we call it directly.
  py_val_to_cst_converter.register_py_val_to_cst_converter(is_plain_dtype)(
      convert_np_type_to_cst
  )

  for dtype, _ in _np_type_importables:
    daglish_extensions.register_immutable(dtype)
