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

"""Fiddle extensions to handle TensorFlow code more elegantly.

Currently this affects codegen, graphviz, serialization, and other
debugging functions.
"""

import functools
from fiddle.codegen import import_manager
from fiddle.codegen import py_val_to_cst_converter
from fiddle.codegen import special_value_codegen
from fiddle.experimental import serialization
import libcst as cst
import tensorflow as tf

tf_dtypes = ("bfloat16", "bool", "complex128", "complex64", "float16",
             "float32", "float64", "int16", "int32", "int64", "int8", "qint16",
             "qint32", "qint8", "quint16", "quint8", "resource", "string",
             "uint16", "uint32", "uint64", "uint8", "variant")

_import_aliases = (
    # Rewrite TensorFlow imports to standard format.
    ("tensorflow", "import tensorflow as tf"),)


def is_tensor(value):
  """Returns true if `value` is an eager TensorFlow `Tensor`."""
  return isinstance(value, tf.Tensor) and hasattr(value, "numpy")


def enable():
  """Registers TensorFlow fiddle extensions.

  This allows for things like nicer handling of tf dtypes.
  """

  def make_dtype_name(module, dtype_name):
    return f"{module}.{dtype_name}"

  for dtype_name in tf_dtypes:
    dtype_val = getattr(tf.dtypes, dtype_name)
    serialization.register_constant(
        "tensorflow", dtype_name, compare_by_identity=False)
    importable = special_value_codegen.SingleImportable(
        "tensorflow", functools.partial(make_dtype_name, dtype_name=dtype_name))
    special_value_codegen.register_exact_value(dtype_val, importable)

  for module_str, import_stmt in _import_aliases:
    import_manager.register_import_alias(module_str, import_stmt)

  @py_val_to_cst_converter.register_py_val_to_cst_converter(is_tensor)
  def convert_tensor_to_cst(value, convert_child):
    return cst.Call(
        func=cst.Attribute(value=convert_child(tf), attr=cst.Name("constant")),
        args=[
            cst.Arg(convert_child(value.numpy().tolist())),
            py_val_to_cst_converter.kwarg_to_cst("dtype",
                                                 convert_child(value.dtype)),
            py_val_to_cst_converter.kwarg_to_cst(
                "shape", convert_child(value.shape.as_list()))
        ])

  @py_val_to_cst_converter.register_py_val_to_cst_converter(tf.DType)
  def convert_dtype_to_cst(value, convert_child):
    return cst.Attribute(value=convert_child(tf), attr=cst.Name(value.name))

  @py_val_to_cst_converter.register_py_val_to_cst_converter(tf.TensorShape)
  def convert_tensor_shape_to_cst(value, convert_child):
    shape_list = None if value.rank is None else value.as_list()
    return cst.Call(
        func=cst.Attribute(
            value=convert_child(tf), attr=cst.Name("TensorShape")),
        args=[cst.Arg(convert_child(shape_list))])
