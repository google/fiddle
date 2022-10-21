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

"""Tests for fiddle.extensions.tf."""

from typing import List

from absl.testing import absltest
from absl.testing import parameterized
import fiddle as fdl
from fiddle import graphviz
from fiddle import printing
from fiddle.codegen import codegen
from fiddle.codegen import py_val_to_cst_converter
from fiddle.codegen import special_value_codegen
from fiddle.experimental import serialization
import fiddle.extensions.tf
import libcst as cst
import tensorflow as tf


def foo(dtype):
  return dtype


def tokens(code: str) -> List[str]:
  return code.strip().split()


def foo_with_default(dtype=tf.bfloat16):
  return dtype


class NoopImportManager(special_value_codegen.ImportManagerApi):

  def __init__(self):
    self.calls = []

  def add_by_name(self, module_name: str) -> str:
    self.calls.append(module_name)
    return module_name


class TfTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    fiddle.extensions.tf.enable()

  def test_special_value_codegen(self):
    import_manager = NoopImportManager()
    float32_value = special_value_codegen.transform_py_value(
        tf.float32, import_manager)
    self.assertEqual(import_manager.calls, ["tensorflow"])
    self.assertEqual(repr(float32_value), "tensorflow.float32")

  def test_codegen(self):
    config = fdl.Config(foo, dtype=tf.bfloat16)
    code = "\n".join(codegen.codegen_dot_syntax(config).lines())
    if __name__ == "__main__":
      tf_test_import = ""
      foo_name = "foo"
    else:
      # Running via PyTest; tf_test is importable.
      tf_test_import = "from fiddle.extensions import tf_test\n"
      foo_name = "tf_test.foo"
    expected = f"""
import fiddle as fdl
{tf_test_import}import tensorflow as tf


def build_config():
  root = fdl.Config({foo_name})
  root.dtype = tf.bfloat16

  return root
    """
    self.assertEqual(tokens(code), tokens(expected))

  def test_graphviz(self):
    config = fdl.Config(foo, dtype=tf.bfloat16)
    result = graphviz.render(config)
    self.assertIn("tf.bfloat16", result.source)

  def test_str_printing(self):
    config = fdl.Config(foo, dtype=tf.bfloat16)
    result = printing.as_str_flattened(config)
    self.assertEqual("dtype = tf.bfloat16", result)

  def test_history_printing(self):
    config = fdl.Config(foo, dtype=tf.float16)
    config.dtype = tf.float32
    result = printing.history_per_leaf_parameter(config)
    expected = r"""
dtype = tf\.float32 @ .*/tf_test.py:\d+:test_history_printing
  - previously: tf.float16 @ .*/tf_test.py:\d+:test_history_printing
""".strip()
    self.assertRegex(result, expected)

  def test_default_printing(self):
    config = fdl.Config(foo_with_default)
    result = printing.as_str_flattened(config)
    self.assertEqual("dtype = <[unset; default: tf.bfloat16]>", result)

  @parameterized.parameters([
      (tf.constant([1, 2, 3]),
       "tensorflow.constant([1, 2, 3], dtype=tensorflow.int32, shape=[3])"),
      (tf.int32, "tensorflow.int32"),
      (tf.TensorShape(None), "tensorflow.TensorShape(None)"),
      (tf.TensorShape([1, 2, 3]), "tensorflow.TensorShape([1, 2, 3])"),
      (tf.TensorShape([1, None, 3]), "tensorflow.TensorShape([1, None, 3])"),
  ])
  def test_py_val_to_cst_converter(self, value, expected):
    cst_expr = py_val_to_cst_converter.convert_py_val_to_cst(value)
    cst_module = cst.Module([cst.SimpleStatementLine([cst.Expr(cst_expr)])])
    self.assertEqual(cst_module.code.strip(), expected)

  def test_serialization(self):
    serialized_cfg = serialization.dump_json(fdl.Config(foo, tf.int32))
    deserialized_cfg = serialization.load_json(serialized_cfg)
    self.assertIs(deserialized_cfg.dtype, tf.int32)


if __name__ == "__main__":
  absltest.main()
