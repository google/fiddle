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

"""Tests for fiddle.extensions.numpy."""

from typing import List

from absl.testing import absltest
import fiddle as fdl
from fiddle import graphviz
from fiddle import printing
from fiddle._src.codegen import legacy_codegen
from fiddle._src.codegen import special_value_codegen
from fiddle._src.extensions.test_modules import dtype_module
from fiddle.codegen import py_val_to_cst_converter
from fiddle.codegen.auto_config import experimental_top_level_api
import fiddle.extensions.numpy
import numpy as np


def foo(dtype):
  return dtype


def tokens(code: str) -> List[str]:
  return code.strip().split()


def foo_with_default(dtype=np.float16):
  return dtype


class NoopImportManager(special_value_codegen.ImportManagerApi):

  def __init__(self):
    self.calls = []

  def add_by_name(self, module_name: str) -> str:
    self.calls.append(module_name)
    return module_name


class NumpyTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    fiddle.extensions.numpy.enable()

  def test_enable(self):
    """Directly tests the effects of enable() through special_value_codegen."""
    import_manager = NoopImportManager()
    float32_value = special_value_codegen.transform_py_value(
        np.float32, import_manager
    )
    self.assertEqual(import_manager.calls, ["numpy"])
    self.assertEqual(repr(float32_value), "numpy.float32")

  def test_codegen(self):
    config = fdl.Config(dtype_module.foo, dtype=np.float16)
    code = "\n".join(legacy_codegen.codegen_dot_syntax(config).lines())
    expected = """
import fiddle as fdl
from fiddle._src.extensions.test_modules import dtype_module
import numpy as np


def build_config():
  root = fdl.Config(dtype_module.foo)
  root.dtype = np.float16

  return root
    """
    self.assertEqual(tokens(code), tokens(expected))

  def test_codegen_dtype_instance(self):
    config = fdl.Config(dtype_module.foo, dtype=np.dtype("complex128"))
    code = experimental_top_level_api.auto_config_codegen(config)
    # TODO(b/330590216): This code is not actually correct, we should be
    # importing numpy and using its common alias. Fiddle's codegen is a little
    # fragmented, as historically it has evolved in three phases: legacy
    # codegen, diff codegen, and auto_config codegen. For the time being, usage
    # is limited enough and we're OK with a little bit of manual cleanup, so
    # it's better to have this work and produce something fixable than to throw
    # an error.
    expected = """
from fiddle._src.extensions.test_modules import dtype_module
from fiddle.experimental import auto_config


@auto_config.auto_config
def config_fixture():
    return dtype_module.foo(dtype=numpy.dtype('complex128'))
    """
    self.assertEqual(tokens(code), tokens(expected))

  def test_graphviz(self):
    config = fdl.Config(foo, dtype=np.float16)
    result = graphviz.render(config)
    self.assertIn("np.float16", result.source)

  def test_str_printing(self):
    config = fdl.Config(foo, dtype=np.float16)
    result = printing.as_str_flattened(config)
    self.assertEqual("dtype = np.float16", result)

  def test_str_printing_plain_dtype(self):
    """This is default behavior, NOT the behavior of the numpy extension.

    But it seems to work well enough, so might as well add a test. Feel free
    to change if we do issue qualified import statements.
    """
    config = fdl.Config(foo, dtype=np.dtype("float16"))
    result = printing.as_str_flattened(config)
    self.assertEqual("dtype = dtype('float16')", result)

  def test_history_printing(self):
    config = fdl.Config(foo, dtype=np.float16)
    config.dtype = np.float32
    result = printing.history_per_leaf_parameter(config)
    expected = r"""
dtype = np\.float32 @ .*/numpy_test.py:\d+:test_history_printing
  - previously: np.float16 @ .*/numpy_test.py:\d+:test_history_printing
""".strip()
    self.assertRegex(result, expected)

  def test_default_printing(self):
    config = fdl.Config(foo_with_default)
    result = printing.as_str_flattened(config)
    self.assertEqual("dtype = <[unset; default: np.float16]>", result)

  def test_py_val_to_cst_converter_error(self):
    # This is currently unsupported, but just check that the error message is
    # reasonable until we add support.
    array = np.array([1, 2, 3], "float32")
    with self.assertRaisesRegex(ValueError, ".*ndarray.*"):
      py_val_to_cst_converter.convert_py_val_to_cst(array)


if __name__ == "__main__":
  absltest.main()
