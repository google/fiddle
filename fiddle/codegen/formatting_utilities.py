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

"""Routines for nice string representations of cfg values.

Some types (e.g. numpy arrays, jax dtypes, TensorFlow, etc) are often referred
unambiguously by a name that differs from their qualified path. This library
exposes the pretty-printing functionality that powers the codegen features to
other string representations, such as graphviz or string serialization.
"""

from typing import Any

from fiddle.codegen import codegen
from fiddle.codegen import special_value_codegen


class LazyImportManager(special_value_codegen.ImportManagerApi):
  """Import manager for GraphViz & string formatting.

  We create an instance of the codegen's import manager on every method call,
  which effectively will alias imports like 'jax.numpy' to 'jnp', but will not
  create new import aliases when two modules of the same name are referenced.
  """

  def add_by_name(self, module_name: str) -> str:
    """Returns the usable name generated from an ephemeral import manager."""
    namespace = codegen.Namespace()
    import_manager = codegen.ImportManager(namespace)
    return import_manager.add_by_name(module_name)


def pretty_print(value: Any) -> str:
  """Returns the nicest eval-able string representation for `value`."""
  return special_value_codegen.transform_py_value(value, LazyImportManager())
