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

"""Specialized code generation for certain types/values.

Several Python values, such as `jnp.bfloat16` (`jax.numpy.bfloat16`), will
render badly if we just try to call repr() on them. This module creates an
extensible API to register better formatting handlers for ML libraries.
"""

import abc
import dataclasses
from typing import Any, Callable, List

import typing_extensions


class ImportManagerApi(typing_extensions.Protocol):
  """Defines an API for a helper class that imports modules."""

  def add_by_name(self, module_name: str) -> str:
    pass


@dataclasses.dataclass(frozen=True)
class ReprString:
  """Helper class whose repr() is just the provided string."""
  repr_value: str

  def __repr__(self):
    return self.repr_value


class Importable(metaclass=abc.ABCMeta):

  @abc.abstractproperty
  def import_modules(self) -> List[str]:
    raise NotImplementedError()

  @abc.abstractmethod
  def repr_string(self, imported_module_names: List[str]) -> ReprString:
    raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class SingleImportable(Importable):
  """An Importable instance, that only needs a single module."""
  import_module: str
  repr_fn: Callable[[str], str]

  @property
  def import_modules(self) -> List[str]:
    return [self.import_module]

  def repr_string(self, imported_module_names: List[str]) -> ReprString:
    imported_module_name, = imported_module_names
    return ReprString(self.repr_fn(imported_module_name))


_EXACT_VALUE_LOOKUP = {}


def register_exact_value(value: Any, resolution: Importable) -> None:
  """Registers an exact-value match."""
  _EXACT_VALUE_LOOKUP[value] = resolution


def transform_py_value(value: Any, import_manager: ImportManagerApi) -> Any:
  """Transforms a Python value for code generation or printing with repr().

  Args:
    value: Python value, typically a leaf node if there are nested data
      structures. Data structures are not automatically transformed by this
      function.
    import_manager: Import manager used to resolve imports.

  Returns:
    Either `value` unchanged, or a ReprString containing code, which when
    evaluated, returns an equivalent object to `value`. (The exact semantics
    of equivalence--by object identity or regular equality--is not guaranteed,
    and mostly a function of what extensions the user has enabled.)
  """
  try:
    importable: Importable = _EXACT_VALUE_LOOKUP[value]
  except (KeyError, TypeError):
    pass
  else:
    imported_names = [
        import_manager.add_by_name(module_name)
        for module_name in importable.import_modules
    ]
    return importable.repr_string(imported_names)

  # Nothing matched, return the value unchanged and hope its repr() string is
  # valid Python.
  return value
