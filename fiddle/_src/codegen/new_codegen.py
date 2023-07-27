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

"""Generates Fiddle code with the auto_config codegen codebase.

Currently this API is under development; please do not depend on it.
"""

import dataclasses
from typing import Any, Callable, Dict, Optional, Type

import fiddle as fdl
from fiddle._src.codegen import newcg_symbolic_references
from fiddle._src.codegen.auto_config import add_type_signatures
from fiddle._src.codegen.auto_config import code_ir
from fiddle._src.codegen.auto_config import experimental_top_level_api
from fiddle._src.codegen.auto_config import make_symbolic_references as old_symbolic_references
from fiddle._src.experimental import auto_config


@dataclasses.dataclass(frozen=True)
class RemoveAutoConfigFn(experimental_top_level_api.CodegenPass):

  def __call__(self, value: Any) -> Any:
    assert isinstance(value, code_ir.CodegenTask)
    value.auto_config_fn = None
    return value


@dataclasses.dataclass(frozen=True)
class ImportSymbols(experimental_top_level_api.MutationCodegenPass):
  """Preprocessing step to import symbols, for better naming."""

  fn: Callable[..., Any] = newcg_symbolic_references.import_symbols


@dataclasses.dataclass(frozen=True)
class MakeSymbolicReferences(experimental_top_level_api.MutationCodegenPass):
  """Converts Buildable sub-objects to IR nodes representing function calls.

  In this case, we emit fdl.Config(), fdl.Partial(), and fdl.ArgFactory() calls,
  instead of calls to the configured functions/classes.
  """

  fn: Callable[..., Any] = (
      newcg_symbolic_references.replace_callables_and_configs_with_symbols
  )
  format_history: Callable[..., Any] = (
      old_symbolic_references.noop_history_comments
  )


@dataclasses.dataclass(frozen=True)
class AddTypeSignatures(experimental_top_level_api.MutationCodegenPass):
  """Adds return type signatures to fixtures."""

  fn: Callable[..., Any] = add_type_signatures.add_return_types


def _get_pass_idx(
    codegen_config: fdl.Config[experimental_top_level_api.Codegen],
    cls: Type[experimental_top_level_api.CodegenPass],
) -> int:
  for i, codegen_pass in enumerate(codegen_config.passes):
    if issubclass(fdl.get_callable(codegen_pass), cls):
      return i
  raise ValueError(f"Could not find codegen pass {cls}")


@auto_config.auto_unconfig
def code_generator(
    top_level_fixture_name: str = "config_fixture",
    max_expression_complexity: Optional[int] = None,
    include_history: bool = False,
    debug_print: bool = False,
) -> experimental_top_level_api.Codegen:
  """Returns a low-level Codegen instance; see experimental_top_level_api."""
  config = experimental_top_level_api.code_generator.as_buildable(
      top_level_fixture_name=top_level_fixture_name,
      max_expression_complexity=max_expression_complexity,
      include_history=include_history,
      debug_print=debug_print,
  )

  # Unset auto_config right after initialization.
  config.passes.insert(1, fdl.Config(RemoveAutoConfigFn))

  # Remove LowerArgFactories
  idx = _get_pass_idx(config, experimental_top_level_api.LowerArgFactories)
  config.passes.pop(idx)

  # Replace ImportSymbols
  idx = _get_pass_idx(config, experimental_top_level_api.ImportSymbols)
  fdl.update_callable(config.passes[idx], ImportSymbols)

  # Replace MakeSymbolicReferences
  idx = _get_pass_idx(config, experimental_top_level_api.MakeSymbolicReferences)
  fdl.update_callable(config.passes[idx], MakeSymbolicReferences)

  # Insert type annotations before MakeSymbolicReferences. These type
  # annotations currently make more sense for non-auto_config cases.
  config.passes.insert(idx, fdl.Config(AddTypeSignatures))

  return config


def new_codegen(
    config,
    *,
    top_level_fixture_name: str = "config_fixture",
    sub_fixtures: Optional[Dict[str, Any]] = None,
    max_expression_complexity: Optional[int] = None,
    include_history: bool = False,
    debug_print: bool = False,
) -> str:
  """Generates code for an auto_config fixture.

  Args:
    config: Configuration to generate auto_config code for.
    top_level_fixture_name: Name of the top-level fixture. When its as_buildable
      path is called, e.g. `config_fixture.as_buildable()`, this should return a
      config equivalent to `config`.
    sub_fixtures: Dictionary from function name to sub-configuration object,
      declaring sub-functions to create. This helps manage complexity of huge
      configurations.
    max_expression_complexity: Breaks complex expressions into variables for
      readability.
    include_history: Whether history should be included. These currently appear
      as trailing comments in the field of Buildable's.
    debug_print: Whether to use the IR printer to print intermediate
      representations as various passes run to generate code.

  Returns:
    Python module code.
  """
  codegen_obj: experimental_top_level_api.Codegen = code_generator(
      top_level_fixture_name=top_level_fixture_name,
      max_expression_complexity=max_expression_complexity,
      include_history=include_history,
      debug_print=debug_print,
  )
  return codegen_obj(config, sub_fixtures=sub_fixtures).code
