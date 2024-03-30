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

"""Experimental high-level API for auto_config codegen.

Do NOT depend on these interfaces for non-experimental code.
"""

import abc
import dataclasses
from typing import Any, Callable, Dict, List, Optional

from fiddle._src.codegen import namespace as namespace_lib
from fiddle._src.codegen.auto_config import complex_to_variables
from fiddle._src.codegen.auto_config import get_history_comments
from fiddle._src.codegen.auto_config import init_task
from fiddle._src.codegen.auto_config import ir_printer
from fiddle._src.codegen.auto_config import ir_to_cst
from fiddle._src.codegen.auto_config import make_symbolic_references
from fiddle._src.codegen.auto_config import naming
from fiddle._src.codegen.auto_config import shared_to_variables
from fiddle._src.codegen.auto_config import split_arg_factories
from fiddle._src.codegen.auto_config import sub_fixture
from fiddle._src.experimental import auto_config


class CodegenPass(metaclass=abc.ABCMeta):
  """Common API for codegen passes."""

  PASS_INPUT_KWARGS = []

  def print_debug(self, value: Any) -> None:
    print(
        f"\n\nAfter {self.__class__.__name__} ({self.__class__.__doc__})",
        ir_printer.format_task(value),
        sep="\n",
    )

  @abc.abstractmethod
  def __call__(self, value: Any, **pass_kwargs) -> Any:
    raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class MutationCodegenPass(CodegenPass):
  """Codegen pass that calls a function with attrs as keyword arguments."""

  fn: Callable[..., Any]

  def __call__(self, value: Any, **pass_kwargs) -> Any:
    kwargs = {
        field.name: getattr(self, field.name)
        for field in dataclasses.fields(self)
        if field.name != "fn"
    }
    kwargs.update(pass_kwargs)
    self.fn(value, **kwargs)
    return value


@dataclasses.dataclass(frozen=True)
class Codegen:
  """Top-level codegen object."""

  passes: List[CodegenPass]
  debug_print: bool = False

  def __call__(self, value, **kwargs):
    for codegen_pass in self.passes:
      pass_kwargs = {
          key: value
          for key, value in kwargs.items()
          if key in codegen_pass.PASS_INPUT_KWARGS
      }
      value = codegen_pass(value, **pass_kwargs)
      if self.debug_print:
        codegen_pass.print_debug(value)
    return value


@dataclasses.dataclass(frozen=True)
class InitTask(CodegenPass):
  """Initializes the task from the config."""

  top_level_fixture_name: str = "config_fixture"

  def __call__(self, value):
    return init_task.init_task(
        value, top_level_fixture_name=self.top_level_fixture_name
    )


@dataclasses.dataclass(frozen=True)
class ImportSymbols(MutationCodegenPass):
  """Preprocessing step to import symbols, for better naming."""

  fn: Callable[..., Any] = make_symbolic_references.import_symbols


@dataclasses.dataclass(frozen=True)
class TransformSubFixtures(CodegenPass):
  """Pulls out sub-configurations into sub-fixtures."""

  PASS_INPUT_KWARGS = ["sub_fixtures"]
  make_namer: Callable[[namespace_lib.Namespace], naming.Namer] = (
      naming.PathFirstNamer
  )

  def __call__(self, value: Any, *, sub_fixtures: Dict[str, Any]) -> Any:
    if sub_fixtures:
      sub_fixture.transform_sub_fixtures(
          value, sub_fixtures=sub_fixtures, make_namer=self.make_namer
      )
    return value


@dataclasses.dataclass(frozen=True)
class LowerArgFactories(MutationCodegenPass):
  """Lowers arg factories."""

  fn: Callable[..., Any] = split_arg_factories.lower_arg_factories


@dataclasses.dataclass(frozen=True)
class MoveSharedNodesToVariables(MutationCodegenPass):
  """Moves shared nodes to variables."""

  fn: Callable[..., Any] = shared_to_variables.move_shared_nodes_to_variables
  make_namer: Callable[[namespace_lib.Namespace], naming.Namer] = (
      naming.PathFirstNamer
  )


@dataclasses.dataclass(frozen=True)
class MoveComplexNodesToVariables(MutationCodegenPass):
  """Moves complex sub-expressions to variables."""

  fn: Callable[..., Any] = complex_to_variables.move_complex_nodes_to_variables
  is_complex: Callable[[Any], bool] = lambda x: False
  make_namer: Callable[[namespace_lib.Namespace], naming.Namer] = (
      naming.PathFirstNamer
  )


@dataclasses.dataclass(frozen=True)
class MakeSymbolicReferences(MutationCodegenPass):
  """Converts Buildable sub-objects to IR nodes representing function calls."""

  fn: Callable[..., Any] = (
      make_symbolic_references.replace_callables_and_configs_with_symbols
  )
  format_history: Callable[..., Any] = (
      make_symbolic_references.noop_history_comments
  )


@dataclasses.dataclass(frozen=True)
class IrToCst(CodegenPass):
  """Converts a codegen IR task to a LibCST module."""

  def print_debug(self, value: Any) -> None:
    # This pass doesn't print debug information.
    pass

  def __call__(self, value: Any) -> Any:
    return ir_to_cst.code_for_task(value)


@auto_config.auto_config(experimental_allow_control_flow=True)
def code_generator(
    top_level_fixture_name: str = "config_fixture",
    max_expression_complexity: Optional[int] = None,
    include_history: bool = False,
    debug_print: bool = False,
) -> Codegen:
  """Returns a low-level Codegen instance.

  This can be useful for building tooling that has rather custom code
  generation.

  Args:
    top_level_fixture_name: Name of the top-level fixture. When its as_buildable
      path is called, e.g. `config_fixture.as_buildable()`, this should return a
      config equivalent to `config`.
    max_expression_complexity: Breaks complex expressions into variables for
      readability.
    include_history: Whether history should be included. These currently appear
      as trailing comments in the field of Buildable's.
    debug_print: Whether to use the IR printer to print intermediate
      representations as various passes run to generate code.
  """
  passes = [
      InitTask(top_level_fixture_name=top_level_fixture_name),
      ImportSymbols(),
      TransformSubFixtures(),
  ]
  passes.extend([
      LowerArgFactories(),
      MoveSharedNodesToVariables(),
  ])
  if max_expression_complexity is not None:
    is_complex = complex_to_variables.more_complex_than(
        max_expression_complexity
    )
    passes.append(MoveComplexNodesToVariables(is_complex=is_complex))
  format_history = (
      get_history_comments.format_history_for_buildable
      if include_history
      else make_symbolic_references.noop_history_comments
  )
  passes.extend([
      MakeSymbolicReferences(format_history=format_history),
      IrToCst(),
  ])
  return Codegen(passes=passes, debug_print=debug_print)


def auto_config_codegen(
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
  codegen = code_generator(
      top_level_fixture_name=top_level_fixture_name,
      max_expression_complexity=max_expression_complexity,
      include_history=include_history,
      debug_print=debug_print,
  )
  return codegen(config, sub_fixtures=sub_fixtures).code
