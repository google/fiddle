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

"""Changes callables to symbol references."""

import inspect
from typing import Any

from fiddle import config as config_lib
from fiddle import daglish
from fiddle.codegen.auto_config import code_ir


def is_plain_symbol(value: Any) -> bool:
  """Returns whether the value is a plain symbol."""
  if isinstance(value, config_lib.Buildable):
    return False
  elif isinstance(value, type):
    return True
  elif inspect.isfunction(value):
    return True
  else:
    return False


def import_symbols(task: code_ir.CodegenTask) -> None:
  """Pass that just adds imports for symbols.

  It can be useful to run this pass early, so that other naming passes don't
  generate names which conflict with imports.

  Args:
    task: Codegen task.
  """

  task.import_manager.add(task.auto_config_fn)
  for value, _ in daglish.iterate(task.top_level_call.all_fixture_functions()):
    if isinstance(value, config_lib.Buildable):
      task.import_manager.add(config_lib.get_callable(value))
    elif is_plain_symbol(value):
      task.import_manager.add(value)


def replace_callables_and_configs_with_symbols(
    task: code_ir.CodegenTask,
) -> None:
  """Replaces callables and Buildables with symbolic versions."""

  def traverse(value, state: daglish.State):
    value = state.map_children(value)
    if isinstance(value, config_lib.Buildable):
      symbol = task.import_manager.add(config_lib.get_callable(value))
      if isinstance(value, config_lib.Config):
        return code_ir.SymbolCall(
            symbol_expression=symbol,
            arg_expressions=config_lib.ordered_arguments(value),
        )
      elif isinstance(value, config_lib.Partial):
        return code_ir.SymbolCall(
            symbol_expression=symbol,
            arg_expressions=config_lib.ordered_arguments(value),
        )
      else:
        raise TypeError(f"Unsupported Buildable {type(value)}")
    elif is_plain_symbol(value):
      symbol = task.import_manager.add(value)
      return code_ir.SymbolReference(symbol)
    return value

  for fn in task.top_level_call.all_fixture_functions():
    fn.replace_with(daglish.MemoizedTraversal.run(traverse, fn))
