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

import enum
import functools
import inspect
from typing import Any

from fiddle import arg_factory
from fiddle import daglish
from fiddle._src import config as config_lib
from fiddle._src.codegen.auto_config import code_ir


def is_plain_symbol_or_enum_value(value: Any) -> bool:
  """Returns whether the value is a plain symbol."""
  if isinstance(value, config_lib.Buildable):
    return False
  elif isinstance(value, type):
    return True
  elif inspect.isfunction(value):
    return True
  elif isinstance(value, enum.Enum):
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
    elif is_plain_symbol_or_enum_value(value):
      task.import_manager.add(value)


def replace_callables_and_configs_with_symbols(
    task: code_ir.CodegenTask,
) -> None:
  """Replaces callables and Buildables with symbolic versions."""

  fn_name = None

  def _handle_partial(
      value: config_lib.Partial, state: daglish.State, symbol: str
  ):
    """Split-out helper method to handle Partial() nodes."""
    symbol_ref = code_ir.SymbolReference(symbol)

    arguments = config_lib.ordered_arguments(value)

    # Arguments which were config_lib.ArgFactory arguments; these need to be
    # turned into regular calls.
    arg_factory_args = {}

    # All other arguments.
    regular_args = {}

    for name, arg_value in arguments.items():
      if isinstance(arg_value, code_ir.ArgFactoryExpr):
        arg_factory_args[name] = state.call(
            arg_value.expression, daglish.Attr(name)
        )
      else:
        regular_args[name] = state.call(arg_value, daglish.Attr(name))

    def _arg_factory_partial():
      return code_ir.SymbolCall(
          task.import_manager.add(arg_factory.partial),
          positional_arg_expressions=[symbol_ref],
          arg_expressions=arg_factory_args,
      )

    if not arg_factory_args:
      # The common case: there are no arg_factory's, so just emit a functools
      # partial. We currently emit a functools partial even if there are no
      # arguments, because this will mean there's a fdl.Partial when we call
      # the auto_config fixture's as_buildable() method. If we got rid of the
      # functools.partial, then we couldn't configure any attributes.
      return code_ir.SymbolCall(
          task.import_manager.add(functools.partial),
          positional_arg_expressions=[symbol_ref],
          arg_expressions=regular_args,
      )
    elif not regular_args:
      # There are only arg_factory args, so we can emit an arg_factory.partial.
      return _arg_factory_partial()
    else:
      # We have both functools.partial and arg_factory args. It doesn't matter
      # which order, but we need to emit both decorators. Go with functools
      # on the outer level.
      return code_ir.SymbolCall(
          task.import_manager.add(functools.partial),
          positional_arg_expressions=[_arg_factory_partial()],
          arg_expressions=regular_args,
      )

  def traverse(value, state: daglish.State):
    if isinstance(value, config_lib.Buildable):
      symbol = task.import_manager.add(config_lib.get_callable(value))
      if isinstance(value, config_lib.Config):
        value = state.map_children(value)
        return code_ir.SymbolCall(
            symbol_expression=symbol,
            positional_arg_expressions=[],
            arg_expressions=config_lib.ordered_arguments(value),
        )
      elif isinstance(value, config_lib.Partial):
        return _handle_partial(value, state, symbol)
      elif isinstance(value, config_lib.ArgFactory):
        paths = " , ".join(
            daglish.path_str(path) for path in state.get_all_paths()
        )
        raise TypeError(
            "fdl.ArgFactory instances should be inside fdl.Partial's, and "
            "appropriately lowered with the split_arg_factories pass. Either "
            "your config is malformed, or a previous codegen pass introduced "
            f"an error. Path to misformed object in codegen DAG: {paths}.\n\n("
            f"in function definition {fn_name}; `.output_value` "
            "indicates the value in the `return` statement.)"
        )
      else:
        raise TypeError(f"Unsupported Buildable {type(value)}")
    elif is_plain_symbol_or_enum_value(value):
      symbol = task.import_manager.add(value)
      return code_ir.SymbolReference(symbol)
    else:
      return state.map_children(value)

  for fn in task.top_level_call.all_fixture_functions():
    fn_name = fn.name.value
    fn.replace_with(daglish.MemoizedTraversal.run(traverse, fn))
