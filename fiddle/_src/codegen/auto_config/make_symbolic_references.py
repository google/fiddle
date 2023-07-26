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
from typing import Any, Callable

from fiddle import arg_factory
from fiddle import daglish
from fiddle._src import config as config_lib
from fiddle._src import partial
from fiddle._src.codegen.auto_config import code_ir
from fiddle._src.codegen.auto_config import import_manager_wrapper


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
      # Import the tags too.
      for arg_tags in value.__argument_tags__.values():
        for tag in arg_tags:
          task.import_manager.add(tag)
    elif is_plain_symbol_or_enum_value(value):
      task.import_manager.add(value)


def noop_history_comments(unused_buildable):
  return code_ir.HistoryComments()


def replace_callables_and_configs_with_symbols(
    task: code_ir.CodegenTask,
    *,
    format_history: Callable[
        ..., code_ir.HistoryComments
    ] = noop_history_comments,
) -> None:
  """Replaces callables and Buildables with symbolic versions.

  Args:
    task: Codegen task.
    format_history: Function used to format history for a buildable. Set to
      get_history_comments.format_history_for_buildable (or a functools.partial
      variant) to populate histories.
  """

  fn_name = None

  def _handle_partial(
      value: partial.Partial,
      state: daglish.State,
      ir_for_symbol: code_ir.CodegenNode,
  ):
    """Split-out helper method to handle Partial() nodes."""
    arguments = config_lib.ordered_arguments(value)
    all_tags = value.__argument_tags__

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

    for dict_of_args in (arg_factory_args, regular_args):
      for arg in dict_of_args:
        if arg in all_tags:
          tags = all_tags[arg]
          tags_expr = [task.import_manager.add(tag) for tag in tags]
          dict_of_args[arg] = code_ir.WithTagsCall(
              tag_symbol_expressions=tags_expr,
              item_to_tag=dict_of_args[arg],
          )

    def _arg_factory_partial():
      return code_ir.SymbolOrFixtureCall(
          import_manager_wrapper.add(arg_factory.partial, task.import_manager),
          positional_arg_expressions=[ir_for_symbol],
          arg_expressions=arg_factory_args,
          history_comments=format_history(value),
      )

    if not arg_factory_args:
      # The common case: there are no arg_factory's, so just emit a functools
      # partial. We currently emit a functools partial even if there are no
      # arguments, because this will mean there's a fdl.Partial when we call
      # the auto_config fixture's as_buildable() method. If we got rid of the
      # functools.partial, then we couldn't configure any attributes.
      return code_ir.SymbolOrFixtureCall(
          import_manager_wrapper.add(functools.partial, task.import_manager),
          positional_arg_expressions=[ir_for_symbol],
          arg_expressions=regular_args,
          history_comments=format_history(value),
      )
    elif not regular_args:
      # There are only arg_factory args, so we can emit an arg_factory.partial.
      return _arg_factory_partial()
    else:
      # We have both functools.partial and arg_factory args. It doesn't matter
      # which order, but we need to emit both decorators. Go with functools
      # on the outer level.
      return code_ir.SymbolOrFixtureCall(
          import_manager_wrapper.add(functools.partial, task.import_manager),
          positional_arg_expressions=[_arg_factory_partial()],
          arg_expressions=regular_args,
          history_comments=format_history(value),
      )

  def traverse(value, state: daglish.State):
    if isinstance(value, config_lib.Buildable):
      ir_for_symbol = import_manager_wrapper.add(
          config_lib.get_callable(value), task.import_manager
      )
      if isinstance(value, config_lib.Config):
        all_tags = value.__argument_tags__
        value = state.map_children(value)
        for arg, arg_tags in all_tags.items():
          tag_expr = [task.import_manager.add(tag) for tag in arg_tags]
          if arg not in value.__arguments__:
            raise ValueError(
                f"Tagged field '{arg}' of {value!r} is not found in its"
                f" arguments: {value.__arguments__}. This is likely because the"
                " tagged field doesn't yet have a value. Consider assigning a"
                " value to the field first or removing field tags from your"
                " config, for example using `fdl.clear_tags`."
            )
          value.__arguments__[arg] = code_ir.WithTagsCall(
              tag_symbol_expressions=tag_expr,
              item_to_tag=value.__arguments__[arg],
          )
        return code_ir.SymbolOrFixtureCall(
            symbol_expression=ir_for_symbol,
            positional_arg_expressions=[],
            arg_expressions=config_lib.ordered_arguments(value),
            history_comments=format_history(value),
        )
      elif isinstance(value, partial.Partial):
        return _handle_partial(value, state, ir_for_symbol)
      elif isinstance(value, partial.ArgFactory):
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
      return import_manager_wrapper.add(value, task.import_manager)
    else:
      return state.map_children(value)

  for fn in task.top_level_call.all_fixture_functions():
    fn_name = fn.name.value
    fn.replace_with(daglish.MemoizedTraversal.run(traverse, fn))
