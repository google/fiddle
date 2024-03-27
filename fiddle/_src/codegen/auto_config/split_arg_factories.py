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

"""Splits arg factories into ArgFactoryExpr(Partial(X)) expressions.

This lowering pass makes it a bit easier to pull things into variables, function
calls, etc. robustly. Though these passes still have to handle ArgFactoryExpr
correctly, but this should be less work than handling all ArgFactory edge cases
precisely each time.

For example, when lowering the ArgFactory's into
"""

import copy

from fiddle import daglish
from fiddle._src import casting
from fiddle._src import config as config_lib
from fiddle._src import partial
from fiddle._src.codegen.auto_config import code_ir


def lower_arg_factories(task: code_ir.CodegenTask) -> None:
  """Replaces fdl.ArgFactory(X) with ArgFactoryExpr(fdl.Partial(X))."""

  def traverse(value, state: daglish.State):
    def _convert_arg(arg_value):
      if isinstance(arg_value, partial.ArgFactory):
        return code_ir.ArgFactoryExpr(casting.cast(partial.Partial, arg_value))
      else:
        return arg_value

    if isinstance(value, partial.Partial):
      arguments = config_lib.ordered_arguments(value)
      arguments = {
          name: state.call(_convert_arg(arg), daglish.attr_or_index(name))
          for name, arg in arguments.items()
      }
      value = copy.copy(value)
      for key, arg_value in arguments.items():
        if isinstance(key, str):
          setattr(value, key, arg_value)
        elif isinstance(key, int):
          value[key] = arg_value
        else:
          raise TypeError(f'Unknown key type: {key}')
      return value
    else:
      return state.map_children(value)

  for fn in task.top_level_call.all_fixture_functions():
    fn.replace_with(daglish.MemoizedTraversal.run(traverse, fn))
