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

"""API to use command line flags with Fiddle Buildables."""

import re
import types
from typing import Any, Optional, TypeVar

from absl import flags
from fiddle._src import config
from fiddle._src.absl_flags import legacy_flags
from fiddle._src.absl_flags import utils
from fiddle._src.experimental import auto_config


# Legacy API aliases
apply_fiddlers_to = legacy_flags.apply_fiddlers_to
apply_overrides_to = legacy_flags.apply_overrides_to
create_buildable_from_flags = legacy_flags.create_buildable_from_flags
flags_parser = legacy_flags.flags_parser
rewrite_fdl_args = legacy_flags.rewrite_fdl_args
fdl_flags_supplied = legacy_flags.fdl_flags_supplied

# New API
_COMMAND_RE = re.compile(r"^(config|fiddler|set):(.+)$")
_F = TypeVar("_F")


class FiddleFlag(flags.MultiFlag):
  """ABSL flag class for a Fiddle config flag."""

  def __init__(
      self,
      *args,
      default_module: Optional[types.ModuleType] = None,
      allow_imports: bool = True,
      **kwargs,
  ):
    self.allow_imports = allow_imports
    self.default_module = default_module
    self.first_command = None
    self._initial_config_expression = None
    super().__init__(*args, **kwargs)

  def _initial_config(self, expression: str):
    call_expr = utils.CallExpression.parse(expression)
    base_name = call_expr.func_name
    base_fn = utils.resolve_function_reference(
        base_name,
        utils.ImportDottedNameDebugContext.BASE_CONFIG,
        self.default_module,
        self.allow_imports,
        "Could not init a buildable from",
    )
    if auto_config.is_auto_config(base_fn):
      return base_fn.as_buildable(*call_expr.args, **call_expr.kwargs)
    else:
      return base_fn(*call_expr.args, **call_expr.kwargs)

  def _apply_fiddler(self, cfg: config.Buildable, expression: str):
    call_expr = utils.CallExpression.parse(expression)
    base_name = call_expr.func_name
    fiddler = utils.resolve_function_reference(
        base_name,
        utils.ImportDottedNameDebugContext.FIDDLER,
        self.default_module,
        self.allow_imports,
        "Could not init a buildable from",
    )
    return fiddler(cfg, *call_expr.args, **call_expr.kwargs)

  def parse(self, argument):
    parsed = self._parse(argument)
    for item in parsed:
      match = _COMMAND_RE.fullmatch(item)
      if not match:
        raise ValueError(
            f"All flag values to {self.name} must begin with 'config:', "
            "'set:', or 'fiddler:'."
        )
      command, expression = match.groups()

      if self.first_command is None:
        if command != "config":
          raise ValueError(
              "First flag command must specify the input config. Received"
              f" command: {command} instead."
          )
        self.first_command = command

      if command == "config":
        if self._initial_config_expression:
          raise ValueError(
              "Only one base configuration is permitted. Received"
              f" {expression} after {self._initial_config_expression} was"
              " already provided."
          )
        else:
          self._initial_config_expression = expression
        self.value = self._initial_config(expression)
      elif command == "set":
        utils.set_value(self.value, expression)
      elif command == "fiddler":
        self.value = self._apply_fiddler(self.value, expression)
      else:
        raise AssertionError("Internal error; should not be reached.")


def DEFINE_fiddle_config(  # pylint: disable=invalid-name
    name: str,
    *,
    default: Any = None,
    help_string: str,
    default_module: Optional[types.ModuleType] = None,
    flag_values: flags.FlagValues = flags.FLAGS,
) -> flags.FlagHolder[Any]:
  return flags.DEFINE_flag(
      FiddleFlag(
          name=name,
          default_module=default_module,
          default=default,
          parser=flags.ArgumentParser(),
          serializer=None,
          help_string=help_string,
      ),
      flag_values=flag_values,
  )
