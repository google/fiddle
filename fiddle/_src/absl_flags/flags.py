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


class _FiddleFlag(flags.MultiFlag):
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

    # An immutable fiddler function is one that doesn't mutate the original
    # `fdl.Buildable` object that's passed into it and instead returns a new
    # `fdl.Buildable` built from the original `fdl.Buildable` after applying the
    # transform `fiddler`. It is different from a mutable fiddler function that
    # directly mutates the original `fdl.Buildable` object. Immutable fiddlers
    # are useful when using various util functions such as
    # `daglish.MemoizedTraversal.run()` that return a new `fdl.Buildable`
    # object. Immutable fiddler functions return a non-None value.
    new_cfg = fiddler(cfg, *call_expr.args, **call_expr.kwargs)

    # If the fiddler is immutable, then return the transformed `fdl.Buildable`
    # passed into the function. Else, return the mutated original
    # `fdl.Buildable` object.
    return new_cfg if new_cfg is not None else cfg

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
  r"""Declare and define a fiddle command line flag object.

  When used in a python binary, after the flags have been parsed from the
  command line, this command line flag object will have the `fdl.Config` object
  built.

  Example usage in a python binary:
  ```
  .... other imports ...
  from fiddle import absl_flags as fdl_flags

  _MY_FLAG = fdl_flags.DEFINE_fiddle_config(
      "my_config",
      help_string="My binary's fiddle config handle",
      default_module=sys.modules[__name__],
  )

  def base_config() -> fdl.Config:
    return some_import.fixture.as_buildable()

  def set_attributes(config, value: str):
    config.some_attr = value
    return config

  def main(argv) -> None:
    if len(argv) > 1:
      raise app.UsageError("Too many command-line arguments.")
    # _MY_FLAG.value contains the config object, built from `base_config()`
    # with any command line flags and overrides applied in the order passed in
    # the command line.
    some_import.do_something(_MY_FLAG.value.some_attr)

  if __name__ == "__main__":
    app.run(main)
  ```

  Invoking the above binary with:
  python3 -m path.to.my.binary --my_config=config:base_config \
  --my_config=fiddler:'set_attributes(value="float64")' \
  --my_config=set:some_other_attr.enable=True

  results in the `_MY_FLAG.value` set to the built config object with all the
  command line flags applied in the order they were passed in.

  Args:
    name: name of the command line flag.
    default: default value of the flag.
    help_string: help string describing what the flag does.
    default_module: the python module where this flag is defined.
    flag_values: the ``FlagValues`` instance with which the flag will be
      registered. This should almost never need to be overridden.

  Returns:
    A handle to defined flag.
  """
  return flags.DEFINE_flag(
      _FiddleFlag(
          name=name,
          default_module=default_module,
          default=default,
          parser=flags.ArgumentParser(),
          serializer=None,
          help_string=help_string,
      ),
      flag_values=flag_values,
  )
