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

import dataclasses
import re
import types
from typing import Any, List, Optional, Text, TypeVar, Union

from absl import flags
from etils import epath
from fiddle._src import config
from fiddle._src.absl_flags import legacy_flags
from fiddle._src.absl_flags import utils
from fiddle._src.experimental import auto_config
from fiddle._src.experimental import serialization

# Legacy API aliases
apply_fiddlers_to = legacy_flags.apply_fiddlers_to
apply_overrides_to = legacy_flags.apply_overrides_to
create_buildable_from_flags = legacy_flags.create_buildable_from_flags
flags_parser = legacy_flags.flags_parser
rewrite_fdl_args = legacy_flags.rewrite_fdl_args
fdl_flags_supplied = legacy_flags.fdl_flags_supplied

# New API
_COMMAND_RE = re.compile(r"^(config|config_file|config_str|fiddler|set):(.+)$")
_F = TypeVar("_F")
_BASE_CONFIG_DIRECTIVES = {"config", "config_file", "config_str"}


class FiddleFlagSerializer(flags.ArgumentSerializer):
  """ABSL serializer for a Fiddle config flag.

  This class can be provided to `FiddleFlag` as a serializer
  to serialize a fiddle config to a single command line argument.

  Example usage:
  ```
  from absl import flags
  from fiddle import absl_flags as fdl_flags

  FLAGS = flags.FLAGS

  flags.DEFINE_flag(
    fdl_flags.FiddleFlag(
        name="config",
        default=None,
        parser=flags.ArgumentParser(),
        serializer=FiddleFlagSerializer(),
        help_string="My fiddle flag",
    ),
    flag_values=FLAGS,
  )

  serialized_flag = FLAGS['config'].serialize()
  ```

  Users will more commonly use `fdl_flags.DEFINE_fiddle_config`
  which provides this serializer by default.
  """

  def __init__(self, pyref_policy: serialization.PyrefPolicy):
    self._pyref_policy = pyref_policy

  def serialize(self, value: config.Buildable) -> str:
    serializer = utils.ZlibJSONSerializer()
    serialized = serializer.serialize(value, pyref_policy=self._pyref_policy)
    return f"config_str:{serialized}"


@dataclasses.dataclass
class _LazyFlagValue:
  """Represents a lazily evaluated Fiddle flag value.

  This is separate from FiddleFlag because it is used by both defaults and
  provided flags.

  Lazy flag values are useful because they allow other parts of the system to
  be set up, so things like logging can be configured before a configuration is
  loaded.
  """

  flag_name: str
  remaining_directives: List[str] = dataclasses.field(default_factory=list)
  first_command: Optional[str] = None
  initial_config_expression: Optional[str] = None

  default_module: Optional[types.ModuleType] = None
  allow_imports: bool = True
  pyref_policy: Optional[serialization.PyrefPolicy] = None

  # Only set internally, please use get_value() / set_value().
  _value: Optional[Any] = None

  def _initial_config(self, expression: str):
    """Generates the initial config from a config:<expression> directive."""
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
    """Modifies the config from the given CLI flag."""
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

  def _parse_config(self, command: str, expression: str) -> None:
    """Sets the initial config from the given CLI flag/directive."""
    if self.initial_config_expression:
      raise ValueError(
          "Only one base configuration is permitted. Received"
          f"{command}:{expression} after "
          f"{self.first_command}:{self.initial_config_expression} was"
          " already provided."
      )
    else:
      self.initial_config_expression = expression
    if command == "config":
      self._value = self._initial_config(expression)
    elif command == "config_file":
      with epath.Path(expression).open() as f:
        self._value = serialization.load_json(
            f.read(), pyref_policy=self.pyref_policy
        )
    elif command == "config_str":
      serializer = utils.ZlibJSONSerializer()
      self._value = serializer.deserialize(
          expression, pyref_policy=self.pyref_policy
      )

  def get_value(self):
    """Gets the current value (parsing any directives)."""
    while self.remaining_directives:
      # Pop already processed `directive` so that _value won't be updated twice
      # by the same argument.
      item = self.remaining_directives.pop(0)
      match = _COMMAND_RE.fullmatch(item)
      if not match:
        raise ValueError(
            f"All flag values to {self.flag_name} must begin with 'config:', "
            "'config_file:', 'config_str:', 'set:', or 'fiddler:'."
        )
      command, expression = match.groups()

      if self.first_command is None:
        if command not in _BASE_CONFIG_DIRECTIVES:
          raise ValueError(
              "First flag command must specify the input config via either "
              "config or config_file or config_str commands. "
              f"Received command: {command} instead. If you have a default "
              "value set, you must re-provide that on the CLI before setting "
              "values or running fiddlers."
          )
        self.first_command = command

      if (
          command == "config"
          or command == "config_file"
          or command == "config_str"
      ):
        self._parse_config(command, expression)

      elif command == "set":
        utils.set_value(self._value, expression)
      elif command == "fiddler":
        self._value = self._apply_fiddler(self._value, expression)
      else:
        raise AssertionError("Internal error; should not be reached.")
    return self._value

  def set_value(self, value: Any):
    self._value = value


class FiddleFlag(flags.MultiFlag):
  """ABSL flag class for a Fiddle config flag.

  This class is used to parse command line flags to construct a Fiddle `Config`
  object with certain transformations applied as specified in the command line
  flags.

  Most users should rely on the `DEFINE_fiddle_config()` API below. Using this
  class directly provides flexibility to users to parse Fiddle flags themselves
  programmatically. Also see the documentation for `DEFINE_fiddle_config()`
  below.

  Example usage where this flag is parsed from existing flag:
  ```
  from fiddle import absl_flags as fdl_flags

  _MY_CONFIG = fdl_flags.DEFINE_multi_string(
      "my_config",
      "Name of the fiddle config"
  )

  fiddle_flag = fdl_flags.FiddleFlag(
      name="config",
      default_module=my_module,
      default=None,
      parser=flags.ArgumentParser(),
      serializer=None,
      help_string="My fiddle flag",
  )
  fiddle_flag.parse(_MY_CONFIG.value)
  config = fiddle_flag.value
  ```
  """

  def __init__(
      self,
      *args,
      name: Text,
      default_module: Optional[types.ModuleType] = None,
      allow_imports: bool = True,
      pyref_policy: Optional[serialization.PyrefPolicy] = None,
      **kwargs,
  ):
    self.allow_imports = allow_imports
    self.default_module = default_module
    self._pyref_policy = pyref_policy
    self._lazy_default = _LazyFlagValue(
        flag_name=name,
        default_module=default_module,
        allow_imports=allow_imports,
        pyref_policy=pyref_policy,
    )
    self._lazy_value = _LazyFlagValue(
        flag_name=name,
        default_module=default_module,
        allow_imports=allow_imports,
        pyref_policy=pyref_policy,
    )
    kwargs["name"] = name
    super().__init__(*args, **kwargs)

  def parse(self, arguments):
    new_parsed = self._parse(arguments)
    self._lazy_value.remaining_directives.extend(new_parsed)
    self.present += len(new_parsed)

  def _parse_from_default(
      self, value: Union[Text, List[Any]]
  ) -> Optional[List[Any]]:
    lazy_default_value = _LazyFlagValue(
        flag_name=self.name,
        default_module=self.default_module,
        allow_imports=self.allow_imports,
        pyref_policy=self._pyref_policy,
    )
    value = self._parse(value)
    assert isinstance(value, list)
    lazy_default_value.remaining_directives.extend(value)
    return lazy_default_value  # pytype: disable=bad-return-type

  def unparse(self) -> None:
    self.value = self.default
    self.using_default_value = True
    # Reset it so that all `directives` not being processed yet will be
    # discarded.
    self._lazy_value.remaining_directives = []
    self.present = 0

  def _serialize(self, value) -> str:
    # Skip MultiFlag serialization as we don't truly have a multi-flag.
    # This will invoke Flag._serialize
    return super(flags.MultiFlag, self)._serialize(value)

  @property
  def value(self):
    return self._lazy_value.get_value()

  @value.setter
  def value(self, value):
    self._lazy_value.set_value(value)

  @property
  def default(self):
    return self._lazy_default.get_value()

  @default.setter
  def default(self, value):
    if isinstance(value, _LazyFlagValue):
      # Note: This is only for _set_default(). We might choose to override that
      # instead of just _parse_from_default(), in which case this branch can be
      # removed.
      self._lazy_default = value
    else:
      self._lazy_default.set_value(value)


def DEFINE_fiddle_config(  # pylint: disable=invalid-name
    name: str,
    *,
    default_flag_str: Optional[str] = None,
    help_string: str,
    default_module: Optional[types.ModuleType] = None,
    pyref_policy: Optional[serialization.PyrefPolicy] = None,
    flag_values: flags.FlagValues = flags.FLAGS,
    required: bool = False,
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

  To load a Fiddle config from a serialized file:
  python3 -m path.to.my.binary --my_config=config_file:path/to/file

  Args:
    name: Name of the command line flag.
    default_flag_str: Default value of the flag.
    help_string: Help string describing what the flag does.
    default_module: The python module where this flag is defined.
    pyref_policy: A policy for importing references to Python objects.
    flag_values: The ``FlagValues`` instance with which the flag will be
      registered. This should almost never need to be overridden.
    required: bool, is this a required flag. This must be used as a keyword
      argument.

  Returns:
    A handle to defined flag.
  """
  return flags.DEFINE_flag(
      FiddleFlag(
          name=name,
          default_module=default_module,
          default=default_flag_str,
          pyref_policy=pyref_policy,
          parser=flags.ArgumentParser(),
          serializer=FiddleFlagSerializer(pyref_policy=pyref_policy),
          help_string=help_string,
      ),
      flag_values=flag_values,
      required=required,
  )
