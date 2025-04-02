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

r"""A flag for specifying a 'sweep' of one or more fiddle configs to launch.

This flag supports the config:, set:, and fiddler: commands of the
DEFINE_fiddle_config flag. (See documentation for DEFINE_fiddle_config for
more details on those commands.)

Here we support an additional sweep: command, which allows you to specify
multiple configs by sweeping over any combination of:
* Arguments to the config function specified by config:
* Overrides to the resulting config.

A sweep: command should specify a function call returning a list of
dictionaries, where each dictionary represents a single item in the sweep.
The entries in the dictionary are the overrides to apply, where keys can be of
the form:
* kwarg:foo -- to specify or override a keyword argument to the config function
* arg:0 -- to specify or override a positional argument to the config function
* path.to.field -- to specify an override to a field in the resulting config
  returned by the config function. These paths follow the same format as is
  accepted by set: commands and can take quite general forms like
  foo.bar['baz'][0].boz.
  Like set: commands, these overrides are applied via mutating the config. If a
  single object is referenced in multiple places in your config, mutating it
  in one place will affect everywhere it is referenced. If you make separate
  copies of the same object in your config, you will need to mutate them all
  separately.
  If you prefer not to worry about this, consider using an argument to your
  config function instead, to set the same value in multiple places.

Multiple sweep: commands can be specified, which will result in taking the
product of the separate sweeps.

An example showing off the functionality, which hopefully documents most of it:

def model_config(hidden_size: int = 64, dropout_rate: float = 0.1):
  return fdl.Config(
      MyModel,
      hidden_size=hidden_size,
      dropout_rate=dropout_rate,
      layer_norm=True,
      log_level=logging.INFO,
      submodule=fdl.Config(
          Submodule,
          hidden_size=hidden_size,
          dropout_rate=dropout_rate,
          num_layers=10,
          log_level=logging.INFO,
      )
  )

def hidden_size_sweep():
  return [
      # Overrides kwarg to the config: function:
      {"kwarg:hidden_size": n} for n in [128, 256, 512]
  ]

def num_layers_sweep(max_layers: int = 5):
  return [
      {"submodule.num_layers": n} for n in range(2, max_layers + 1)
  ]

def add_debug_logging(config):
  config.log_level = logging.DEBUG
  config.submodule.log_level = logging.DEBUG

my_binary \
    --config 'config:model_config(dropout_rate=0.2)' \
    --config 'sweep:hidden_size_sweep' \
    --config 'sweep:num_layers_sweep(max_layers=10)' \
    --config 'set:layer_norm=True' \
    --config 'fiddler:add_debug_logging' \

This will apply the product of the two sweeps, varying both the arguments to
`model_config` and overriding the returned configs, as specified by the sweeps.
It then applies the set: and fiddler: commands (see fiddle.absl_flags) to all
the resulting configs.
"""

import dataclasses
import itertools
import re
import types
from typing import Any, Mapping, Optional, Sequence, Tuple

from absl import flags
from fiddle._src import config
from fiddle._src.absl_flags import utils
from fiddle._src.experimental import auto_config


_COMMAND_RE = re.compile(r"^(config|fiddler|set|sweep):(.+)$")
_KWARG_SWEEP_PREFIX = "kwarg:"
_ARG_SWEEP_PREFIX = "arg:"


@dataclasses.dataclass
class SweepItem:
  """A config, together with metadata.

  Attributes:
    config: A fdl.Buildable
    overrides_applied: A dictionary of overrides applied -- either to arguments
      of the config function or the config itself. Useful as metadata to
      distinguish this from other items in the sweep.
  """

  config: config.Buildable
  overrides_applied: Mapping[str, Any]


class DEFINE_fiddle_sweep:  # pylint: disable=invalid-name
  """Defines a flag for a sweep of one or more fiddle configs.

  Its .value property returns a list of SweepItem objects, each a config paired
  with the overrides applied to it.

  (While it has a .value like the FlagHolders returned by flags.DEFINE_*, it's
  not a 'real' FlagHolder, just a wrapper around a DEFINE_multi_string flag.
  This means the value appearing under flags.FLAGS will just be the strings from
  the multi_string flag.)
  """

  # Note flags.FiddleFlag goes through some contortions to subclass
  # flags.MultiFlag while ensuring lazy parsing, which the superclass is not
  # really built for. This approach of wrapping a plain multi-string flag is
  # simpler, but has the (mild) downside stated above.

  def __init__(
      self,
      name: str,
      required: bool = False,
      help: str = "Multi-flag for a fiddle config sweep.",  # pylint: disable=redefined-builtin
      default_module: Optional[types.ModuleType] = None,
      allow_imports: bool = True,
  ):
    self.name = name
    self._allow_imports = allow_imports
    self._default_module = default_module
    self._multi_flag = flags.DEFINE_multi_string(
        name=name,
        default=None,
        help=help,
        required=required,
    )
    self._value = None

  @property
  def value(self) -> Sequence[SweepItem]:
    if self._value is None:
      self._value = self._parse(self._multi_flag.value)
    return self._value

  def _parse_call_expression(
      self, expression: str, mode: utils.ImportDottedNameDebugContext
  ):
    """Parses a call expression as supported by fiddle.absl_flags."""
    call_expr = utils.CallExpression.parse(expression)
    base_name = call_expr.func_name
    base_fn = utils.resolve_function_reference(
        base_name,
        mode,
        self._default_module,
        self._allow_imports,
        failure_msg_prefix="Could not resolve reference from fiddle sweep_flag",
    )
    if auto_config.is_auto_config(base_fn):
      base_fn = base_fn.as_buildable
    return base_fn, call_expr.args, call_expr.kwargs

  def _parse_command(self, item: str):
    match = _COMMAND_RE.fullmatch(item)
    if not match:
      raise ValueError(
          f"All flag values to {self.name} must begin with 'config:', "
          "'set:', 'fiddler:' or 'sweep:'."
      )
    command, expression = match.groups()
    return command, expression

  def _parse(self, commands: Sequence[str]) -> Sequence[SweepItem]:
    """Parse a sequence of commands describing a config sweep."""

    command_type, expression = self._parse_command(commands[0])
    if command_type != "config":
      raise ValueError(
          'First command must be a "config:" command, specifying a call to a '
          "function returning a config. Got: "
          + commands[0]
      )

    config_fn, args, kwargs = self._parse_call_expression(
        expression, mode=utils.ImportDottedNameDebugContext.BASE_CONFIG
    )

    sweeps = []
    non_sweep_commands = []
    for command in commands[1:]:
      command_type, expr = self._parse_command(command)
      if command_type == "sweep":
        sweep_fn, sweep_args, sweep_kwargs = self._parse_call_expression(
            expr, mode=utils.ImportDottedNameDebugContext.SWEEP
        )
        sweep = sweep_fn(*sweep_args, **sweep_kwargs)
        sweeps.append(sweep)
      else:
        non_sweep_commands.append((command_type, expr))

    # Take product of all sweeps:
    sweep = [_merge_dicts(dicts) for dicts in itertools.product(*sweeps)]

    configs = [
        _apply_sweep_overrides(config_fn, args, kwargs, overrides)
        for overrides in sweep
    ]
    configs = self._apply_sets_and_fiddlers(configs, non_sweep_commands)
    return [
        SweepItem(config=config, overrides_applied=overrides)
        for config, overrides in zip(configs, sweep)
    ]

  def _apply_sets_and_fiddlers(
      self,
      configs: Sequence[config.Buildable],
      commands: Sequence[Tuple[str, str]],
  ) -> Sequence[config.Buildable]:
    """Apply set: and fiddler: commands to multiple configs. May mutate."""
    for command_type, expr in commands:
      if command_type == "set":
        for cfg in configs:
          utils.set_value(cfg, expr)
      elif command_type == "fiddler":
        fiddler, args, kwargs = self._parse_call_expression(
            expr, mode=utils.ImportDottedNameDebugContext.FIDDLER
        )
        new_configs = []
        for cfg in configs:
          new_cfg = fiddler(cfg, *args, **kwargs)
          if new_cfg is None:
            new_cfg = cfg  # Fiddler mutated it.
          new_configs.append(new_cfg)
        configs = new_configs
      else:
        raise ValueError(
            f"Unexpected command type {command_type} in {self.name}."
        )
    return configs


def _apply_sweep_overrides(config_fn, args, kwargs, overrides):
  """Apply overrides to the args/kwargs and the result of a config function."""
  kwargs = dict(kwargs)
  args = list(args)
  config_overrides = {}
  for key, value in overrides.items():
    if key.startswith(_KWARG_SWEEP_PREFIX):
      key = key[len(_KWARG_SWEEP_PREFIX) :]
      kwargs[key] = value
    elif key.startswith(_ARG_SWEEP_PREFIX):
      index = int(key[len(_ARG_SWEEP_PREFIX) :])
      # Pad long enough to accept the new arg:
      args.extend([None] * (index + 1 - len(args)))
      args[index] = value
    else:
      config_overrides[key] = value

  cfg = config_fn(*args, **kwargs)
  return utils.with_overrides(cfg, config_overrides)


def _merge_dicts(dicts: Sequence[Mapping[Any, Any]]) -> Mapping[Any, Any]:
  return {k: v for d in dicts for k, v in d.items()}  # pylint: disable=g-complex-comprehension
