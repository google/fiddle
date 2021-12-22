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

"""Utilities to use command line flags with Fiddle Buildables.

While it's generally better to check in the full configuration into a source-
control system, there are some instances (e.g. hyperparameter sweeps) where it's
most effective to set initialization parameters (hyperparameters) from command
line flags. This library exposes functions you can call within your program.


## Quickstart

> Note: although different bits of functionality are designed to be usable
> piecemeal, this is the fastest way to get all flag-integrated features.


First, some definitions:

 - *A module for base configurations*: So that libraries can be easily reused
   with or without fiddle, code to define fiddle configurations are often put
   in a separate package / module. Most commonly, there is one function (called
   `default` by convention)
 - *Fiddlers*: A fiddler is a function that takes a `fdl.Buildable` and performs
   arbitrary mutations. These are commonly also grouped into the same module
   that defines the base configuration(s).
 - *Overrides*: Overrides are command line options that set values to
   parameters within the (nested) fiddle tree.

The simplest way to use the fiddle support for absl flags is to pass
`flags_parser=absl_flags.flags_parser` when calling `app.run`. Then, simply call
`cfg = absl_flags.create_buildable_from_flags(configs)` where `configs` is a
module for base configurations and fiddlers.

For example:


```py
## ---- config.py ----
import fiddle as fdl

def baseline() -> fdl.Config:  # A base configuration.
  my_model = fdl.Config(MyModel)
  my_optimizer = fdl.Config(MyOptimizer)
  return fdl.Config(MyTrainer, optimizer=my_optimizer, model=my_model)

def set_dtypes_to_bfloat_16(cfg: fdl.Config):  # A fiddler.
  cfg.model.dtype = 'bfloat16'  # Arbitrary python.

##  ---- main.py ----
from absl import app
import fiddle as fdl
from fiddle import absl_flags
import config

def main(argv):
  cfg = absl_flags.create_buildable_from_flags(configs)
  trainer = fdl.build(cfg)
  trainer.train()

if __name__ == '__main__':
  app.run(main, flags_parser=absl_flags.flags_parser)
```

## Overrides-only

If you simply want to use command line flags to override values in a config
built through arbitrary python, follow these steps:

 1. *main*: Pass `fdl.absl_flags.flags_parser` when calling `absl.app.run`. For
    example, your `app.run` call should look like the following:
        `app.run(my_main_function, flags_parser=fiddle.absl_flags.flags_parser)`
 2. *Augment a config*: Somewhere in your program, call
    `fdl.absl_flags.apply_overrides_to(cfg)` to apply settings defined on the
    command line to `cfg`. (This is often best done right before the call to
    `fdl.build(cfg)`.)
 3. Set values on the command line: `--fdl.my.property=3 --fdl.other.setting=5`

### Configuring multiple fiddle objects

If you have multiple fiddle objects that you'd like to configure on the command
line, create an empty top-level config and use that. Example:

```
my_program_part_1 = fdl.Config(...)
my_program_part_2 = fdl.Config(...)

top_level = fiddle.experimental.DictConfig(part1=my_program_part_1)
top_level.part2 = my_program_part_2
fdl.absl_flags.apply_overrides_to(top_level)

run_part_1(fdl.build(my_program_part_1))
run_part_2(fdl.build(my_program_part_2))
```

## Acknowledgements

This implementation has drawn inspiration from multiple sources, including:

 - [T5X](https://github.com/google-research/t5x)
 - JAX Borealis
 - And many others!

Thank you!
"""

import ast
import sys
from typing import Any, List, Sequence

from absl import app
from absl import flags
from absl import logging

from fiddle import config
from fiddle import module_reflection
from fiddle import printing

flags.DEFINE_string(
    'fdl_config',
    default=None,
    help='The name of the no-arity function to construct a fdl.Buildable.')
flags.DEFINE_multi_string(
    'fiddler', default=[], help='Fiddlers are functions that tweak a config.')
flags.DEFINE_multi_string(
    'fdl_set', default=[], help='Fiddle configuration settings.')
flags.DEFINE_bool(
    'fdl_help', False, help='Print out the flags-built config and exit.')


def apply_overrides_to(cfg: config.Buildable):
  """Applies all command line flags to `cfg`."""
  for flag in flags.FLAGS.fdl_set:
    path, value = flag.split(' = ', maxsplit=1)
    *parents, name = path.split('.')  # TODO: Support list indexes.
    walk = cfg
    try:
      for parent in parents:
        walk = getattr(walk, parent)
    except Exception:  # pylint: disable=broad-except
      raise ValueError(f'Invalid path "{path}".')
    try:
      setattr(walk, name, ast.literal_eval(value))
    except:
      raise ValueError(f'Could not set "{value}" path "{path}".')


def _rewrite_fdl_args(args: Sequence[str]) -> List[str]:
  """Rewrites `--fdl.NAME=VALUE` to `--fdl_set=NAME = VALUE`."""

  def _rewrite(arg: str) -> str:
    if not arg.startswith('--fdl.'):
      return arg
    if '=' not in arg:
      raise ValueError(
          'Fiddle settings must be of the form `--fdl.NAME=VALUE`; got: '
          f'"{arg}".')
    arg = arg[len('--fdl.'):]  # Strip --fdl. prefix.
    path, value = arg.split('=', maxsplit=1)
    rewritten = f'--fdl_set={path} = {value}'
    logging.debug('Rewrote flag "%s" to "%s".', arg, rewritten)
    return rewritten

  return list(map(_rewrite, args))


def flags_parser(args: Sequence[str]):
  """Flag parser.

  See absl.app.parse_flags_with_usage and absl.app.main(..., flags_parser).

  Args:
    args: All command line arguments.

  Returns:
    Whatever `absl.app.parse_flags_with_usage` returns. Sorry!
  """
  return app.parse_flags_with_usage(_rewrite_fdl_args(args))


def apply_fiddlers_to(cfg: config.Buildable, source_module: Any):
  """Applies fiddlers to `cfg`."""
  # TODO: Consider allowing arbitrary imports using importlib.
  for fiddler_name in flags.FLAGS.fiddler:
    if not hasattr(source_module, fiddler_name):
      available_fiddlers = ', '.join(
          module_reflection.find_fiddler_like_things(source_module))
      raise ValueError(
          f'No fiddler named {fiddler_name} found; available fiddlers: '
          f'{available_fiddlers}.')
    fiddler = getattr(source_module, fiddler_name)
    fiddler(cfg)


def create_buildable_from_flags(module: Any) -> config.Buildable:
  """Returns a fdl.Buildable based on standardized flags.

  Args:
    module: A common namespace to use as the basis for finding configs and
      fiddlers.
  """
  # TODO: Explore allowing arbitrary imports.
  base_name = flags.FLAGS.fdl_config
  if not hasattr(module, base_name):
    available_names = module_reflection.find_base_config_like_things(module)
    raise ValueError(f'Could not init a buildable from {base_name}; '
                     f'available names: {", ".join(available_names)}.')
  buildable = getattr(module, base_name)()
  apply_fiddlers_to(buildable, source_module=module)
  apply_overrides_to(buildable)
  if flags.FLAGS.fdl_help:
    printing.as_str_flattened(buildable)
    sys.exit()
  return buildable
