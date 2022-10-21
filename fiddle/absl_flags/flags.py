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

This implementation has drawn inspiration from multiple sources, including
[T5X](https://github.com/google-research/t5x), among others. Thank you!
"""

import ast
import dataclasses
import importlib
import inspect
import re
import sys
import textwrap
import typing
from typing import Any, List, Optional, Sequence, Union

from absl import app
from absl import flags
from absl import logging

from etils import epath

from fiddle import config
from fiddle import module_reflection
from fiddle import printing
from fiddle import selectors
from fiddle import tagging
from fiddle.experimental import auto_config
from fiddle.experimental import serialization

_FDL_CONFIG = flags.DEFINE_string(
    'fdl_config',
    default=None,
    help='The name of a function to construct a base Fiddle config.')
_FDL_CONFIG_FILE = epath.DEFINE_path(
    'fdl_config_file',
    default=None,
    help='The path to a file containing a serialized base Fiddle config in '
    'JSON format.')
_FIDDLER = flags.DEFINE_multi_string(
    'fiddler',
    default=[],
    help='The name of a fiddler. Fiddlers are functions that modify a config.')
_FDL_SET = flags.DEFINE_multi_string(
    'fdl_set',
    default=[],
    help='Per-parameter configuration settings. '
    'Typically accessed via the alias --fdl.foo.bar=123.')
_FDL_TAGS_SET = flags.DEFINE_multi_string(
    'fdl_tags_set',
    default=[],
    help='Fiddle tags setting, by name. Typically accessed via the alias '
    '--fdl_tag.foo.bar=123')
_FDL_HELP = flags.DEFINE_bool(
    'fdl_help',
    default=False,
    help='Print out Fiddle-specific help (including '
    'information on any loaded configuration) and exit.')


@dataclasses.dataclass
class _IndexKey:
  """Wraps a string or integer index into a dictionary or list/tuple."""
  key: Union[str, int]

  def __post_init__(self):
    try:
      self.key = ast.literal_eval(self.key)
    except Exception:  # pylint: disable=broad-except
      pass  # leave self as-is.

  def __call__(self, obj):
    return obj[self.key]

  def update(self, obj, new_value):
    """Updates `obj`'s `self.key` to `new_value`."""
    obj[self.key] = new_value


@dataclasses.dataclass
class _AttributeKey:
  """Wraps the name of an attribute."""
  __slots__ = ('name',)
  name: str

  def __call__(self, obj):
    return getattr(obj, self.name)

  def update(self, obj, new_value):
    setattr(obj, self.name, new_value)


_PATH_COMPONENT_REGEX = re.compile(r'(?:\.([A-Za-z_][^\.\[]*))|\[([^]]+)\]')


def _print_stderr(*args, **kwargs):
  print(*args, **kwargs, file=sys.stderr)


def _parse_path(path: str) -> List[Union[_AttributeKey, _IndexKey]]:
  """Parses a path into a list of either attributes or index lookups."""
  path = f'.{path}'  # Add a leading `.` to make parsing work properly.
  result = []
  curr_index = 0
  while curr_index < len(path):
    match = _PATH_COMPONENT_REGEX.match(path, curr_index)
    if match is None:
      raise ValueError(
          f'Could not parse {path[1:]!r} (failed index: {curr_index - 1}).')
    curr_index = match.end(0)  # Advance
    if match[1] is not None:
      result.append(_AttributeKey(match[1]))
    else:
      result.append(_IndexKey(match[2]))
  return result


def _import_dotted_name(name: str) -> Any:
  """Returns the Python object with the given dotted name.

  Args:
    name: The dotted name of a Python object, including the module name.

  Returns:
    The named value.

  Raises:
    ValueError: If `name` is not a dotted name.
    ModuleNotFoundError: If no dotted prefix of `name` can be imported.
    AttributeError: If the imported module does not contain a value with
      the indicated name.
  """
  name_pieces = name.split('.')
  if len(name_pieces) < 2:
    raise ValueError('Expected a dotted name including the module name.')

  # We don't know where the module ends and the name begins; so we need to
  # try different split points.  Longer module names take precedence.
  for i in range(len(name_pieces) - 1, 0, -1):
    try:
      value = importlib.import_module('.'.join(name_pieces[:i]))
      for name_piece in name_pieces[i:]:
        value = getattr(value, name_piece)  # Can raise AttributeError.
      return value
    except ModuleNotFoundError:
      if i == 1:  # Final iteration through the loop.
        raise

  # The following line should be unreachable -- the "if i == 1: raise" above
  # should have raised an exception before we exited the loop.
  raise ModuleNotFoundError(f'No module named {name_pieces[0]!r}')


def apply_overrides_to(cfg: config.Buildable):
  """Applies all command line flags to `cfg`."""
  for flag in _FDL_SET.value:
    path, value = flag.split('=', maxsplit=1)
    *parents, last = _parse_path(path)
    walk = typing.cast(Any, cfg)
    try:
      for parent in parents:
        walk = parent(walk)
    except Exception as e:
      raise ValueError(f'Invalid path "{path}".') from e
    try:
      last.update(walk, ast.literal_eval(value))
    except Exception as e:
      raise ValueError(f'Could not set "{path}" to "{value}".') from e


def _rewrite_fdl_args(args: Sequence[str]) -> List[str]:
  """Rewrites short-form Fiddle flags.

  There are two main rewrites:

    * `--fdl.NAME=VALUE` to `--fdl_set=NAME = VALUE`.
    * `--fdl_tag.NAME=VALUE` to `--fdl_tags_set=NAME = VALUE`.

  Args:
    args: Command-line args.

  Returns:
    Rewritten args.
  """

  def _rewrite(arg: str) -> str:
    if arg.startswith('--fdl.') or arg.startswith('--fdl_tag.'):
      if '=' not in arg:
        prefix = arg.split('.', maxsplit=1)[0]
        raise ValueError(
            f'Fiddle setting must be of the form `{prefix}.NAME=VALUE`; '
            f'`got: "{arg}".')
      if arg.startswith('--fdl.'):
        explicit_name = 'fdl_set'
      elif arg.startswith('--fdl_tag.'):
        explicit_name = 'fdl_tags_set'
      _, arg = arg.split('.', maxsplit=1)  # Strip --fdl. or --fdl_tag. prefix.
      path, value = arg.split('=', maxsplit=1)
      rewritten = f'--{explicit_name}={path}={value}'
      logging.debug('Rewrote flag "%s" to "%s".', arg, rewritten)
      return rewritten
    else:
      return arg

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


def set_tags(cfg: config.Buildable):
  """Sets tags based on their name, from CLI flags."""
  all_tags = tagging.list_tags(cfg, add_superclasses=True)
  for flag in _FDL_TAGS_SET.value:
    name, value = flag.split('=', maxsplit=1)
    matching_tags = [tag for tag in all_tags if tag.name == name]
    if not matching_tags:
      # TODO: Improve and unify these errors.
      loose_matches = [
          tag for tag in all_tags
          if name.lower().replace('_', '') in tag.name.lower().replace('_', '')
      ]
      did_you_mean = f' Did you mean {loose_matches}?' if loose_matches else ''
      raise ValueError(f'No tags with name {name!r} in config.{did_you_mean}')
    elif len(matching_tags) > 1:
      raise EnvironmentError(
          f'There were multiple tags with name {name!r}; perhaps some module '
          'reloading weirdness is going on? This should be very rare. Please '
          'make sure that you are not dynamically creating subclasses of '
          '`fdl.Tag` and using 2 instances with the same module and name '
          'in the configuration.')
    selectors.select(
        cfg, tag=matching_tags[0]).replace(value=ast.literal_eval(value))


def apply_fiddlers_to(cfg: config.Buildable,
                      source_module: Any,
                      allow_imports=False):
  """Applies fiddlers to `cfg`."""
  for fiddler_name in _FIDDLER.value:
    if hasattr(source_module, fiddler_name):
      fiddler = getattr(source_module, fiddler_name)
    elif allow_imports:
      try:
        fiddler = _import_dotted_name(fiddler_name)
      except (ValueError, ModuleNotFoundError, AttributeError) as e:
        raise ValueError(f'Could not load fiddler {fiddler_name!r}: {e}') from e
    else:
      available_fiddlers = ', '.join(
          module_reflection.find_fiddler_like_things(source_module))
      raise ValueError(
          f'No fiddler named {fiddler_name!r} found; available fiddlers: '
          f'{available_fiddlers}.')
    fiddler(cfg)


def _print_help(module, allow_imports):
  """Prints flag help, including available symbols in `module`."""
  flags_help_text = flags.FLAGS.module_help(sys.modules[__name__])
  flags_help_text = '\n'.join(flags_help_text.splitlines()[2:])
  _print_stderr('Fiddle-specific command line flags:')
  _print_stderr()
  _print_stderr(flags_help_text)

  if module is not None:
    source = f' in {module.__name__}' if hasattr(module, '__name__') else ''
    _print_stderr()
    _print_stderr(f'Available symbols (base configs or fiddlers){source}:')
    _print_stderr()
    for name in dir(module):
      obj = getattr(module, name)
      # Ignore private and module objects.
      if name.startswith('_') or inspect.ismodule(obj):
        continue
      if hasattr(obj, '__doc__') and obj.__doc__:
        obj_doc = '\n'.join(obj.__doc__.strip().split('\n\n')[:2])
        obj_doc = re.sub(r'\s+', ' ', obj_doc)
        docstring = f' - {obj_doc}'
      else:
        docstring = ''
      _print_stderr(f'  {name}{docstring}')

  if allow_imports:
    _print_stderr()
    _print_stderr('Base configs and fiddlers may be specified using '
                  'fully-qualified dotted names.')


def create_buildable_from_flags(
    module: Optional[Any],
    allow_imports=False,
    pyref_policy: Optional[serialization.PyrefPolicy] = None
) -> config.Buildable:
  """Returns a fdl.Buildable based on standardized flags.

  Args:
    module: A common namespace to use as the basis for finding configs and
      fiddlers. May be `None`; if `None`, only fully qualified Fiddler imports
      will be used (or alternatively a base configuration can be specified using
      the `--fdl_config_file` flag.)
    allow_imports: If true, then fully qualified dotted names may be used to
      specify configs or fiddlers that should be automatically imported.
    pyref_policy: An optional `serialization.PyrefPolicy` to use if parsing a
      serialized Fiddle config (passed via `--fdl_config_file`).
  """
  missing_base_config = not _FDL_CONFIG.value and not _FDL_CONFIG_FILE.value
  if not _FDL_HELP.value and missing_base_config:
    raise app.UsageError(
        'At least one of --fdl_config or --fdl_config_file is required.')
  elif _FDL_CONFIG.value and _FDL_CONFIG_FILE.value:
    raise app.UsageError(
        '--fdl_config and --fdl_config_file are mutually exclusive.')

  if _FDL_HELP.value:
    _print_help(module, allow_imports)
    if missing_base_config:
      sys.exit()
    _print_stderr()

  if module is None and not allow_imports:
    err_msg = (
        'A module must be passed to `create_buildable_from_flags` to use the '
        '`{flag}` flag unless `allow_imports` is `True`.')
    if _FDL_CONFIG.value:
      raise ValueError(err_msg.format(flag='--fdl_config'))
    if _FIDDLER.value:
      raise ValueError(err_msg.format(flag='--fiddler'))

  if _FDL_CONFIG.value:
    base_name = _FDL_CONFIG.value
    if hasattr(module, base_name):
      base_fn = getattr(module, base_name)
    elif allow_imports:
      try:
        base_fn = _import_dotted_name(base_name)
      except (ValueError, ModuleNotFoundError, AttributeError) as e:
        raise ValueError(
            f'Could not init a buildable from {base_name!r}: {e}') from e
    else:
      available_names = module_reflection.find_base_config_like_things(module)
      raise ValueError(f'Could not init a buildable from {base_name!r}; '
                       f'available names: {", ".join(available_names)}.')

    if auto_config.is_auto_config(base_fn):
      buildable = base_fn.as_buildable()
    else:
      buildable = base_fn()
  elif _FDL_CONFIG_FILE.value:
    with _FDL_CONFIG_FILE.value.open() as f:
      buildable = serialization.load_json(f.read(), pyref_policy=pyref_policy)
  else:
    raise AssertionError('This should be unreachable.')

  apply_fiddlers_to(
      buildable, source_module=module, allow_imports=allow_imports)
  set_tags(buildable)
  apply_overrides_to(buildable)

  if _FDL_HELP.value:
    _print_stderr('Tags (override as --fdl_tag.<name>=<value>):')
    _print_stderr()
    tags = tagging.list_tags(buildable)
    if tags:
      for tag in tags:
        _print_stderr(f'  {tag.name} - {tag.description}')
    else:
      print('  No tags present in config.')
    _print_stderr()
    _print_stderr('Config values (override as --fdl.<name>=<value>): ')
    _print_stderr()
    _print_stderr(textwrap.indent(printing.as_str_flattened(buildable), '  '))
    sys.exit()

  return buildable
