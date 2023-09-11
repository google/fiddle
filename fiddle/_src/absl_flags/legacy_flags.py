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

"""Legacy API to use command line flags with Fiddle Buildables.

NOTE: Deprecation: These are legacy APIs. Please refer to the new API usage in
flags.py and find the documentation in flags_code_lab.md.

While it's generally better to check in the full configuration into a source-
control system, there are some instances (e.g. hyperparameter sweeps) where it's
most effective to set initialization parameters (hyperparameters) from command
line flags. This library exposes functions you can call within your program.

NOTE: The flags aren't applied in the order they are passed in on the command
  line. The order followed is:
  - all fiddlers are applied first, followed by
  - all tags, followed by
  - all overrides.

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

import inspect
import re
import sys
import textwrap
from typing import Any, List, Optional, Sequence

from absl import app
from absl import flags
from absl import logging
from etils import epath
from fiddle import printing
from fiddle import selectors
from fiddle._src import config
from fiddle._src import tagging
from fiddle._src.absl_flags import utils
from fiddle._src.experimental import auto_config
from fiddle._src.experimental import serialization


_FDL_CONFIG = flags.DEFINE_string(
    'fdl_config',
    default=None,
    help='The name of a function to construct a base Fiddle config.')
_FDL_CONFIG_FILE = epath.DEFINE_path(
    'fdl_config_file',
    default=None,
    help=(
        'The path to a file containing a serialized base Fiddle config in '
        'JSON format. '
    ),
)
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


def _print_stderr(*args, **kwargs):
  print(*args, **kwargs, file=sys.stderr)


def apply_overrides_to(cfg: config.Buildable):
  """[DEPRECATED] Applies all command line flags to `cfg`.

  Deprecation: This is a legacy API. Please refer to the new API usage in
  flags.py and find the documentation in flags_code_lab.md.

  Args:
    cfg: The configuration to apply overrides to.
  """
  for flag in _FDL_SET.value:
    utils.set_value(cfg, flag)


def rewrite_fdl_args(args: Sequence[str]) -> List[str]:
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
  return app.parse_flags_with_usage(rewrite_fdl_args(args))


def set_tags(cfg: config.Buildable):
  """[DEPRECATED] Sets tags based on their name, from CLI flags.

  Deprecation: This is a legacy API. Please refer to the new API usage in
  flags.py and find the documentation in flags_code_lab.md.

  Args:
    cfg: The configuration to set tags on.

  Raises:
    ValueError: If no tags with the given name are found.
    EnvironmentError: If multiple tags with the given name are found.
  """
  all_tags = tagging.list_tags(cfg, add_superclasses=True)
  for flag in _FDL_TAGS_SET.value:
    name, value = flag.split('=', maxsplit=1)
    matching_tags = [tag for tag in all_tags if tag.name == name]
    if not matching_tags:
      # TODO(b/219988937): Improve and unify these errors.
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
    selectors.select(cfg, tag=matching_tags[0]).replace(
        value=utils.parse_value(value=value, path=str(matching_tags[0]))
    )


def apply_fiddlers_to(cfg: config.Buildable,
                      source_module: Any,
                      allow_imports=False):
  """[DEPRECATED] Applies fiddlers to `cfg`.

  Deprecation: This is a legacy API. Please refer to the new API usage in
  flags.py and find the documentation in flags_code_lab.md.

  Args:
    cfg: The configuration to apply fiddlers to.
    source_module: source py module where fiddlers are defined.
    allow_imports: If true, then fully qualified dotted names may be used to
      specify configs or fiddlers that should be automatically imported.
  """
  for fiddler_value in _FIDDLER.value:
    call_expr = utils.CallExpression.parse(fiddler_value)
    fiddler_name = call_expr.func_name
    fiddler = utils.resolve_function_reference(
        fiddler_name,
        utils.ImportDottedNameDebugContext.FIDDLER,
        source_module,
        allow_imports,
        'Could not load fiddler',
    )
    fiddler(cfg, *call_expr.args, **call_expr.kwargs)


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
    _print_stderr(
        'Base configs and fiddlers may be specified using '
        'fully-qualified dotted names.'
    )


def fdl_flags_supplied() -> bool:
  """Returns True if if required Fiddle flags are defined."""
  config_defined = bool(_FDL_CONFIG.value or _FDL_CONFIG_FILE.value)
  return config_defined or _FDL_HELP.value


def create_buildable_from_flags(
    module: Optional[Any],
    allow_imports=False,
    pyref_policy: Optional[serialization.PyrefPolicy] = None,
) -> config.Buildable:
  """[DEPRECATED] Returns a fdl.Buildable based on standardized flags.

  Deprecation: This is a legacy API. Please refer to the new API usage in
  flags.py and find the documentation in flags_code_lab.md.

  NOTE: the flags aren't applied in the order they are passed in on the command
  line. The order followed is:
  - all fiddlers are applied first, followed by
  - all tags, followed by
  - all overrides.

  Args:
    module: A common namespace to use as the basis for finding configs and
      fiddlers. May be `None`; if `None`, only fully qualified Fiddler imports
      will be used (or alternatively a base configuration can be specified using
      the `--fdl_config_file` flag.)
    allow_imports: If true, then fully qualified dotted names may be used to
      specify configs or fiddlers that should be automatically imported.
    pyref_policy: An optional `serialization.PyrefPolicy` to use if parsing a
      serialized Fiddle config (passed via `--fdl_config_file`).

  Returns:
    A `fdl.Buildable` based on standardized flags.
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
    call_expr = utils.CallExpression.parse(_FDL_CONFIG.value)
    base_name = call_expr.func_name

    base_fn = utils.resolve_function_reference(
        base_name,
        utils.ImportDottedNameDebugContext.BASE_CONFIG,
        module,
        allow_imports,
        'Could not init a buildable from',
    )

    if auto_config.is_auto_config(base_fn):
      buildable = base_fn.as_buildable(*call_expr.args, **call_expr.kwargs)
    else:
      buildable = base_fn(*call_expr.args, **call_expr.kwargs)
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
