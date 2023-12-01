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

"""Utilities to use command line flags with Fiddle Buildables."""

import ast
import base64
import dataclasses
import enum
import importlib
import re
import types
import typing
from typing import Any, Dict, Optional, Tuple
import zlib

from fiddle._src import config
from fiddle._src import daglish
from fiddle._src import daglish_extensions
from fiddle._src import module_reflection
from fiddle._src.experimental import serialization


class ImportDottedNameDebugContext(enum.Enum):
  """Context of importing a sumbol, for error messages."""

  BASE_CONFIG = 1
  FIDDLER = 2

  def error_prefix(self, name: str) -> str:
    if self == ImportDottedNameDebugContext.BASE_CONFIG:
      return f'Could not init a buildable from {name!r}'
    assert self == ImportDottedNameDebugContext.FIDDLER
    return f'Could not load fiddler {name!r}'


def import_dotted_name(name: str, mode: ImportDottedNameDebugContext) -> Any:
  """Returns the Python object with the given dotted name.

  Args:
    name: The dotted name of a Python object, including the module name.
    mode: Whether we're looking for a base config function or a fiddler.

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
    raise ValueError(
        f'{mode.error_prefix(name)}: Expected a dotted name including the '
        'module name.'
    )

  # We don't know where the module ends and the name begins; so we need to
  # try different split points.  Longer module names take precedence.
  for i in range(len(name_pieces) - 1, 0, -1):
    try:
      value = importlib.import_module('.'.join(name_pieces[:i]))
      for j, name_piece in enumerate(name_pieces[i:]):
        try:
          value = getattr(value, name_piece)  # Can raise AttributeError.
        except AttributeError:
          available_names = ', '.join(
              repr(n) for n in dir(value) if not n.startswith('_')
          )
          module_name = '.'.join(name_pieces[: i + j])
          failing_name = name_pieces[i + j]
          raise ValueError(
              f'{mode.error_prefix(name)}: module {module_name!r} has no '
              f'attribute {failing_name!r}; available names: {available_names}'
          ) from None
      return value
    except ModuleNotFoundError:
      if i == 1:  # Final iteration through the loop.
        raise

  # The following line should be unreachable -- the "if i == 1: raise" above
  # should have raised an exception before we exited the loop.
  raise ModuleNotFoundError(f'No module named {name_pieces[0]!r}')


@dataclasses.dataclass(frozen=True)
class CallExpression:
  """Parsed components of a call expression (or bare function name).

  Examples:

  >>> CallExpression.parse("fn('foo', False, x=[1, 2])")
  CallExpression(func_name='fn', args=('foo', False'), kwargs={'x': [1, 2]})
  >>> CallExpression.parse("fn")  # Bare function name: empty args/kwargs.
  CallExpression(func_name='fn', args=()), kwargs={})

  Attributes:
    func_name: The name fo the function that should be called.
    args: Parsed values of positional arguments for the function.
    kwargs: Parsed values of keyword arguments for the function.
  """

  func_name: str
  args: Optional[Tuple[Any, ...]]
  kwargs: Optional[Dict[str, Any]]

  _PARSE_RE = re.compile(r'(?P<func_name>[\w\.]+)(?:\((?P<args>.*)\))?')

  @classmethod
  def parse(cls, value: str) -> 'CallExpression':
    """Returns a CallExpression parsed from a string.

    Args:
      value: A string containing positional and keyword arguments for a
        function.  Must consist of an open paren followed by comma-separated
        argument values followed by a close paren.  Argument values must be
        literal constants (i.e., must be parsable with `ast.literal_eval`).
        var-positional and var-keyword arguments are not supported.

    Raises:
      SyntaxError: If `value` is not a simple call expression with literal
        arguments.
    """
    m = re.fullmatch(cls._PARSE_RE, value)
    if m is None:
      raise SyntaxError(
          f'Expected a function name or call expression; got: {value!r}'
      )
    if m.group('args') is None:  # Bare function name
      return CallExpression(m.group('func_name'), (), {})

    node = ast.parse(value)  # Can raise SyntaxError.
    if not (
        isinstance(node, ast.Module)
        and len(node.body) == 1
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Call)
    ):
      raise SyntaxError(
          f'Expected a function name or call expression; got: {value!r}'
      )
    call_node = node.body[0].value
    args = []
    for arg in call_node.args:
      if isinstance(arg, ast.Starred):
        raise SyntaxError('*args is not supported.')
      try:
        args.append(ast.literal_eval(arg))
      except ValueError as exc:
        raise SyntaxError(
            'Expected arguments to be simple literals; got '
            f'{ast.unparse(arg)!r} (while parsing {value!r})'
        ) from exc
    kwargs = {}
    for kwarg in call_node.keywords:
      if kwarg.arg is None:
        raise SyntaxError('**kwargs is not supported.')
      try:
        kwargs[kwarg.arg] = ast.literal_eval(kwarg.value)
      except ValueError as exc:
        raise SyntaxError(
            'Expected arguments to be simple literals; got '
            f'{ast.unparse(kwarg.value)!r} (while parsing {value!r})'
        ) from exc
    return CallExpression(m.group('func_name'), tuple(args), kwargs)


def parse_path(path: str) -> daglish.Path:
  """Parses a path into a list of either attributes or index lookups."""
  if not path.startswith('[') and not path.startswith('.'):
    path = f'.{path}'  # Add a leading `.` to make parsing work properly.

  return daglish_extensions.parse_path(path)


def parse_value(value: str, path: str) -> Any:
  """Parses a string `value` (e.g. from the command line) to a Python type."""

  # Apply a few rules that are unambiguous and would otherwise just cause
  # spurious failures (e.g. `--fdl_set=use_cached=false`).
  if value.lower() == 'false':
    return False
  elif value.lower() == 'true':
    return True

  try:
    return ast.literal_eval(value)
  except Exception as e:
    raise ValueError(
        f'Could not parse literal value "{value}" while trying to set '
        f'"{path}". Does a string need to be quoted? For example, you might '
        f"want to pass --fdl_set={path}='{value}' instead of "
        f'--fdl_set={path}={value}.'
    ) from e


def resolve_function_reference(
    function_name: str,
    mode: ImportDottedNameDebugContext,
    module: Optional[types.ModuleType],
    allow_imports: bool,
    failure_msg_prefix: str,
):
  """Returns function that produces `fdl.Buildable` from its name.

  Args:
    function_name: The name of the function.
    mode: Whether we're looking for a base config function or a fiddler.
    module: A common namespace to use as the basis for finding configs and
      fiddlers. May be `None`; if `None`, only fully qualified Fiddler imports
      will be used (or alternatively a base configuration can be specified using
      the `--fdl_config_file` flag.)
    allow_imports: If true, then fully qualified dotted names may be used to
      specify configs or fiddlers that should be automatically imported.
    failure_msg_prefix: Prefix string to prefix log messages in case of
      failures.

  Returns:
    The named value.
  """
  if hasattr(module, function_name):
    return getattr(module, function_name)
  elif allow_imports:
    try:
      return import_dotted_name(
          function_name,
          mode=mode,
      )
    except ModuleNotFoundError as e:
      raise ValueError(f'{failure_msg_prefix} {function_name!r}: {e}') from e
  else:
    available_names = module_reflection.find_base_config_like_things(module)
    raise ValueError(
        f'{failure_msg_prefix} {function_name!r}; '
        f'available names: {", ".join(available_names)}.'
    )


def set_value(cfg: config.Buildable, assignment: str) -> None:
  """Set an attribute's value.

  Args:
    cfg: A `fdl.Buildable` whose attribute is to be overridden.
    assignment: String representing attribute's override expression. Of the form
      `attribute=value`.
  """
  path, value = assignment.split('=', maxsplit=1)
  *parents, last = parse_path(path)

  walk = typing.cast(Any, cfg)
  try:
    for parent in parents:
      walk = parent.follow(walk)
  except Exception as e:
    raise ValueError(f'Invalid path "{path}".') from e

  literal_value = parse_value(value=value, path=path)
  try:
    if isinstance(last, daglish.Attr):
      setattr(walk, last.name, literal_value)
    elif isinstance(last, daglish.Key):
      walk[last.key] = literal_value
    else:
      raise ValueError(f'Unexpected path element {last}.')
  except Exception as e:
    raise ValueError(f'Could not set "{path}" to "{value}".') from e


class ZlibJSONSerializer:
  """Serializer that uses JSON, zlib, and base64 encoding."""

  def serialize(
      self,
      cfg: config.Buildable,
      pyref_policy: Optional[serialization.PyrefPolicy] = None,
  ) -> str:
    return base64.urlsafe_b64encode(
        zlib.compress(serialization.dump_json(cfg, pyref_policy).encode())
    ).decode('ascii')

  def deserialize(
      self,
      serialized: str,
      pyref_policy: Optional[serialization.PyrefPolicy] = None,
  ) -> config.Buildable:
    return serialization.load_json(
        zlib.decompress(base64.urlsafe_b64decode(serialized)).decode(),
        pyref_policy=pyref_policy,
    )
