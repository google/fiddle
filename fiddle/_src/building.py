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

"""Implements Fiddle's build() function."""

import contextlib
import functools
import logging
import threading
from typing import Any, Callable, Dict, TypeVar, overload

from fiddle._src import config as config_lib
from fiddle._src import daglish
from fiddle._src import reraised_exception

T = TypeVar('T')


class _BuildGuardState(threading.local):

  def __init__(self):
    super().__init__()
    self.in_build = False


_state = _BuildGuardState()


@contextlib.contextmanager
def _in_build():
  """A context manager to ensure fdl.build is not called recursively."""
  if _state.in_build:
    raise ValueError(
        'It is forbidden to call `fdl.build` inside another `fdl.build` call.')
  _state.in_build = True
  try:
    yield
  finally:
    _state.in_build = False


def _format_arg(arg: Any) -> str:
  """Returns repr(arg), returning a constant string if repr() fails."""
  try:
    return repr(arg)
  except Exception:  # pylint: disable=broad-except
    return f'<ERROR FORMATTING {type(arg)} ARGUMENT>'


def _make_message(current_path: daglish.Path, buildable: config_lib.Buildable,
                  arguments: Dict[str, Any]) -> str:
  """Returns Fiddle-related debugging information for an exception."""
  path_str = '<root>' + daglish.path_str(current_path)
  fn_or_cls = config_lib.get_callable(buildable)
  try:
    fn_or_cls_name = fn_or_cls.__qualname__
  except AttributeError:
    fn_or_cls_name = str(fn_or_cls)  # callable instances, etc.
  kwargs_str = ', '.join(
      f'{name}={_format_arg(value)}' for name, value in arguments.items())

  tag_information = ''
  bound_args = buildable.__signature__.bind_partial(**arguments)
  bound_args.apply_defaults()
  unset_arg_tags = []
  for param in buildable.__signature__.parameters:
    if param in bound_args.arguments:
      continue  # User supplied it, all good.
    tags = buildable.__argument_tags__.get(param, None)
    if tags:
      tag_str = ' '.join(sorted(str(tag) for tag in tags))
      unset_arg_tags.append(f' - {param}: {tag_str}')
  if unset_arg_tags:
    tag_details = '\n'.join(unset_arg_tags)
    tag_information = f'\nTags for unset arguments:\n{tag_details}'

  return (
      '\n\nFiddle context: failed to construct or call '
      f'{fn_or_cls_name} at {path_str} '
      f'with arguments ({kwargs_str}){tag_information}'
  )


def call_buildable(
    buildable: config_lib.Buildable,
    arguments: Dict[str, Any],
    *,
    current_path: daglish.Path,
) -> Any:
  make_message = functools.partial(_make_message, current_path, buildable,
                                   arguments)
  with reraised_exception.try_with_lazy_message(make_message):
    return buildable.__build__(**arguments)


# Define typing overload for `build(Partial[T])`
@overload
def build(buildable: config_lib.Partial[T]) -> Callable[..., T]:
  ...


# Define typing overload for `build(Config[T])`
@overload
def build(buildable: config_lib.Config[T]) -> T:
  ...


# Define typing overload for nested structures.
@overload
def build(buildable: Any) -> Any:
  ...


# This is a free function instead of a method on the `Buildable` object in order
# to avoid potential naming collisions (e.g., if a function or class has a
# parameter named `build`).
def build(buildable):
  """Builds ``buildable``, recursively building nested ``Buildable`` instances.

  This is the core function for turning a ``Buildable`` into a usable object. It
  recursively walks through ``buildable``'s parameters, building any nested
  ``Config`` instances. Depending on the specific ``Buildable`` type passed
  (``Config`` or ``Partial``), the result is either the result of calling
  ``config.__fn_or_cls__`` with the configured parameters, or a partial function
  or class with those parameters bound.

  If the same ``Buildable`` instance is seen multiple times during traversal of
  the configuration tree, ``build`` is called only once (for the first instance
  encountered), and the result is reused for subsequent copies of the instance.
  This is achieved via the ``memo`` dictionary (similar to ``deepcopy``). This
  has the effect that for configured class instances, each separate config
  instance is in one-to-one correspondence with an actual instance of the
  configured class after calling ``build`` (shared config instances <=> shared
  class instances).

  Args:
    buildable: A ``Buildable`` instance to build, or a nested structure of
      ``Buildable`` objects.

  Returns:
    The built version of ``buildable``.
  """
  is_built = False

  def _build(value: Any, state: daglish.State) -> Any:
    """Inner method / implementation of build()."""
    nonlocal is_built
    if isinstance(value, config_lib.Buildable):
      sub_traversal = state.flattened_map_children(value)
      metadata: config_lib.BuildableTraverserMetadata = sub_traversal.metadata
      arguments = metadata.arguments(sub_traversal.values)
      is_built = True
      return call_buildable(value, arguments, current_path=state.current_path)
    else:
      return state.map_children(value)

  with _in_build():
    result = daglish.MemoizedTraversal.run(_build, buildable)

  if not is_built:
    logging.warning(
        'No Buildables found in value passed to `fdl.build()`: '
        '%s with type %s.',
        str(buildable),
        type(buildable),
    )

  return result
