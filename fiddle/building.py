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
import threading

from typing import Any, Dict, Tuple

from fiddle import config
from fiddle import tagging
from fiddle.experimental import daglish


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


class BuildError(ValueError):
  """Error raised when building a Config fails."""

  def __init__(
      self,
      buildable: config.Buildable,
      path_from_config_root: str,
      original_error: Exception,
      args: Tuple[Any, ...],
      kwargs: Dict[str, Any],
  ) -> None:
    super().__init__(str(original_error))
    self.buildable = buildable
    self.path_from_config_root = path_from_config_root
    self.original_error = original_error
    self.args = args
    self.kwargs = kwargs

  def __str__(self):
    fn_or_cls_name = self.buildable.__fn_or_cls__.__qualname__
    return (f'Failed to construct or call {fn_or_cls_name} '
            f'(at {self.path_from_config_root}) with arguments\n'
            f'    args: {self.args}\n'
            f'    kwargs: {self.kwargs}')


# This is a free function instead of a method on the `Buildable` object in order
# to avoid potential naming collisions (e.g., if a function or class has a
# parameter named `build`).
def build(buildable):
  """Builds `buildable`, recursively building any nested `Buildable` instances.

  This is the core function for turning a `Buildable` into a usable object. It
  recursively walks through `buildable`'s parameters, building any nested
  `Config` instances. Depending on the specific `Buildable` type passed
  (`Config` or `Partial`), the result is either the result of calling
  `config.__fn_or_cls__` with the configured parameters, or a partial function
  or class with those parameters bound.

  If the same `Buildable` instance is seen multiple times during traversal of
  the configuration tree, `build` is called only once (for the first instance
  encountered), and the result is reused for subsequent copies of the instance.
  This is achieved via the `memo` dictionary (similar to `deepcopy`). This has
  the effect that for configured class instances, each separate config instance
  is in one-to-one correspondence with an actual instance of the configured
  class after calling `build` (shared config instances <=> shared class
  instances).

  Args:
    buildable: A `Buildable` instance to build, or a nested structure of
      `Buildable` objects.

  Returns:
    The built version of `buildable`.
  """
  memo = {}

  # The implementation here performs explicit recursion instead of using
  # `daglish.memoized_traverse` in order to avoid unnecessary unflattening of
  # `Buildable`s (which can cause errors for certain `Buildable` subclasses).
  def traverse(value):
    if id(value) not in memo:
      if isinstance(value, config.Buildable):
        arguments = daglish.map_children(traverse, value.__arguments__)
        try:
          memo[id(value)] = value.__build__(**arguments)
        except tagging.TaggedValueNotFilledError:
          raise
        except Exception as e:
          paths = daglish.collect_paths_by_id(buildable, memoizable_only=True)
          path_str = '<root>' + daglish.path_str(paths[id(value)][0])
          raise BuildError(value, path_str, e, (), arguments) from e
      elif daglish.is_traversable_type(type(value)):
        memo[id(value)] = daglish.map_children(traverse, value)
      else:
        memo[id(value)] = value

    return memo[id(value)]

  with _in_build():
    return traverse(buildable)
