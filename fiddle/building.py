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

from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, overload

from fiddle import config
from fiddle import daglish
from fiddle import tag_type

T = TypeVar('T')
CallableProducingT = Callable[..., T]


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
  buildable: config.Buildable
  path_from_config_root: str
  original_error: Exception
  args: Tuple[Any, ...]
  kwargs: Dict[str, Any]

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

  def __str__(self) -> str:
    fn_or_cls_name = self.buildable.__fn_or_cls__.__qualname__
    return (f'Failed to construct or call {fn_or_cls_name} '
            f'(at {self.path_from_config_root}) with arguments\n'
            f'    args: {self.args}\n'
            f'    kwargs: {self.kwargs}')


# Define typing overload for `build(Partial[T])`
@overload
def build(buildable: config.Partial[T]) -> CallableProducingT:
  ...


# Define typing overload for `build(Partial)`
@overload
def build(buildable: config.Partial) -> Callable[..., Any]:
  ...


# Define typing overload for `build(Config[T])`
@overload
def build(buildable: config.Config[T]) -> T:
  ...


# Define typing overload for `build(Config)`
@overload
def build(buildable: config.Config) -> Any:
  ...


# Define typing overload for `build(Buildable)`
@overload
def build(buildable: config.Buildable) -> Any:
  ...


# Define typing overload for nested structures.
@overload
def build(buildable: Any) -> Any:
  ...


def _build(value: Any, state: Optional[daglish.State] = None) -> Any:
  """Inner method / implementation of build()."""
  state = state or daglish.MemoizedTraversal.begin(_build, value)

  if isinstance(value, config.Buildable):
    sub_traversal = state.flattened_map_children(value)
    metadata: config.BuildableTraverserMetadata = sub_traversal.metadata
    arguments = metadata.arguments(sub_traversal.values)
    try:
      return value.__build__(**arguments)
    except tag_type.TaggedValueNotFilledError:
      raise
    except Exception as e:
      path_str = '<root>' + daglish.path_str(state.current_path)
      raise BuildError(value, path_str, e, (), arguments) from e
  else:
    return state.map_children(value)


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

  with _in_build():
    return _build(buildable)
