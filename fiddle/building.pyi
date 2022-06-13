"""Type stub file for building.py.

This is needed to work around an issue where pytype doesn't respect the ordering
of @overload annotations in .py files (b/158386228).

TODO: Remove this file once b/158386228 is resolved.
"""
from typing import Any, Callable, Dict, TypeVar, Tuple, Union, overload

from fiddle import config

T = TypeVar('T')
CallableProducingT = Callable[..., T]


class BuildError(ValueError):
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
    ...

  def __str__(self) -> str:
    ...


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
