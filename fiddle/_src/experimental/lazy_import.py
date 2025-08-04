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

"""Wrapper for a callable that doesn't get imported until it's called."""

import dataclasses
import importlib
import inspect
from typing import Any, Callable, ClassVar, Optional

from fiddle._src import signatures


def lazy_import(module: str, qualname: Optional[str] = None):
  """Decorator used to lazily import a callable (i.e., a function or class).

  This decorator should be applied to a stub function or class which
  provides the signature of the callable.  E.g.:

  >>> # pylint: disable=unused-argument
  >>> @lazy_import("my_module", "my_function")
  ... def my_function(x: list[float], n: int) -> list[float]:
  ...   ...
  >>> @lazy_import("my_module", "MyClass")
  ... class MyClass:
  ...   def __init__(self, x: int, y: Optional[float] = None):
  ...     ...
  >>> # pylint: enable=unused-argument

  When a lazily imported function or class is called, it will first import the
  function or class (if it hasn't been imported aleady) and then call it and
  return the result.

  This can be used to define a `fdl.Config` for a callable without importing
  the callable.  The callable will then be imported if and when the `fdl.Config`
  is built.

  Note: pylint and pytype can generate spurious warnings for stub declarations.
  In particular, pytype warns about unused arguments and pytype complains
  return type annotation mismatches.  Until we identify a better solution, we
  currently recommend that you wrap your stub declarations with
  `pylint: disable=unused-argument` and `pytype: disable=bad-return-type`.

  Args:
    module: The dotted name of the module containing the function or class.
    qualname: The dotted name used to access the function or class within
      `module`.  If not specified, defaults to the decorated object's
      `__qualname__`.

  Returns:
    A decorator.

  Raises:
    ValueError: If the decorated value isn't a stub function (i.e., a function
      whose body is ellipsis or a docstring) or a stub class (i.e., a class
      that contains only a stub function named `__init__` and an optional
      docstring).
  """

  def decorator(fn_or_cls):
    _sanity_check_stub(fn_or_cls)
    signature = signatures.get_signature(fn_or_cls)
    name = fn_or_cls.__qualname__ if qualname is None else qualname
    lazy_fn = LazyCallable(module, name, signature)
    if isinstance(fn_or_cls, type):
      bases = (LazyClass,)
      namespace = {
          "lazy_fn": lazy_fn,
          "__module__": fn_or_cls.__module__,
          "__qualname__": fn_or_cls.__qualname__,
          "__name__": fn_or_cls.__name__,
          "__doc__": fn_or_cls.__doc__,
          "__signature__": signature,
      }
      return type(fn_or_cls.__name__, bases, namespace)
    else:
      return lazy_fn

  return decorator


# Trivial class used by _sanity_check_stub to determine what keys a classes'
# __dict__ will have if it only defines __init__ (this could potentially
# change with different Python versions.)
class _SimpleClassWithConstructor:

  def __init__(self):
    pass


# Trivial function used by _sanity_check_stub to determine the `co_code` for
# a function whose body is just a single ellipsis.
def _function_with_ellipsis_body():
  ...


# Trivial function used by _sanity_check_stub to determine the `co_code` for
# a function whose body is just a docstring
def _function_with_docstring_body():
  """Sample docstring."""


def _sanity_check_stub(fn_or_cls):
  """Checks that `fn_or_cls` is a valid stub function or class.

  Args:
    fn_or_cls: The callable to validate.

  Raises:
    ValueError: If `fn_or_cls` is not a valid stub.
  """

  if isinstance(fn_or_cls, type):
    if set(fn_or_cls.__dict__) != set(_SimpleClassWithConstructor.__dict__):
      raise ValueError(
          "Classes decorated with `lazy_import` should define "
          "a stub method __init__ (and nothing else)."
      )
    _sanity_check_stub(fn_or_cls.__init__)

  elif hasattr(fn_or_cls, "__code__"):
    if fn_or_cls.__code__.co_code not in (
        _function_with_ellipsis_body.__code__.co_code,
        _function_with_docstring_body.__code__.co_code,
    ):
      raise ValueError(
          "The body for functions decorated with `lazy_import` "
          "should be an ellipsis mark or a docstring."
      )


@dataclasses.dataclass
class LazyCallable:
  """Callable that is lazily loaded the first time it is called."""

  module: str
  qualname: str
  signature: inspect.Signature
  fn_or_cls: Optional[Callable[..., Any]] = dataclasses.field(
      default=None, init=False
  )

  def __post_init__(self):
    if not _is_dotted_name(self.module):
      raise ValueError(
          f"Expected `module` to be a dotted name, got {self.module!r}."
      )
    if not _is_dotted_name(self.qualname):
      raise ValueError(
          f"Expected `qualname` to be a dotted name, got {self.qualname!r}."
      )
    self.__module__ = self.module
    self.__qualname__ = self.qualname
    self.__signature__ = self.signature

  def __call__(self, *args, **kwargs):
    if self.fn_or_cls is None:
      self._load_fn_or_cls()
    return self.fn_or_cls(*args, **kwargs)

  def _load_fn_or_cls(self):
    """Imports and returns `self.qualname` from `self.module`."""
    value = importlib.import_module(self.module)
    for name_piece in self.qualname.split("."):
      try:
        value = getattr(value, name_piece)
      except AttributeError:
        raise ImportError(
            f"cannot import {self.qualname} from {self.module}"
        ) from None

    if not callable(value):
      raise TypeError(
          f"Expected {self.module}:{self.qualname} to be callable; "
          f"got {value!r}"
      )

    self._validate_signature(value)
    self.fn_or_cls = value

  def _validate_signature(self, value: Callable[..., Any]):
    """Raise ValueError if self.signature doesn't match value's signature."""
    signature = inspect.signature(value)
    actual_params = signature.parameters
    expect_params = self.__signature__.parameters

    # Check parameter names and kinds; but don't check annotations or
    # default values.
    if set(actual_params) != set(expect_params) or any(
        actual_params[name].kind != expect_params[name].kind
        for name in actual_params
    ):
      raise ValueError(
          f"Expected {self.__module__}:{self.__qualname__} to "
          f"have signature {self.__signature__}; got {value} "
          f"with signature {inspect.signature(value)}."
      )


class LazyClass:
  """Base class for lazily imported classes."""

  lazy_fn: ClassVar[LazyCallable]

  def __new__(cls, *args, **kwargs):
    return cls.lazy_fn(*args, **kwargs)


def _is_dotted_name(name):
  """Returns true if `name` is a valid Python dotted name."""
  return all(piece.isidentifier() for piece in name.split("."))


def _replace_default_with_ellipsis(
    param: inspect.Parameter,
) -> inspect.Parameter:
  if param.default is param.empty:
    return param
  else:
    return param.replace(default=Ellipsis)


# TODO(b/277097342): Change type annotations we can't resolve to Any.
# TODO(b/277097342): Add support for nested classes.
def codegen_lazy_import_module(module) -> str:
  """Generate fiddle lazy_import wrappers for a given module.

  Args:
    module: The source module:

  Returns:
    Python code that defines `lazy_import` wrappers for all callable symbols
    in `module`.
  """
  module_name = module.__name__

  lines = []
  write_line = lines.append
  write_line(f"# Fiddle lazy_import wrappers for {module_name}")
  write_line("# THIS FILE WAS AUTOMATICALLY GENERATED.")
  write_line("")
  write_line("from fiddle.experimental import lazy_import as _lazy_import")
  write_line("")
  write_line("# pylint: disable=unused-argument")
  write_line("# pytype: disable=bad-return-type")

  for name, value in module.__dict__.items():
    if callable(value):
      write_line("")
      write_line("")
      qualname = value.__qualname__
      if not _is_dotted_name(qualname):
        write_line(f"# Skipped {qualname}.")
        continue

      signature = inspect.signature(value)

      # Replace default values with ellipsis.
      signature = signature.replace(
          parameters=[
              _replace_default_with_ellipsis(param)
              for param in signature.parameters.values()
          ]
      )

      write_line(f"@_lazy_import.lazy_import({module_name}, {qualname})")
      if isinstance(value, type):
        write_line(f"class {name}:")
        write_line(f"  def __init__{signature}:")
        write_line("    ...")
      else:
        write_line(f"def {name}{signature}:")
        write_line("  ...")

  write_line("")
  return "\n".join(lines)
