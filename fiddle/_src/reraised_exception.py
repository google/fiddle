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

"""Wraps exceptions with additional Fiddle-specific information.

Python provides an official solution, "raise FiddleException(...) from e", to do
this. Unfortunately, the stack traces highlight Fiddle internals in this case,
when often a user error (from `e`'s original stack trace) is more relevant.

Hence, we reraise a proxied version of the original exception that adds
additional Fiddle-specific context to the exception message. This approach was
inspired by Gin's exception logic.
"""

import contextlib
import functools
import logging
from typing import Callable

from absl import logging


@functools.lru_cache(maxsize=None)
def make_exception_class(exception_type):
  """Subclasses `exception_type` with one that appends a message."""

  class ExceptionProxy(exception_type):
    """Acts as a proxy for an exception with an augmented message."""
    __module__ = exception_type.__module__

    def __init__(self, proxy_base_exception, proxy_message):
      self.proxy_base_exception = proxy_base_exception
      self.proxy_message = proxy_message

    def __getattr__(self, attr_name):
      return getattr(self.proxy_base_exception, attr_name)

    def __reduce__(self):
      return decorate_exception, (self.proxy_base_exception, self.proxy_message)

    def __str__(self):
      return str(self.proxy_base_exception) + self.proxy_message

  ExceptionProxy.__name__ = exception_type.__name__
  ExceptionProxy.__qualname__ = exception_type.__qualname__
  return ExceptionProxy


def decorate_exception(exception, message: str):
  try:
    proxy_cls = make_exception_class(type(exception))
    return proxy_cls(exception, message).with_traceback(exception.__traceback__)
  except Exception:  # pylint: disable=broad-except
    logging.exception('Creating the proxy class failed.')
    return exception


@contextlib.contextmanager
def try_with_lazy_message(lazy_message: Callable[[], str]):
  """Context manager which reraises exceptions."""
  try:
    yield
  except Exception as exc:  # pylint: disable=broad-except
    try:
      message = lazy_message()
    except:  # pylint: disable=broad-except
      logging.exception('Formatting the debug information failed.')
      raise exc from None
    else:
      raise decorate_exception(exc, message) from None
