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

"""Module for getting (and caching) inspect.Signature objects."""

import inspect
import weakref

_signature_cache = weakref.WeakKeyDictionary()


def _get_uncached(fn_or_cls) -> inspect.Signature:
  try:
    return inspect.signature(fn_or_cls)
  except ValueError:
    if isinstance(fn_or_cls, type) and hasattr(fn_or_cls, '__call__'):
      try:
        return inspect.signature(fn_or_cls.__call__)
      except ValueError:
        pass
    raise


def get_signature(fn_or_cls) -> inspect.Signature:
  """Returns (and maybe caches) the signature for `fn_or_cls`.

  If a signature is not available for `fn_or_cls` and `fn_or_cls` is a type,
  this function will instead try to get a signature for `fn_or_cls.__call__`.
  This behavior allows a reasonable signature to be inferred for certain builtin
  types such as `dict`.

  Args:
    fn_or_cls: The function or class to return a signature for.
  """
  try:
    return _signature_cache[fn_or_cls]
  except TypeError:  # Unhashable value...
    return _get_uncached(fn_or_cls)
  except KeyError:
    signature = _get_uncached(fn_or_cls)
    _signature_cache[fn_or_cls] = signature
    return signature


def has_signature(fn_or_cls) -> bool:
  """Returns whether `fn_or_cls` has an available signature.

  Note that this will return `True` even if a signature for `fn_or_cls` is not
  yet in the cache. The signature for `fn_or_cls` will be cached if it is
  avialable.

  Args:
    fn_or_cls: The function or class to check signature availability for.
  """
  try:
    get_signature(fn_or_cls)
  except (ValueError, TypeError):
    return False
  return True
