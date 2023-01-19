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

"""Internal implementation of autofill functionality.

For additional context, please see the module comment in the API
file.
"""

import inspect
from typing import Any, Callable, Dict, Union, Type
import typing_extensions


Autofill = object()


# Set to `True` to see type annotation errors.
# TODO(b/265956870): Make this more user discoverable.
_DEBUG_TYPE_ANNOTATIONS = False


def parameters_to_autofill(
    obj: Any,
    signature: inspect.Signature,
) -> Dict[str, Union[Type[Any], Callable[[], Any]]]:
  """Inspects `obj` to determine all parameters annotated with autofill."""
  autofill_params = {}

  if isinstance(obj, type):
    # Use `__init__`'s annotations.
    try:
      annotations = typing_extensions.get_type_hints(
          obj.__init__, include_extras=True
      )
    except (TypeError, NameError):
      # Can occur when the type annotations don't actually refer to types, or
      # there's an unknown name.
      if _DEBUG_TYPE_ANNOTATIONS:
        raise
      return {}
  else:
    try:
      annotations = typing_extensions.get_type_hints(obj, include_extras=True)
    except (TypeError, NameError):
      # Happens when `obj` is not a module, class, method, or function, or when
      # the types (e.g. Any) isn't appropriate imported.
      if _DEBUG_TYPE_ANNOTATIONS:
        raise
      return {}
  for param in signature.parameters:
    annotation = annotations.get(param)
    if (
        typing_extensions.get_origin(annotation)
        is not typing_extensions.Annotated
    ):
      continue
    type_args = typing_extensions.get_args(annotation)
    if any(arg is Autofill for arg in type_args):
      autofill_params[param] = type_args[0]
  return autofill_params
