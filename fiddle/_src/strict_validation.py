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

"""APIs to ensure configs are constructed with valid children."""

import contextlib
import threading
from typing import Any, Collection, Type

from fiddle._src import daglish


_validation_enabled = threading.local()
_validation_enabled.v = False


@contextlib.contextmanager
def strict_validation():
  try:
    _validation_enabled.v = True
    yield
  finally:
    _validation_enabled.v = False


def is_valid(
    structure,
    additional_allowed_types: Collection[Type[Any]],
    stop_at_allowed_type: bool = True,
) -> bool:
  """Returns True if a structure is valid or validation is disabled.

  Args:
    structure: Arbitrary nested value.
    additional_allowed_types: Set of allowed types (will include subtypes).
    stop_at_allowed_type: Do not traverse further into allowed types, assuming
      they have been validated.
  """
  if not _validation_enabled.v:
    return True

  def traverse(value, state: daglish.State):
    if daglish.is_namedtuple_subclass(type(value)):
      return False
    elif daglish.is_internable(value):
      return True
    elif any(isinstance(value, typ) for typ in additional_allowed_types):
      if stop_at_allowed_type or not state.is_traversable(value):
        return True
    elif not isinstance(value, (list, tuple, dict)):
      return False
    return all(state.flattened_map_children(value).values)

  return daglish.MemoizedTraversal.run(traverse, structure)
