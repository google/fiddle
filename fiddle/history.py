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

"""History tracking for config objects.

In order to make it easier to understand how config objects have been
constructed, config objects keep track of the sequence of changes made to them.
This file contains the machinery to record this history and easily manipulate
it.

The history functionality has a few bits of non-trivial logic:

 1. Each history entry is associated with a monotonically increasing number.
    This allows sequencing history across multiple config objects. The
    implementation of this capability is guaranteed to be threadsafe, so users
    can mutate (different!) config objects from multiple threads without fear.
 2. The logic to determine where in the user's source code a mutation was
    generated is configurable. Simply use `custom_location` to temporarily set a
    different location provider; see the documentation on the `custom_location`
    context manager for an example.
"""

import contextlib
import dataclasses
import enum
import itertools
import traceback
from typing import Any, Callable, FrozenSet, Iterator, Optional, Set, Union

from fiddle import tag_type

# An incrementing counter to allow for time-travel debugging.
_set_counter = itertools.count()


@dataclasses.dataclass(frozen=True)
class Location:
  """Information about where a parameter was set."""
  filename: str
  line_number: int
  function_name: Optional[str]

  def __str__(self) -> str:
    if self.function_name is None:
      return f"{self.filename}:{self.line_number}"
    return f"{self.filename}:{self.line_number}:{self.function_name}"


# A function that returns a location.
LocationProvider = Callable[[], Location]


def _stacktrace_location_provider() -> Location:
  """Returns a string corresponding to the user-function that set the field.

  Raises:
    RuntimeError: if no suitable stack frame can be found.
  """
  for frame in reversed(traceback.extract_stack()):
    if (not frame.filename.endswith("fiddle/config.py") and
        not frame.filename.endswith("fiddle/daglish.py") and
        not frame.filename.endswith("fiddle/history.py") and
        not frame.filename.endswith("fiddle/materialize.py")):
      return Location(
          filename=frame.filename,
          line_number=frame.lineno,
          function_name=frame.name)
  raise RuntimeError("Cannot find a suitable frame in the stack trace!")


# The location provider to use when instantiating new HistoryEntry's.
_location_provider: LocationProvider = _stacktrace_location_provider


class _Deleted:
  """A marker object to indicated deletion."""

  def __repr__(self):
    return "DELETED"


# A marker object to record when a field was deleted.
DELETED = _Deleted()


class ChangeKind(enum.Enum):
  """Indicates the kind of change that occurred to the parameter.

  NEW_VALUE indicates the value of the parameter changed.
  UPDATE_TAGS indicates the tag set was updated.
  """
  NEW_VALUE = 1
  UPDATE_TAGS = 2


@dataclasses.dataclass(frozen=True)
class HistoryEntry:
  """An entry in the history table for a config object.

  Attributes:
    sequence_id: The global sequence number, to allow ordering of history
      sequences across config objects.
    param_name: The parameter name for the history entry.
    kind: The kind of change that occurred, which influences the interpretation
      of `value`.
    new_value: The new value of the field, the updated set of tags, or DELETED
      if the field was `del`d.
    location: The location in user code that made the modification.
  """
  sequence_id: int
  param_name: str
  kind: ChangeKind
  new_value: Union[Any, FrozenSet[tag_type.TagType], _Deleted]
  location: Location


def new_value(param_name: str, value: Any) -> HistoryEntry:
  """Returns a newly constructed history entry.

  Args:
    param_name: The name of the config parameter the entry is associated with.
    value: The new value, or DELETED to indicate a deletion of the param.

  Returns:
    The newly constructed HistoryEntry.
  """
  return HistoryEntry(
      sequence_id=next(_set_counter),
      param_name=param_name,
      kind=ChangeKind.NEW_VALUE,
      new_value=value,
      location=_location_provider())


def deleted_value(param_name: str) -> HistoryEntry:
  """Returns a newly constructed history entry.

  Args:
    param_name: The name of the config parameter the entry is associated with.

  Returns:
    The newly constructed HistoryEntry.
  """
  return HistoryEntry(
      sequence_id=next(_set_counter),
      param_name=param_name,
      kind=ChangeKind.NEW_VALUE,
      new_value=DELETED,
      location=_location_provider())


def update_tags(param_name: str,
                updated_tags: Set[tag_type.TagType]) -> HistoryEntry:
  """Returns a newly constructed history entry.

  Args:
    param_name: The name of the config parameter the entry is associated with.
    updated_tags: The new set of tags associated with `param_name`.

  Returns:
    The newly constructed HistoryEntry.
  """
  return HistoryEntry(
      sequence_id=next(_set_counter),
      param_name=param_name,
      kind=ChangeKind.UPDATE_TAGS,
      new_value=frozenset(updated_tags),
      location=_location_provider())


@contextlib.contextmanager
def custom_location(
    temporary_provider: LocationProvider) -> Iterator[LocationProvider]:
  """Temporarily sets a custom LocationProvider.

  Example usage:
  ```py
  my_config =  # ...
  with custom_location(lambda: Location('my_loc', 123, None)):
    my_config.x = 123  # the location of my_config.x will be 'my_loc:123'.
  ```

  Args:
    temporary_provider: A location provider to use for the duration of the with
      block.

  Yields:
    The temporary provider.
  """
  global _location_provider
  original_location_provider = _location_provider
  _location_provider = temporary_provider
  yield temporary_provider
  _location_provider = original_location_provider
