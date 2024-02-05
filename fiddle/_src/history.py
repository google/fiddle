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
import inspect
import itertools
import os
import threading
from typing import Any, Callable, FrozenSet, Iterator, Optional, Set, Tuple, Union

from fiddle._src import tag_type

# An incrementing counter to allow for time-travel debugging.
_set_counter = itertools.count()


@dataclasses.dataclass(frozen=True)
class Location:
  """Information about where a parameter was set."""
  filename: str
  line_number: int
  function_name: Optional[str]

  def format(self, max_filename_parts: Optional[int] = None):
    filename = self.filename
    if max_filename_parts is not None:
      filename_parts = filename.split(os.path.sep)
      if len(filename_parts) > max_filename_parts:
        filename = os.path.sep.join(
            ["...", *filename_parts[-max_filename_parts:]]
        )
    if self.function_name is None:
      return f"{filename}:{self.line_number}"
    return f"{filename}:{self.line_number}:{self.function_name}"

  def __str__(self) -> str:
    return self.format()

  def __deepcopy__(self, memo):
    del memo  # unused
    return self


# A function that returns a location.
LocationProvider = Callable[[], Location]

_exclude_locations: Tuple[str, ...] = tuple(
    map(
        os.path.normpath,
        [
            "fiddle/_src/config.py",
            "fiddle/_src/copying.py",
            "fiddle/_src/daglish.py",
            "fiddle/_src/history.py",
            "fiddle/_src/materialize.py",
            "fiddle/_src/mutate_buildable.py",
            "fiddle/_src/experimental/auto_config.py",
        ],
    )
)


def add_exclude_location(location: str):
  """Adds a filename pattern to exclude from history stacktraces."""
  global _exclude_locations
  _exclude_locations += (location,)


def _stacktrace_location_provider() -> Location:
  """Returns a string corresponding to the user-function that set the field.

  Raises:
    RuntimeError: if no suitable stack frame can be found.
  """
  frame = inspect.currentframe()
  while frame:
    line_number = frame.f_lineno
    filename = frame.f_code.co_filename
    if not filename.endswith(_exclude_locations):
      function_name = frame.f_code.co_name
      return Location(
          filename=filename,
          line_number=line_number,
          function_name=function_name)
    frame = frame.f_back
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

  def __deepcopy__(self, memo):
    del memo  # unused
    return self


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


@dataclasses.dataclass
class _TrackingState(threading.local):
  enabled: bool = True


_tracking_state = _TrackingState()


@contextlib.contextmanager
def suspend_tracking():
  """A context manager for temporarily suspending history tracking.

  This can be useful in certain cases for performance reasons, or for specific
  operations that want to avoid associated history modifications.

  Example:

      with history.suspend_tracking():
        ...  # Modifications made here won't affect Buildable history.

  Yields:
    There is no associated yield value.
  """
  previous_enabled = tracking_enabled()
  set_tracking(enabled=False)
  try:
    yield
  finally:
    set_tracking(enabled=previous_enabled)


def set_tracking(enabled: bool):
  """Sets whether Fiddle performs history tracking.

  For performance reasons, it can be valuable to disable history tracking. This
  function enables callers to imperatively control whether Fiddle tracks
  mutations to Buildable objects. To disable history tracking::

    history.set_tracking(enabled=False)

  Note: where possible, prefer the context manager ``suspend_tracking`` instead.

  Args:
    enabled: Whether history tracking should be enabled.
  """
  _tracking_state.enabled = enabled


def tracking_enabled() -> bool:
  """Returns whether history tracking is currently enabled."""
  return _tracking_state.enabled


class History(dict):
  """Utility class to manage argument histories."""

  def __missing__(self, key):
    return self.setdefault(key, [])

  def add_new_value(self, param_name, value):
    """Adds a history entry for a new value, created via `new_value`."""
    if tracking_enabled():
      self[param_name].append(new_value(param_name, value))

  def add_deleted_value(self, param_name):
    """Adds a history entry for a deleted value, created via `delete_value`."""
    if tracking_enabled():
      self[param_name].append(deleted_value(param_name))

  def add_updated_tags(self, param_name, updated_tags):
    """Adds a history entry for updated tags, created via `update_tags`."""
    if tracking_enabled():
      self[param_name].append(update_tags(param_name, updated_tags))


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
  try:
    yield temporary_provider
  finally:
    _location_provider = original_location_provider
