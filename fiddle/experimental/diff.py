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

"""Library for finding differences between Fiddle configurations."""

from typing import Any, Dict, Sequence, List, Tuple
from fiddle.experimental import daglish


class DiffAlignment:
  """Alignment between two structures, named `old` and `new`.

  `DiffAlignment` is a partial bidirectional mapping between nested objects in
  two structures (`old` and `new`).  When building diffs, this alignment is
  used to decide which objects in `old` should be mutated to become objects in
  `new`.

  This class places the following restrictions on when objects in `old` and
  `new` may be aligned:

  * Each `old_value` may be aligned with at most one `new_value`
    (and vice versa).
  * `type(new_value)` must be equal to `type(old_value)`.
  * If `isinstance(old_value, Sequence)`, then `len(new_value)` must be equal
    to `len(old_value)`.
  * If `old_value` and `new_value` are aligned, then it must be possible to
    mutate `old_value` to become `new_value`.
  * Alignments may not create cycles.  E.g., `old_value` may not be aligned
    with `new_value` if some value contained in `old_value` is aligned with a
    value that contains `new_value`.
  * Aligned values must be "memoizable" (as defined by `daglish.is_memoizable`).

  These restrictions help ensure that a `Diff` can be built from the alignment.
  Note: `DiffAlignment` is not guaranteed to catch all violations of these
  restrictions, since some are difficult or expensive to detect.
  """

  def __init__(self,
               old: Any,
               new: Any,
               old_name: str = 'old',
               new_name: str = 'new'):
    """Creates an empty alignment between objects in `old` and `new`.

    In the new alignment, no objects in `old` are aligned with any objects
    in `new` (including the root objects `old` and `new` themselves).  Call
    `DiffAlignment.align` to add alignments.

    Args:
      old: The root object of the `old` structure.
      new: The root object of the `new` structure.
      old_name: A name for the `old` structure (used to print alignments).
      new_name: A name for the `new` structure (used to print alignments).
    """
    self._old: Any = old
    self._new: Any = new
    self._old_name: str = old_name
    self._new_name: str = new_name
    self._old_to_new: Dict[int, Any] = {}  # id(old_value) -> new_value
    self._new_to_old: Dict[int, Any] = {}  # id(new_value) -> old_value

  @property
  def old(self) -> Any:
    """The root object of the `old` structure for this alignment."""
    return self._old

  @property
  def new(self) -> Any:
    """The root object of the `new` structure for this alignment."""
    return self._new

  @property
  def old_name(self) -> str:
    """The name of the `old` structure for this alignment."""
    return self._old_name

  @property
  def new_name(self) -> str:
    """The name of the `new` structure for this alignment."""
    return self._new_name

  def is_old_value_aligned(self, old_value):
    """Returns true if `old_value` is aligned with any value."""
    return id(old_value) in self._old_to_new

  def is_new_value_aligned(self, new_value):
    """Returns true if `new_value` is aligned with any value."""
    return id(new_value) in self._new_to_old

  def old_to_new(self, old_value):
    """Returns the object in `new` that is aligned with `old_value`."""
    return self._old_to_new[id(old_value)]

  def new_to_old(self, new_value):
    """Returns the object in `old` that is aligned with `new_value`."""
    return self._new_to_old[id(new_value)]

  def aligned_values(self) -> List[Tuple[Any, Any]]:
    """Returns a list of `(old_value, new_value)` for all aligned values."""
    return [(self._new_to_old[id(new_value)], new_value)
            for (old_id, new_value) in self._old_to_new.items()]

  def aligned_value_ids(self) -> List[Tuple[int, int]]:
    """Returns a list of `(id(old_val), id(new_val))` for all aligned values."""
    return [(old_id, id(new_value))
            for (old_id, new_value) in self._old_to_new.items()]

  def align(self, old_value: Any, new_value: Any):
    """Aligns `old_value` with `new_value`.

    Assumes that `old_value` is contained in `self.old`, and that `new_value`
    is contained in `self.new`.

    Args:
      old_value: The value in `old` that should be aligned with `new_value.`
      new_value: The value in `new` that should be aligned with `old_value.`

    Raises:
      ValueError: If `old_value` and `new_value` can not be aligned.  For
        example, this can happen if either value is already aligned, or if
        the values are incompatible.  See the docstring for `DiffAlignment`
        for a full list of restrictions.  Note: the `align` method is not
        guaranteed to catch all violations of these restrictions, since some
        are difficult or expensive to detect.
    """
    self._validate_alignment(old_value, new_value)
    self._old_to_new[id(old_value)] = new_value
    self._new_to_old[id(new_value)] = old_value

  def _validate_alignment(self, old_value, new_value):
    """Raises ValueError if `old_value` can not be aligned with `new_value`."""
    if not daglish.is_memoizable(old_value):
      raise ValueError(f'old_value={old_value!r} may not be aligned because '
                       'it is not memoizable.')
    if not daglish.is_memoizable(new_value):
      raise ValueError(f'new_value={new_value!r} may not be aligned because '
                       'it is not memoizable.')
    if self.is_old_value_aligned(old_value):
      raise ValueError('An alignment has already been added for ' +
                       f'old value {old_value!r}')
    if self.is_new_value_aligned(new_value):
      raise ValueError('An alignment has already been added for ' +
                       f'new value {new_value!r}')
    if type(old_value) is not type(new_value):
      raise ValueError(f'Aligning objects of different types is not currently '
                       f'supported.  ({type(old_value)} vs {type(new_value)})')
    if isinstance(old_value, Sequence):
      if len(old_value) != len(new_value):
        raise ValueError(
            f'Aligning sequences with different lengths is not '
            f'currently supported.  ({len(old_value)} vs {len(new_value)})')

  def __repr__(self):
    return (f'<DiffAlignment from {self._old_name!r} to ' +
            f'{self._new_name!r}: {len(self._old_to_new)} object(s) aligned>')

  def __str__(self):
    id_to_old_path = daglish.collect_paths_by_id(self.old)
    id_to_new_path = daglish.collect_paths_by_id(self.new)
    old_to_new_paths = [(id_to_old_path[old_id][0], id_to_new_path[new_id][0])
                        for (old_id, new_id) in self.aligned_value_ids()]
    lines = [
        f'    {self.old_name}{daglish.path_str(old_path)}'
        f' -> {self.new_name}{daglish.path_str(new_path)}'
        for (old_path, new_path) in old_to_new_paths
    ]
    if not lines:
      lines.append('    (no objects aligned)')
    return 'DiffAlignment:\n' + '\n'.join(lines)
