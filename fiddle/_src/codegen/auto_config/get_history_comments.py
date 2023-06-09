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

"""Pass that adds history comments to variables and outputs."""

import os.path
from typing import List

from fiddle import history
from fiddle._src import config as config_lib
from fiddle._src.codegen.auto_config import code_ir


def _format_location(
    location: history.Location, shorten_filenames: bool = True
) -> str:
  filename_parts = location.filename.split(os.path.sep)
  if shorten_filenames and len(filename_parts) > 2:
    filename = os.path.sep.join(["...", *filename_parts[-2:]])
  else:
    filename = location.filename
  return f"{filename}:{location.line_number}:{location.function_name}"


def _prefix_for_entry(entry: history.HistoryEntry, capitalize: bool) -> str:
  set_or_deleted = (
      "Set in " if entry.new_value is not history.DELETED else "Deleted in "
  )
  if not capitalize:
    set_or_deleted = set_or_deleted.lower()
  return set_or_deleted


def _concise_history(
    entries: List[history.HistoryEntry],
    shorten_filenames: bool = True,
    num_history_entries: int = 1,
) -> str:
  """Returns a concise location string for a list of history entries."""
  if not entries:
    return "(no history)"
  result_parts = []
  for i, entry in enumerate(entries[: -(num_history_entries + 1) : -1]):
    result_parts.append(
        _prefix_for_entry(entry, capitalize=i == 0)
        + _format_location(entry.location, shorten_filenames=shorten_filenames)
    )
  return ", ".join(result_parts)


def format_history_for_buildable(
    value: config_lib.Buildable,
    shorten_filenames: bool = True,
    num_history_entries: int = 1,
) -> code_ir.HistoryComments:
  per_field = {}
  for key, entries in value.__argument_history__.items():
    per_field[key] = _concise_history(
        entries, shorten_filenames, num_history_entries=num_history_entries
    )
  return code_ir.HistoryComments(per_field)
