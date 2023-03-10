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

"""Manages a namespace to avoid conflicting names.

Generally, when generating code, we do not want variables names to conflict.
This class helps manage names in scope, and for simplicity we try to avoid
conflicts between globals/non-locals/locals, even if they might work when run.
"""

import dataclasses
import itertools
import keyword
import re
from typing import Optional, Set

_CAMEL_TO_SNAKE_RE = re.compile(r"(?<=.)([A-Z])")
_PY_VAR_NAME_INVALID_RE = re.compile(r"[^a-z_0-9]+")
_PY_VAR_NAME_RE = re.compile(r"[a-z][a-z_0-9]+")


def camel_to_snake(name: str) -> str:
  """Converts a camel or studly-caps name to a snake_case name."""
  return _CAMEL_TO_SNAKE_RE.sub(lambda m: "_" + m.group(0), name).lower()


def valid_name_chars(name: str) -> str:
  """Removes invalid characters such as whitespace, parenthesis, etc."""
  return _PY_VAR_NAME_INVALID_RE.sub("", name)


def py_var_name(name: str) -> Optional[str]:
  """Returns a sanitized, lower-case version of `name` for py variables."""
  name = valid_name_chars(camel_to_snake(name))
  match = _PY_VAR_NAME_RE.search(name)
  return match.group(0) if match else None


@dataclasses.dataclass
class Namespace:
  """Manages active Python instance names.

  By default, the namespace will be populated with Python keywords. If you do
  not want this, then initialize `names` manually to the empty set.
  """

  names: Set[str] = dataclasses.field(
      default_factory=lambda: set(keyword.kwlist))

  def __contains__(self, key: str) -> bool:
    """Returns True if a name is already defined.

    In the default Namespace() constructor, names include keywords.

    Args:
      key: Name to check.
    """
    return key in self.names

  def add(self, name: str) -> str:
    """Adds a name and returns it, raising an error if it already exists."""
    if name in self.names:
      raise ValueError(
          f"Tried to add {name!r} (e.g. an import), but it already exists!")
    self.names.add(name)
    return name

  def get_new_name(self, base_name, prefix: str) -> str:
    """Generates a new name for a given instance.

    This method adds the new name to the namespace.

    Example:
      self.names = ["foo"]
      get_new_name("bar", "shared_") == "shared_bar"

    Example 2:
      self.names = ["shared_foo", "bar"]
      get_new_name("Foo", "shared_") == "shared_foo_2"

    Args:
      base_name: Base name to derive a new name for. This will be converted to
        snake case automatically.
      prefix: Prefix to prepend to any names.

    Returns:
      A new unique name.
    """

    name = prefix + camel_to_snake(base_name)
    if name not in self.names:
      return self.add(name)
    for i in itertools.count(start=2):
      if f"{name}_{i}" not in self.names:
        return self.add(f"{name}_{i}")
    raise AssertionError("pytype helper -- itertools.count() is infinite")
