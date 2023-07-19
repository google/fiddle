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

"""API and default implementation for naming objects."""

import abc
import copy
import dataclasses
from typing import Any, List, Optional, Set

from fiddle import daglish
from fiddle._src import config as config_lib
from fiddle._src.codegen import namespace as namespace_lib
from fiddle._src.codegen.auto_config import code_ir


_camel_to_snake = namespace_lib.camel_to_snake  # pylint: disable=protected-access


class NameGenerationError(ValueError):
  """Error thrown when a namer cannot deduce a name for a value."""


@dataclasses.dataclass
class Namer(metaclass=abc.ABCMeta):
  """Assigns names to objects in a config."""

  namespace: namespace_lib.Namespace

  def name_from_candidates(self, candidates: List[str]) -> str:

    # Go through all candidates, and see if one can be used without having to
    # append "_2" kinds of suffixes to it. If so, add and return that one.
    # TODO(b/269464743): Make this more customizable. Sometimes, the user might
    # prefer e.g. path names to type names, and would rather have some suffixes
    # than a worse base name.
    for candidate in candidates:
      if candidate not in self.namespace:
        return self.namespace.add(candidate)
    return self.namespace.get_new_name(candidates[0], "")

  @abc.abstractmethod
  def name_for(self, value: Any, paths: List[daglish.Path]) -> str:
    """Returns a name for an object, given a list of paths to it.

    Args:
      value: Object to name, generally a Buildable or collection.
      paths: List of paths to the object.
    """


def suffix_first_path(path: daglish.Path) -> Optional[str]:
  """Returns a short-ish name from a Daglish path.

  This finds the last Attr in a path, and returns a name from that Attr through
  the end of the path.

  Args:
    path: Daglish path.
  """
  found_attr = False
  result = []

  def _add_sanitized_part(part):
    part = namespace_lib.valid_name_chars(namespace_lib.camel_to_snake(part))
    if part:
      result.insert(0, part)

  for part in reversed(path):
    if isinstance(part, daglish.Attr):
      found_attr = True
      _add_sanitized_part(part.name)
      break
    elif isinstance(part, daglish.Key):
      _add_sanitized_part(str(part.key))
    elif isinstance(part, daglish.Index):
      _add_sanitized_part(str(part.index))

  # Only return a name if a valid Attr was found.
  return namespace_lib.py_var_name("_".join(result)) if found_attr else None


@dataclasses.dataclass
class PathFirstNamer(Namer):
  """Namer that chooses path names over type names.

  Example 1:
    Config: fdl.Config(Foo, bar={"qux": fdl.Config(Qux)})
    The name for `config.bar` will be "bar", the name for `config.bar["qux"]`
    will be "bar_qux", and the name for the root object will be "foo".

  Example 2:
    Config: [{1: "hi"}]
    An exception will be thrown when trying to get a name for `config[0]` and
    `config[0][1]`, and `config` will be named "root".
  """

  def name_for(self, value: Any, paths: List[daglish.Path]) -> str:
    """See base class."""
    paths = [path for path in paths if path]  # remove empty/root paths
    candidates = []
    for path in paths:
      name = suffix_first_path(path)
      if name:
        candidates.append(name)
    if isinstance(value, config_lib.Buildable):
      fn_or_cls = config_lib.get_callable(value)
      try:
        cls_name = fn_or_cls.__name__
      except AttributeError:
        # This can happen on edge cases where we have
        # `fdl.Config(some_callable)`, where `some_callable` is an instance of
        # a class that has a `__call__` method. These cases generally aren't
        # serializable and certainly against our style guidelines/preferences,
        # but we still support them in the core API.
        pass
      else:
        candidates.append(_camel_to_snake(cls_name))

    if not candidates and not paths:
      candidates.append("root")
    if not candidates:
      candidates.append("unnamed_var")
    return self.name_from_candidates(candidates)


@dataclasses.dataclass
class TypeFirstNamer(Namer):
  """Namer that chooses type names over path names.

  See PathFirstNamer

  Example 2:
    Config: [{1: "hi"}]
    An exception will be thrown when trying to get a name for `config[0]` and
    `config[0][1]`, and `config` will be named "root".
  """

  def name_for(self, value: Any, paths: List[daglish.Path]) -> str:
    """See base class."""
    candidates = []

    if isinstance(value, config_lib.Buildable):
      fn_or_cls = config_lib.get_callable(value)
      try:
        cls_name = fn_or_cls.__name__
      except AttributeError:
        # This can happen on edge cases where we have
        # `fdl.Config(some_callable)`, where `some_callable` is an instance of
        # a class that has a `__call__` method. These cases generally aren't
        # serializable and certainly against our style guidelines/preferences,
        # but we still support them in the core API.
        pass
      else:
        candidates.append(_camel_to_snake(cls_name))

    paths = [path for path in paths if path]  # remove empty/root paths
    for path in paths:
      name = suffix_first_path(path)
      if name:
        candidates.append(name)

    if not candidates and not paths:
      candidates.append("root")
    if not candidates:
      raise NameGenerationError(
          f"Could not generate any candidate names for {value!r} with "
          f"paths {paths!r}"
      )
    return self.name_from_candidates(candidates)


def get_task_existing_names(task: code_ir.CodegenTask) -> Set[str]:
  """Get existing names from a CodegenTask."""
  names = copy.copy(task.global_namespace.names)
  names.update(
      fn.name.value for fn in task.top_level_call.all_fixture_functions()
  )
  return names


def get_fn_existing_names(fn: code_ir.FixtureFunction) -> Set[str]:
  """Get existing names from a FixtureFunction."""
  names = set()
  names.update(parameter.name.value for parameter in fn.parameters)
  names.update(variable.name.value for variable in fn.variables)
  return names
