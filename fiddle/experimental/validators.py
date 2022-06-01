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

"""Verify properies about a buildable data structure.

This library is a collection of routines used to verify properties about
Fiddle data structures.
"""

import textwrap
from typing import Any, cast, Collection, Tuple, Type

from fiddle import config
from fiddle.experimental import daglish
from fiddle.experimental import daglish_generator


def _format_type(t: Type[Any]) -> str:
  return f'{t.__module__}.{t.__qualname__}'


class ValidationFailureError(ValueError):
  """An exception marking expected properties of a Fiddle DAG did not hold true.

  Attributes:
    msg: the human-readable message.
    bad_items: A collection of paths and items that failed validation.
  """
  msg: str
  bad_items: list[Tuple[daglish.Path, Type[Any]]]

  def __init__(self, msg: str, bad_items: list[Tuple[daglish.Path, Type[Any]]]):
    super().__init__()
    self.msg = msg
    self.bad_items = bad_items

  def __str__(self):
    bad_paths = '\n- '.join(f'{daglish.path_str(path)}: {_format_type(tpe)}'
                            for (path, tpe) in self.bad_items)
    bad_paths = textwrap.indent('- ' + bad_paths, '  ')
    return f'ValidationFailure: {self.msg}:\n{bad_paths}'


def assert_all_primitive_leaves(
    structure: daglish_generator.Structure,
    *,
    leaf_allowlist: Collection[
        Type[Any]] = daglish._IMMUTABLE_NONCONTAINER_TYPES):  # pylint: disable=protected-access
  """Verifies all objects reachable from `structure` are containers or primimtives.

  Args:
    structure: the root of the data structure to verify.
    leaf_allowlist: a list of types that are acceptable leaves. By default, the
      only allowed types are immutable primitive types (e.g. `int`, `bool`,
      `float`, etc.).

  Raises:
    ValidationFailureError: If there are non-primitive leaves in `structure`.
  """
  leaf_allowlist = set(leaf_allowlist)  # improve performance.
  bad_items = []
  for item in daglish_generator.iterate(
      structure, options=daglish_generator.IterateOptions.PRE_ORDER):
    if not item.is_set:
      continue
    if not item.is_leaf:
      continue
    if type(item.value) not in leaf_allowlist:
      bad_items.append((item.path, type(item.value)))
  if bad_items:
    allowed_type_names = sorted(t.__name__ for t in leaf_allowlist)
    raise ValidationFailureError(
        f'Inappropriate values in {type(structure).__name__} (allowed '
        f'types: {allowed_type_names}); bad items (format: `path: type`)',
        bad_items)


def assert_no_aliasing(structure: daglish_generator.Structure):
  """Asserts that each object occurs at exactly one path in the structure."""
  paths_by_id = daglish_generator.all_paths_by_id(structure)
  aliased_paths = [paths for paths in paths_by_id.values() if len(paths) > 1]
  if aliased_paths:
    raise ValidationFailureError(
        f'Objects are aliased unexpectedly ({len(aliased_paths)} object(s))',
        [
            (path, type(daglish.follow_path(structure, path)))  # pylint: disable=g-complex-comprehension
            for paths in aliased_paths  # pylint: disable=g-complex-comprehension
            for path in paths
        ])  # pylint: disable=g-complex-comprehension


def assert_all_required_values_set(structure: daglish_generator.Structure):
  """Asserts that all non-optional arguments have values."""
  missing_paths = []
  for item in daglish_generator.iterate(
      structure, options=daglish_generator.IterateOptions.PRE_ORDER):
    if not item.is_set:
      assert isinstance(item.parent, config.Buildable)
      parent: config.Buildable = cast(config.Buildable, item.parent)
      assert isinstance(item.path_element, daglish.Attr)
      path_element: daglish.Attr = cast(daglish.Attr, item.path_element)
      param = parent.__signature__.parameters[path_element.name]
      if param.default is not param.empty:
        # There's a default; so it's considered set.
        continue
      missing_paths.append(item.path)
  if missing_paths:
    raise ValidationFailureError('Missing required parameters',
                                 [(path, type(...)) for path in missing_paths])
