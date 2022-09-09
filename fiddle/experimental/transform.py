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

"""Transformation functions for Fiddle buildables."""

from typing import Optional, TypeVar

from fiddle import config
from fiddle import daglish

# Any subclass of Buildable
AnyBuildable = TypeVar("AnyBuildable", bound=config.Buildable)


def unintern_tuples_of_literals(buildable: AnyBuildable) -> AnyBuildable:
  """Uninterns tuples of literals in the given buildable.

  Tuples of literals are tuples containing immutable primitive values. Python
  normally interns (i.e. share the same instance) tuples of literals. Interned
  objects are those that were declared in separate code locations but share the
  same instance (due to Python's optimizations). This can create problems with
  the differ and visualizer. This function "uninterns" these tuples (i.e.
  creates separate instances for each tuple) to solve this issue.

  Args:
    buildable: Any Fiddle buildable.

  Returns:
    A new buildable with the tuples uninterned.
  """

  def transform(value, state: Optional[daglish.State] = None):
    state = state or daglish.MemoizedTraversal.begin(
        transform, value, memoize_internables=False)
    # pylint: disable-next=unidiomatic-typecheck
    # We want tuples only and not things like NamedTuples which are not
    # interned by Python.
    if type(tuple) is tuple and daglish.is_internable(value):
      value = tuple(list(value))
    return state.map_children(value)

  return transform(buildable)
