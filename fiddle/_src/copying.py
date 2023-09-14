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

"""Functions to copy Fiddle `Buidlable`."""

import copy

from fiddle._src import config as config_lib
from fiddle._src import mutate_buildable


def copy_with(
    buildable: config_lib.BuildableT, **kwargs
) -> config_lib.BuildableT:
  """Returns a shallow copy of ``buildable`` with updates to arguments.

  Args:
    buildable: A ``Buildable`` (e.g. a ``fdl.Config``) to copy and mutate.
    **kwargs: The arguments and values to assign.
  """
  buildable = copy.copy(buildable)
  mutate_buildable.assign(buildable, **kwargs)
  return buildable


def deepcopy_with(buildable: config_lib.Buildable, **kwargs):
  """Returns a deep copy of ``buildable`` with updates to arguments.

  Note: if any ``Config``'s inside ``buildable`` are shared with ``Config``'s
  outside of ``buildable``, then they will no longer be shared in the returned
  value. E.g., if ``cfg1.x.y is cfg2``, then
  ``fdl.deepcopy_with(cfg1, ...).x.y is cfg2`` will be ``False``.

  Args:
    buildable: A ``Buildable`` (e.g. a ``fdl.Config``) to copy and mutate.
    **kwargs: The arguments and values to assign.
  """
  buildable = copy.deepcopy(buildable)
  mutate_buildable.assign(buildable, **kwargs)
  return buildable
