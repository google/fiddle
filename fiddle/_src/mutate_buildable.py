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

"""Library for mutating Buildable instances."""

from fiddle import config

_buildable_internals_keys = ('__fn_or_cls__', '__signature__', '__arguments__',
                             '_has_var_keyword', '__argument_tags__',
                             '__argument_history__')


def move_buildable_internals(*, source: config.Buildable,
                             destination: config.Buildable):
  """Changes the internals of `destination` to be equivalent to `source`.

  Currently, this results in aliasing behavior, but this should not be relied
  upon. All the internals of `source` are aliased into `destination`.

  Args:
    source: The source configuration to pull internals from.
    destination: One of the buildables to swap.
  """
  if type(source) is not type(destination):
    # TODO: Relax this constraint such that both types merely need to be
    # buildable's.
    raise TypeError(f'types must match exactly: {type(source)} vs '
                    f'{type(destination)}.')

  # Update in-place using object.__setattr__ to bypass argument checking.
  for attr in _buildable_internals_keys:
    object.__setattr__(destination, attr, getattr(source, attr))
