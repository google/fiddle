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

"""Utilities for type casting various fiddle types."""

import itertools
import logging
from typing import Type

from fiddle._src import config
from fiddle._src import partial

_SUPPORTED_CASTS = set()


def cast(
    new_type: Type[config.BuildableT], buildable: config.Buildable
) -> config.BuildableT:
  """Returns a copy of ``buildable`` that has been converted to ``new_type``.

  Requires that ``type(buildable)`` and ``type(new_type)`` be compatible.
  If the types may not be compatible, a warning will be issued, but the
  conversion will be attempted.

  Args:
    new_type: The type to convert to.
    buildable: The ``config.Buildable`` that should be copied and converted.
  """
  if not isinstance(buildable, config.Buildable):
    raise TypeError(
        f'Expected `buildable` to be a config.Buildable, got {buildable}'
    )
  if not isinstance(new_type, type) and issubclass(new_type, config.Buildable):
    raise TypeError(
        'Expected `new_type` to be a subclass of config.Buildable, got'
        f' {buildable}'
    )
  src_type = type(buildable)
  if (src_type, new_type) not in _SUPPORTED_CASTS:
    logging.warning(
        (
            'Conversion from %s to %s has not been marked as '
            'officially supported.  If you think this conversion '
            'should be supported, contact the Fiddle team.'
        ),
        src_type,
        new_type,
    )
  return new_type.__unflatten__(*buildable.__flatten__())


def register_supported_cast(src_type, dst_type):
  _SUPPORTED_CASTS.add((src_type, dst_type))


for _src_type, _dst_type in itertools.product(
    [config.Config, partial.Partial, partial.ArgFactory], repeat=2
):
  register_supported_cast(_src_type, _dst_type)
