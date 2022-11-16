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

"""A Config that builds a `types.SimpleNamespace` from any argument."""

import types
from typing import Any, Iterable

from fiddle import config


def _kwargs_to_namespace(**kwargs: Any) -> types.SimpleNamespace:
  return types.SimpleNamespace(**kwargs)


class NamespaceConfig(config.Config):
  """A Config that builds a `types.SimpleNamespace` accepting all arg names."""

  def __init__(self, /, *args, **kwargs):
    super().__init__(_kwargs_to_namespace, *args, **kwargs)

  @classmethod
  def __unflatten__(
      cls,
      values: Iterable[Any],
      metadata: config.BuildableTraverserMetadata,
  ):
    return cls(**metadata.arguments(values))
