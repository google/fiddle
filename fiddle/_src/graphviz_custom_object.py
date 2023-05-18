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

"""API for objects that want to render in a particular way in Graphviz.

(This module was partly split out to avoid circular imports.)
"""
import abc
from typing import Any, Callable, Iterable

from fiddle._src import config as config_lib
import typing_extensions


class GraphvizRendererApi(typing_extensions.Protocol):
  """API of _GraphvizRenderer exposed to CustomGraphvizBuildable subclasses."""

  def tag(self, tag: str, **kwargs) -> Callable[[Any], str]:
    raise NotImplementedError()


class CustomGraphvizBuildable(metaclass=abc.ABCMeta):
  """Mixin class that marks a Buildable has having a custom __render_value__.

  This lets certain special-purpose Buildables customize how they are rendered.
  """

  @abc.abstractmethod
  def __render_value__(self, api: GraphvizRendererApi) -> Any:
    """Renders this Buildable as a value."""


def _raise_error():
  """Helper function for `Trimmed` that raises an error."""
  raise ValueError(
      'Please do not call fdl.build() on a config tree with '
      'Trimmed() nodes. These nodes are for visualization only.'
  )


class Trimmed(config_lib.Config[type(None)], CustomGraphvizBuildable):
  """Represents a configuration that has been trimmed."""

  def __init__(self):
    super().__init__(_raise_error)

  def __render_value__(self, api: GraphvizRendererApi) -> Any:
    return api.tag('i')('(trimmed...)')

  @classmethod
  def __unflatten__(
      cls,
      values: Iterable[Any],
      metadata: config_lib.BuildableTraverserMetadata,
  ):
    return cls()
