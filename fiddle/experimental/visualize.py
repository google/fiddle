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

"""Helper functions for visualization of Fiddle configurations.

By default, you should be able to call APIs in `graphviz` and `printing` with no
special handling. But some real-world configurations become gigantic, and so
having a few helper functions for trimming them for interactive demos and such
can be valuable.
"""
import copy
from typing import Any, Dict, Iterable, List

from fiddle import config
from fiddle import graphviz
import tree


def _raise_error():
  """Helper function for `Trimmed` that raises an error."""
  raise ValueError('Please do not call fdl.build() on a config tree with '
                   'Trimmed() nodes. These nodes are for visualization only.')


class Trimmed(config.Config[type(None)], graphviz.CustomGraphvizBuildable):
  """Represents a configuration that has been trimmed."""

  def __init__(self):
    super().__init__(_raise_error)

  def __render_value__(self, api: graphviz.GraphvizRendererApi) -> Any:
    return api.tag('i')('(trimmed...)')

  @classmethod
  def __unflatten__(
      cls,
      values: Iterable[Any],
      metadata: config.BuildableTraverserMetadata,
  ):
    return cls()


def trimmed(cfg: config.Buildable,
            trim: List[config.Buildable]) -> config.Buildable:
  """Returns a deep copy of a Buildable, with certain nodes replaced.

  This copy should have the same DAG structure as the original configuration.

  Example usage:

  graphviz.render(visualize.trimmed(cfg, [cfg.my_large_model_sub_config]))

  Args:
    cfg: Root buildable object.
    trim: List of sub-buildables that will be replaced by Trimmed() instances.

  Returns:
    Copy of the Buildable with nodes replaced.
  """

  return copy.deepcopy(cfg, {id(sub_cfg): Trimmed() for sub_cfg in trim})


def depth_over(cfg: config.Buildable, depth: int) -> List[config.Buildable]:
  """Returns a list of sub-Buildable objects over a given depth from the root.

  If a sub-Buildable has multiple paths from the root to it, its depth is the
  *minimum* distance of those paths.

  Args:
    cfg: Root buildable object.
    depth: Depth value.

  Returns:
    List of all sub-Buildables with depth strictly greater than `depth`.
  """
  node_to_depth: Dict[int, int] = {}  # Buildable hash --> min depth.
  id_to_node: Dict[int, config.Buildable] = {}  # Buildable hash --> Buildable.

  def _make_depth_map(node: config.Buildable, curr_depth: int):
    """Mapper function to generate the `node_to_depth` map."""
    if id(node) in node_to_depth:
      node_to_depth[id(node)] = min(node_to_depth[id(node)], curr_depth)
    else:
      node_to_depth[id(node)] = curr_depth
    id_to_node[id(node)] = node

    def leaf_fn(leaf):
      if isinstance(leaf, config.Buildable):
        _make_depth_map(leaf, curr_depth + 1)

    tree.map_structure(leaf_fn, node.__arguments__)

  _make_depth_map(cfg, 0)

  return [
      id_to_node[key]
      for key, node_depth in node_to_depth.items()
      if node_depth > depth
  ]
