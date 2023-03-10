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

"""Helper class to express Daglish traversals that visit all parents first.

(This implies that all ancestors of a node have been visited before the node is
visited.)
"""

import dataclasses
from typing import Any, Callable, Dict, List, Set, TypeVar

from fiddle import daglish


@dataclasses.dataclass(frozen=True)
class CallbackWithDeps:
  parent_paths: Set[daglish.Path]
  callback: Callable[[], None]

  def __hash__(self):
    return id(self)


@dataclasses.dataclass(frozen=True)
class UniqueResult:
  """Proxy container for a unique result.

  We have some annoying mismatch between parents and sets of daglish paths,
  which this is designed to iron out with minimal intervention.
  """

  value: Any

  def __hash__(self):
    return id(self)


_NodeType = TypeVar("_NodeType")  # pylint: disable=invalid-name
_T = TypeVar("_T")


def traverse_parents_first(
    traverse_fn: Callable[[_NodeType, List[_T]], _T],
    root_obj: _NodeType,
    *,
    traversal_cls=daglish.MemoizedTraversal,
) -> None:
  """Runs a traversal that is dependent upon results from parent nodes.

  This is one of a few things traversals that is not immediately easy with
  daglish. It is also more subtle with DAGs than with trees where an algorithm
  like BFS works.

  Args:
    traverse_fn: The core function that will run once all of its parents have
      been run. Every invocation should produce a value of type T, and will be
      passed a list of its own outputs (of type List[T]) from the immediate
      parents. (For the root invocation, this will be the empty list.)
    root_obj: Root object, usually a `fdl.Buildable`.
    traversal_cls: Traversal class.
  """
  # Maps each path that a node is accessible by to the unique result of
  # processing it. So if a shared node is accessible by {path1, path2} and its
  # processed result (from calling `traverse_fn`) is `result`, then this
  # dictionary will contain {path1: result, path2: result}.
  results_by_any_path: Dict[daglish.Path, UniqueResult] = {}

  # Mapping from an unfinished parent path to callbacks, each of which
  # represents processing an immediate descendant.
  deferrals_by_unfinished_parent: Dict[daglish.Path, Set[CallbackWithDeps]] = {}

  def traverse(value, state: daglish.State) -> None:
    paths: List[daglish.Path] = state.get_all_paths()
    parent_paths: Set[daglish.Path] = {path[:-1] for path in paths if path}

    def run():
      unique_results = set(
          results_by_any_path[parent] for parent in parent_paths
      )
      parent_results = [unique_result.value for unique_result in unique_results]
      result = UniqueResult(traverse_fn(value, parent_results))
      for path in paths:
        results_by_any_path[path] = result

      # Invoke anything that has been waiting on this node.
      all_deferred = set()
      for path in paths:
        all_deferred.update(deferrals_by_unfinished_parent.get(path, {}))
      for deferred in all_deferred:
        if all(path in results_by_any_path for path in deferred.parent_paths):
          deferred.callback()

      # Traverse into children.
      state.map_children(value)

    if all(path in results_by_any_path for path in parent_paths):
      # Run immediately if possible.
      run()
    else:
      unfinished_parents = {
          path for path in parent_paths if path not in results_by_any_path
      }
      deferral = CallbackWithDeps(unfinished_parents, run)
      for path in unfinished_parents:
        deferrals_by_unfinished_parent.setdefault(path, set()).add(deferral)

  traversal_cls.run(traverse, root_obj)
