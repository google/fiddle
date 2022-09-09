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

"""Utility functions for tests that use fiddle.experimental.daglish."""

import re
from typing import Any, Dict, Set

from absl.testing import absltest
from fiddle import config
from fiddle import daglish
from fiddle.experimental import daglish_legacy
from fiddle.experimental import diff


def parse_path(path_str: str) -> daglish.Path:
  """Builds a daglish Path from a string.

  This is intended for use in test files, to make path constants easier to
  read and write.

  Limitations:
    * Only supports Index, Key, BuildableFnOrCls, and Attr.
    * Key.key must be a string, and may not contain nested quote marks.

  Args:
    path_str: A string that would be returned by `daglish.path_str(p)` for some
      path p.

  Returns:
    A Path `p` such that `daglish.path_str(p) == path_str`.
  """
  make_path_re = re.compile(r'\.(?P<attr>\w+)|'
                            r'\[(?P<index>\d+)\]|'
                            r'\[(?P<key>\'[^\']*\'|\"[^\"]+\")\]|'
                            r'(?P<error>.)')

  path = []
  for m in make_path_re.finditer(path_str):
    if m.group('attr'):
      if m.group('attr') == '__fn_or_cls__':
        path.append(daglish.BuildableFnOrCls())
      else:
        path.append(daglish.Attr(m.group('attr')))
    elif m.group('index'):
      path.append(daglish.Index(int(m.group('index'))))
    elif m.group('key'):
      path.append(daglish.Key(m.group('key')[1:-1]))
    else:
      raise ValueError(f'Unable to parse path {path_str!r} at {m}')
  path = tuple(path)
  return path


# Helper function to make expected References easier to write (and read).
def parse_reference(root: str, path: str) -> diff.Reference:
  """Build a diff.Reference from a string."""
  return diff.Reference(root, parse_path(path))


def get_shared_paths(structure: Any) -> Set[Set[daglish.Path]]:
  """Returns the set of paths for each shared object in `structure`.

  Args:
    structure: A traversable structure.

  Returns:
    A set containing one element for each 'shared' object in `structure` (i.e.,
    each object that is reachable via multiple paths).  The element for each
    object is the set of paths that can be used to reach that object.
  """
  result = set()

  def collect_value(paths: daglish.Paths, value: Any):
    if daglish.is_memoizable(value) and len(paths) > 1:
      result.add(frozenset(paths))
    return (yield)

  daglish_legacy.memoized_traverse(collect_value, structure)
  return result


def describe_dag_diffs(x, y):
  """Returns a list of strings describing differences between x and y."""
  diffs = []

  # A pair of dictionaries mapping id(x_val) or id(y_val) to the first path at
  # which that value was reached.  These are used to check that the sharing
  # stucture of `x` and `y` is the same.  In particular, if x_val is in x_memo,
  # then x_memo[id(x_val)] should be equal to y_memo[id(y_val)].  If not, then
  # the sharing structure is different.
  x_memo: Dict[int, daglish.Path] = {}
  y_memo: Dict[int, daglish.Path] = {}

  def values_diff_message(x_val, y_val, path):
    """A message indicating that `x_val` != `y_val` at `path`."""
    path_str = daglish.path_str(path)
    x_repr = repr(x_val)
    y_repr = repr(y_val)
    if len(x_repr) + len(y_repr) + len(path_str) < 70:
      return f'* x{path_str}={x_repr} but y{path_str}={y_repr}'
    else:
      # For longer values, it's easier to spot differences if the two
      # values are displayed on separate lines.
      return f'* x{path_str}={x_repr} but\n  y{path_str}={y_repr}'

  def find_diffs(x_val, y_val, path):
    """Adds differences between `x_val` and `y_val` to `diffs`."""

    # Compare the sharing structure of x_val and y_val.
    shared_x_path = x_memo.get(id(x_val))
    shared_y_path = y_memo.get(id(y_val))
    if shared_x_path is not None and shared_x_path == shared_y_path:
      return  # We have already compared x_val with y_val.

    if shared_x_path is None:
      x_memo[id(x_val)] = path
    else:
      path_str = daglish.path_str(path)
      x_path = daglish.path_str(shared_x_path)
      diffs.append(f'* Sharing diff: x{path_str} is x{x_path} but '
                   f'y{path_str} is not y{x_path}')

    if shared_y_path is None:
      y_memo[id(y_val)] = path
    else:
      path_str = daglish.path_str(path)
      y_path = daglish.path_str(shared_y_path)
      diffs.append(f'* Sharing diff: y{path_str} is y{y_path} but '
                   f'x{path_str} is not x{y_path}')

    # Compare x_val and y_val by type.
    if type(x_val) is not type(y_val):
      path_str = daglish.path_str(path)
      diffs.append(f'* type(x{path_str}) != type(y{path_str}): '
                   f'{type(x_val)} vs {type(y_val)}')
      return  # Don't report any futher differences between x_val and y_val.

    # Compare x_val and y_val by value.
    node_traverser = daglish.find_node_traverser(type(x_val))
    if node_traverser is None:
      if x_val != y_val:
        diffs.append(values_diff_message(x_val, y_val, path))

    else:
      x_children, x_metadata = node_traverser.flatten(x_val)
      y_children, y_metadata = node_traverser.flatten(y_val)
      x_path_elements = node_traverser.path_elements(x_val)
      y_path_elements = node_traverser.path_elements(y_val)

      if isinstance(x_metadata, config.BuildableTraverserMetadata):
        x_metadata = x_metadata.without_history()
      if isinstance(y_metadata, config.BuildableTraverserMetadata):
        y_metadata = y_metadata.without_history()

      if x_path_elements != y_path_elements:
        for path_elt in set(x_path_elements) - set(y_path_elements):
          child_path = daglish.path_str(path + (path_elt,))
          diffs.append(
              f'* x{child_path} has a value but y{child_path} does not.')
        for path_elt in set(y_path_elements) - set(x_path_elements):
          child_path = daglish.path_str(path + (path_elt,))
          diffs.append(
              f'* y{child_path} has a value but x{child_path} does not.')

      elif x_metadata != y_metadata:
        diffs.append(values_diff_message(x_val, y_val, path))

      else:
        # Recursively check children.  Note: we only recurse if type,
        # path_elements, and metadata are all equal.
        assert len(x_children) == len(y_children) == len(x_path_elements)
        for x_child, y_child, path_elt in zip(x_children, y_children,
                                              x_path_elements):
          find_diffs(x_child, y_child, path + (path_elt,))

  find_diffs(x, y, ())
  return sorted(diffs)


class TestCase(absltest.TestCase):
  """Mixin class for absltest.TestCase, that adds assertDagEqual method."""

  def assertDagEqual(self, x, y):
    """Asserts that two values are equal and have the same DAG structure.

    If `x` and `y` are not equal, or if they differ in their DAG (directed
    acyclic graph) structure, then raise a `self.failureException` message
    describing the differences between `x` and `y`.

    Note: `Config` objects that differ in whether they explicitly set a default
    parameter to the default value are considered to have differing DAG
    structure (even though they compare equal with `==`).

    Args:
      x: A structure traversable by daglish.  Must form a DAG (no cycles).
      y: A structure traversable by daglish.  Must form a DAG (no cycles).
    """
    diffs = describe_dag_diffs(x, y)
    if diffs:
      msg = 'x != y:\n' + '\n'.join(diffs)
      raise self.failureException(msg)

    # The following two lines should never fail (because all differences should
    # be caught by `describe_dag_diffs`.  But we include them as a backstop in
    # case `describe_dag_diffs` misses anything, because it's important that
    # test cases not have false positives.
    self.assertEqual(x, y)
    self.assertEqual(get_shared_paths(x), get_shared_paths(y))
