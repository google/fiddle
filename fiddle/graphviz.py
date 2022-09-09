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

"""Provides a renderer to visualize a DAG of `fdl.Buildable`s via Graphviz."""

import abc
import collections
import copy
import dataclasses
import functools
import html
import itertools
import types

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Type, Union

from fiddle import config as fdl
from fiddle import daglish
from fiddle import tag_type
from fiddle import tagging
from fiddle.codegen import formatting_utilities
from fiddle.experimental import daglish_legacy
from fiddle.experimental import diff as fdl_diff
import graphviz
import typing_extensions

_BUILDABLE_INSTANCE_COLORS = [
    '#ffc0cb',  # pink
    '#90ee90',  # lightgreen
    '#fff8dc',  # cornsilk
    '#ffa07a',  # lightsalmon
    '#add8e6',  # lightblue
    '#ff8c00',  # darkorange
    '#8fbc8f',  # darkseagreen
    '#adff2f',  # greenyellow
    '#ff6347',  # tomato
    '#db7093',  # palevioletred
    '#f0e68c',  # khaki
    '#32cd32',  # limegreen
    '#00bfff',  # deepskyblue
    '#7b68ee',  # mediumslateblue
]

# Default color for headers (e.g., for lists and dicts).
_DEFAULT_HEADER_COLOR = '#eeeeee'
_DEFAULT_EDGE_COLOR = '#00000030'

_TAG_COLOR = '#606060'

# Colors for diffs.
_DIFF_FILL_COLORS = {
    'del': '#ffc0c0:#ffa0a0',
    'add': '#c0ffc0:#80c080',
    None: '#dddddd:#bbbbbb'
}
_DIFF_EDGE_COLORS = {'del': '#ff000030', 'add': '#00a00030', None: '#00000030'}


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


@dataclasses.dataclass(frozen=True)
class _NoValue:
  """Sentinel object used by _ChangedValue to mark missing values."""


@dataclasses.dataclass
class _ChangedValue:
  """Node to visualize a value that was changed by a diff."""
  old_value: Union[Any, _NoValue]
  new_value: Union[Any, _NoValue]


@dataclasses.dataclass(frozen=True)
class _AddedTag:
  """Node to visualize tags added by a diff."""
  tag: tag_type.TagType


@dataclasses.dataclass(frozen=True)
class _RemovedTag:
  """Node to visualize tags removed by a diff."""
  tag: tag_type.TagType


@dataclasses.dataclass
class _ChangedBuildable:
  """Node to visualize a buildable that was changed by a diff."""
  buildable_type: Type[fdl.Buildable]
  old_callable: Any
  new_callable: Any
  arguments: Dict[str, Union[Any, _ChangedValue]]
  tags: Dict[str, Set[Union[tag_type.TagType, _AddedTag, _RemovedTag]]]


# Function mapping value -> (header_color, edge_color).
InstanceColorFunc = Callable[[Any], Tuple[str, str]]

# Type for the _GraphvizRenderer.instance_colors parameter.
InstanceColorsType = Optional[Union[List[str], InstanceColorFunc]]


class _GraphvizRenderer:
  """Encapsulates state maintained while rendering a `Config` to Graphviz."""

  def __init__(self,
               instance_colors: InstanceColorsType = None,
               max_sequence_elements_per_row: int = 10,
               curved_edges: bool = False):
    """Initializes the render.

    Args:
      instance_colors: Optional list of HTML hexadecimal color codes (e.g.
        '#a1b2c3') to override the default set used to assign colors to
        `fdl.Buildable` instances encountered during rendering. Colors are
        assigned in order (and repeated if there are more instances than
        provided colors); or a function that maps objects to tuples
        `(header_color, edge_color)`.
      max_sequence_elements_per_row: When rendering sequences, up to this many
        elements will be included in a single row of the output table, with
        additional rows added to render remaining elements.
      curved_edges: If true, then draw edges using curved splines.
    """
    self._instance_colors = instance_colors or _BUILDABLE_INSTANCE_COLORS
    self._max_sequence_elements_per_row = max_sequence_elements_per_row
    self._attr_defaults = {
        # Should contain a mapping of tag names to dicts of attribute defaults.
        'table': {
            'border': 0,
            'cellspacing': 0,
            'cellborder': 1,
            'cellpadding': 3
        },
    }
    self._graph_attr = {'overlap': 'false'}
    self._edge_attr = {'color': '#00000030', 'penwidth': '3'}
    self._node_attr = {
        'fontname': 'Courier',
        'fontsize': '10',
        'shape': 'none',
        'margin': '0',
    }
    if curved_edges:
      self._graph_attr['splines'] = 'curved'
    self._clear()

  def _clear(self):
    """Resets all state associated with this renderer."""
    self._dot = graphviz.Graph(
        graph_attr=self._graph_attr,
        node_attr=self._node_attr,
        edge_attr=self._edge_attr)
    self._node_id_by_value_id = {}
    # The id of the config currently being rendered. This is obtained from
    # _config_counter and used in _add_node_for_value and _render_nested_value.
    self._current_id: Optional[int] = None
    self._config_counter = itertools.count()
    # Used to assign unique port names to table cells for nested configs.
    self._port_counter = itertools.count()
    # Ids of values reachable via multiple paths.
    self._shared_value_ids: Set[int] = set()

  def _color(self, value: Any) -> str:
    """Returns the header color for the given value."""
    if callable(self._instance_colors):
      return self._instance_colors(value)[0]
    else:
      node_id = self._node_id_by_value_id[id(value)]
      return self._instance_colors[node_id % len(self._instance_colors)]

  def _edge_color(self, value: Any) -> str:
    """Returns the color for an edge to the given value."""
    if callable(self._instance_colors):
      return self._instance_colors(value)[1]
    else:
      return _DEFAULT_EDGE_COLOR

  def tag(self, tag: str, **kwargs) -> Callable[[Any], str]:
    """Returns a function that creates HTML tags of type `tag`.

    Example:

        td = self.tag('td', colspan=2)
        td('cell contents') => '<td colspan="2">cell contents</td>

    Args:
      tag: The type of tag to create (e.g. 'table', 'font', etc).
      **kwargs: Attributes to apply to the tag. Tag-specific default values for
        attributes can be provided in `self._attr_defaults`. The values provided
        here in **kwargs take precedence over defaults.

    Returns:
      A function that can be called with contents to put inside the tag, and
      returns HTML markup enclosing the contents inside `tag`. If the given
      contents is a list or tuple, the list elements are automatically
      concatenated. Otherwise, the contents are simply formatted as a string.
    """
    attr_values = {**self._attr_defaults.get(tag, {}), **kwargs}
    attributes = [
        f' {key.replace("_", "-")}="{value}"' if value is not None else ''
        for key, value in attr_values.items()
    ]

    def tag_fn(contents: Any) -> str:
      if isinstance(contents, (list, tuple)):
        contents = ''.join(contents)
      return f'<{tag}{"".join(attributes)}>{contents}</{tag}>'

    return tag_fn

  def _header_row(self,
                  header,
                  colspan: int = 2,
                  bgcolor: str = _DEFAULT_HEADER_COLOR,
                  style: str = 'solid'):
    """Constructs a header table row."""
    tr = self.tag('tr')
    header_td = self.tag('td', colspan=colspan, bgcolor=bgcolor, style=style)
    return tr(header_td(header))

  def render(self, value: Any) -> graphviz.Graph:
    """Renders the given value, recursively rendering any nested values."""
    self._find_shared_value_ids(value)
    self._add_node_for_value(value)
    dot = self._dot
    self._clear()
    return dot

  def _find_shared_value_ids(self, root: Any):
    """Adds ids of shared values in `root` to `self._shared_value_ids`."""

    visited_ids = set()

    def visit(path, value):
      del path  # Unused.
      if not daglish.is_memoizable(value):
        return value
      if id(value) in visited_ids:
        self._shared_value_ids.add(id(value))
        return value
      elif isinstance(value, _ChangedValue):
        daglish_legacy.traverse_with_path(visit, value.old_value)
        daglish_legacy.traverse_with_path(visit, value.new_value)
      elif isinstance(value, _ChangedBuildable):
        daglish_legacy.traverse_with_path(visit, value.arguments)
      visited_ids.add(id(value))
      return (yield)

    daglish_legacy.traverse_with_path(visit, root)

  def _add_node_for_value(self, value: Any):
    """Adds a node for `value` to the graphviz graph `self._dot`.

    Also sets self._node_id_by_value[id(value)] to the node's graphviz id,
    and sets `_current_id` to the node's graphviz id.

    Args:
      value: The value that should be rendered.
    """
    if id(value) in self._node_id_by_value_id:
      return  # Don't do anything if we have already rendered this node.
    last_id = self._current_id
    self._current_id = next(self._config_counter)
    self._node_id_by_value_id[id(value)] = self._current_id
    html_label = self._render_value(value, self._color(value))
    already_tabular_types = (fdl.Buildable, dict, list, tuple,
                             _ChangedBuildable)
    if not (isinstance(value, already_tabular_types) or
            daglish_legacy.is_namedtuple_instance(value)):
      table = self.tag('table')
      tr = self.tag('tr')
      td = self.tag('td')
      html_label = table([
          self._header_row(
              type(value).__name__, bgcolor=self._color(value), colspan=1),
          tr(td(html_label))
      ])
    self._dot.node(str(self._current_id), f'<{html_label}>')

    self._current_id = last_id

  def _render_tagged_value(self, tagged_value: fdl.Buildable,
                           bgcolor: str) -> str:
    """Returns an HTML string rendering `tagged_value`."""
    type_font = self.tag('font', point_size=8)
    type_name = tagged_value.__class__.__name__
    key_names = ', '.join(repr(key.name) for key in tagged_value.tags)
    title = (type_font(html.escape(f'{type_name}: {key_names}')) + '&nbsp;')
    header = self._header_row(title, colspan=1, bgcolor=bgcolor, style='solid')

    table = self.tag('table')
    tr = self.tag('tr')
    value_td = self.tag('td', align='left')
    rendered_value = self._render_nested_value(tagged_value.value)
    return table([header, tr([value_td(rendered_value)])])

  def _render_config(self, config: fdl.Buildable, bgcolor: str) -> str:
    """Returns an HTML string rendering the Buildable `config`."""
    # Generate the header row.
    style = 'dashed' if isinstance(config, fdl.Partial) else 'solid'
    type_font = self.tag('font', point_size=8)
    type_name = config.__class__.__name__
    fn_or_cls_name = getattr(
        config.__fn_or_cls__, '__qualname__',
        getattr(config.__fn_or_cls__, '__name__', repr(config.__fn_or_cls__)))
    title = (
        type_font(html.escape(f'{type_name}:')) + '&nbsp;' +
        html.escape(fn_or_cls_name))
    header = self._header_row(title, bgcolor=bgcolor, style=style)

    # Generate the arguments table.
    if config.__arguments__:
      label = self._render_dict(
          config.__arguments__,
          header=header,
          key_format_fn=str,
          tags=config.__argument_tags__)
    else:
      table = self.tag('table')
      italics = self.tag('i')
      label = table([header, self._header_row(italics('no arguments'))])
    return label

  def _render_changed_buildable(self, config: fdl.Buildable,
                                bgcolor: str) -> str:
    """Returns an HTML string rendering the Buildable `config`."""
    # Generate the header row.
    style = 'dashed' if isinstance(config, fdl.Partial) else 'solid'
    type_font = self.tag('font', point_size=8)
    type_name = config.buildable_type.__name__
    table = self.tag('table', cellborder='0')
    tr = self.tag('tr')
    td = self.tag('td')
    td_old = self.tag('td', bgcolor=_DIFF_FILL_COLORS['del'])
    td_new = self.tag('td', bgcolor=_DIFF_FILL_COLORS['add'])
    if config.old_callable is config.new_callable:
      title = (
          type_font(html.escape(f'{type_name}:')) + '&nbsp;' +
          html.escape(config.old_callable.__name__))
    else:
      title = table(
          tr([
              td(type_font(html.escape(f'{type_name}:'))),
              td_old(html.escape(config.old_callable.__name__)),
              td('&rarr;'),
              td_new(html.escape(config.new_callable.__name__))
          ]))
    header = self._header_row(title, bgcolor=bgcolor, style=style)

    # Generate the arguments table.
    if config.arguments:
      label = self._render_dict(
          config.arguments, header=header, key_format_fn=str, tags=config.tags)
    else:
      table = self.tag('table')
      italics = self.tag('i')
      label = table([header, self._header_row(italics('no arguments'))])
    return label

  def _render_value(self, value: Any, color=_DEFAULT_HEADER_COLOR) -> str:
    """Returns an HTML string rendering `value`."""
    if value is tagging.NO_VALUE:
      return self.tag('i')('tagging.NO_VALUE')
    elif isinstance(value, CustomGraphvizBuildable):
      return value.__render_value__(self)
    elif isinstance(value, tagging.TaggedValueCls):
      return self._render_tagged_value(value, color)
    elif isinstance(value, fdl.Buildable):
      return self._render_config(value, color)
    elif isinstance(value, _ChangedValue):
      return self._render_changed_value(value)
    elif isinstance(value, _ChangedBuildable):
      return self._render_changed_buildable(value, color)
    elif isinstance(value, dict):
      return self._render_dict(
          value, header=self._header_row(type(value).__name__, bgcolor=color))
    elif daglish_legacy.is_namedtuple_instance(value):
      return self._render_dict(
          value._asdict(),
          header=self._header_row(type(value).__name__, bgcolor=color),
          key_format_fn=str)
    elif isinstance(value, (list, tuple)):
      return self._render_sequence(value, color)
    else:
      return self._render_leaf(value)

  def _render_nested_value(self, value: Any):
    """Returns an HTML string rendering `value` inside another object.

    If `value` is a `Buildable` or a shared memoizable value, then it is
    rendered as its own separate unnested graph node (if it hasn't been
    already). The nested rendering is then just a single-celled table whose
    color matches the header color of the separate node rendering. The nested
    cell is then connected via an edge to the separate rendering.

    Otherwise, this returns `self._render_value(value)`.

    Args:
      value: The nested value to render.

    Returns:
      The HTML markup for the rectangle (a single-celled table) that is
      connected by an edge to the node for `value`.
    """
    # If this is not a Buildable or shared value, then render it using
    # _render_value.
    buildable_types = (fdl.Buildable, _ChangedBuildable)
    if not (id(value) in self._shared_value_ids or
            (isinstance(value, buildable_types) and
             not isinstance(value, CustomGraphvizBuildable))):
      return self._render_value(value)

    # First, add the value to the graph. This will add a separate node and
    # render it as a separate table (but is a no-op if it's already added).
    self._add_node_for_value(value)

    # Look up the node id for the value, and get a new unique port name to
    # use for the nested cell (see below).  The port allows the edge to go
    # "inside" the parent value's node and connect directly to the table cell.
    node_id = self._node_id_by_value_id[id(value)]
    port = next(self._port_counter)
    # Now add an edge in the graph to the parent value. The direction here
    # determines the order in which the graph is layed out when using the
    # default "dot" layout engine, so putting the parent value first lays the
    # graph out from root to children.
    if callable(self._instance_colors):
      edge_attrs = dict(color=self._edge_color(value))
    else:
      edge_attrs = {}
    self._dot.edge(f'{self._current_id}:{port}:c', f'{node_id}:c', **edge_attrs)

    # Return a table with a single colored cell, using the port name from above.
    style = 'dashed' if isinstance(value, fdl.Partial) else 'solid'
    table = self.tag('table', style=style)
    tr = self.tag('tr')
    td = self.tag('td', port=port, bgcolor=self._color(value), style=style)
    return table(tr(td('')))

  def _render_sequence(self, sequence: Union[List[Any], Tuple[Any]],
                       color) -> str:
    """Renders the given sequence (list or tuple) as an HTML table."""
    table = self.tag('table')
    tr = self.tag('tr')
    td = self.tag('td')
    index_td = self.tag('td', cellpadding=0, bgcolor=_DEFAULT_HEADER_COLOR)
    index_font = self.tag('font', point_size=6)
    ellipsis_td = self.tag('td', rowspan=2)

    type_name = type(sequence).__name__
    if isinstance(sequence, fdl_diff.ListPrefix):
      type_name = 'Sequence'

    if not sequence and not isinstance(sequence, fdl_diff.ListPrefix):
      return '[]'

    cells, indices = [], []
    for i, value in enumerate(sequence):
      cells.append(td(self._render_nested_value(value)))
      indices.append(index_td(index_font(i)))
    if isinstance(sequence, fdl_diff.ListPrefix):
      sequence = list(sequence) + ['...']
      cells.append(ellipsis_td('...'))
    row_stride = self._max_sequence_elements_per_row
    colspan = min(len(sequence), row_stride)
    rows = [self._header_row(type_name, colspan=colspan, bgcolor=color)]
    for i in range(0, len(sequence), row_stride):
      rows.extend([tr(cells[i:i + row_stride]), tr(indices[i:i + row_stride])])
    return table(rows)

  def _render_dict(
      self,
      dict_: Dict[str, Any],
      header: Optional[str] = None,
      key_format_fn: Callable[[Any], str] = repr,
      tags: Optional[Dict[str, Set[tag_type.TagType]]] = None) -> str:
    """Renders the given `dict_` as an HTML table.

    Args:
      dict_: The `dict` to render.
      header: A table row containing a header row to include. The table row's
        table cell should have colspan="2" to render properly in the table.
      key_format_fn: A function to apply to dictionary keys to conver them into
        a string representation. Defaults to `repr`.
      tags: Optional tags for dictionary entries.

    Returns:
      The HTML markup for the resulting table representing `dict_`.
    """
    table = self.tag('table')
    tr = self.tag('tr')
    key_td = self.tag('td', align='right', bgcolor=_DEFAULT_HEADER_COLOR)
    value_td = self.tag('td', align='left')

    if tags is None:
      tags = {}

    rows = [header] if header is not None else []
    for key, value in dict_.items():
      key_str = html.escape(key_format_fn(key))
      value_str = self._render_nested_value(value)
      key_tags = tags.get(key, ())
      if key_tags:
        key_str = self._render_tags(key_str, key_tags)
      rows.append(tr([key_td(key_str), value_td(value_str)]))
    return table(rows)

  def _render_tags(self, arg_name, tags) -> str:
    """Renders the name and tags for a Buildable argument to HTML markup."""
    tag_table = self.tag('table', border='0', cellborder='0')
    tag_font = self.tag('font', face='arial', color=_TAG_COLOR)
    tr = self.tag('tr')
    td = self.tag('td', align='right')
    add_td = self.tag('td', align='right', bgcolor=_DIFF_FILL_COLORS['add'])
    del_td = self.tag('td', align='right', bgcolor=_DIFF_FILL_COLORS['del'])
    italic = self.tag('i')
    rows = [tr(td(arg_name))]
    for tag in sorted(tags, key=repr):
      if isinstance(tag, _AddedTag):
        rows.append(tr(add_td(tag_font(italic(['Tag: ', repr(tag.tag)])))))
      elif isinstance(tag, _RemovedTag):
        rows.append(tr(del_td(tag_font(italic(['Tag: ', repr(tag.tag)])))))
      else:
        rows.append(tr(td(tag_font(italic(['Tag: ', repr(tag)])))))
    return tag_table(rows)

  def _render_leaf(self, value: Any) -> str:
    """Renders `value` as its `__repr__` string."""
    value = formatting_utilities.pretty_print(value)
    return html.escape(repr(value))

  def _render_changed_value(self, value) -> str:
    """Renders a `_ChangedValue` as an HTML table."""
    table = self.tag('table', border='0', cellborder='0')
    tr = self.tag('tr')
    td = self.tag('td')
    td_del = self.tag('td', bgcolor=_DIFF_FILL_COLORS['del'])
    td_add = self.tag('td', bgcolor=_DIFF_FILL_COLORS['add'])

    row = []
    if value.old_value != _NoValue():
      row.append(td_del(self._render_nested_value(value.old_value)))
    if value.old_value != _NoValue() and value.new_value != _NoValue():
      row.append(td('&rarr;'))
    if value.new_value != _NoValue():
      row.append(td_add(self._render_nested_value(value.new_value)))
    return table(tr(row))


def render(config: Any) -> graphviz.Graph:
  """Renders the given `config` as a `graphviz.Graph`.

  Each config is rendered as a table of keys and values (with a header row).
  Any nested configs get their own separate table, with an edge pointing to
  their location in their parent config. If a config instance is present in
  multiple parent configs, it is only rendered once but will have multiple
  edges to parent configs.

  Args:
    config: The `fdl.Buildable` (or nested structure of `fdl.Buildable`s) to
      render.

  Returns:
    A `graphviz.Graph` object containing the resulting rendering of `config`.
    Standard `graphviz` methods can then be used to export this to a file.
  """
  return _GraphvizRenderer().render(config)


def render_diff(diff: Optional[fdl_diff.Diff] = None,
                *,
                old: Optional[Any] = None,
                new: Optional[Any] = None) -> graphviz.Graph:
  """Renders the given diff as a `graphviz.Graph`.

  Should be called using one of the following signatures:
    * `render_diff(diff=...)`
    * `render_diff(diff=..., old=...)`
    * `render_diff(old=..., new=...)`

  Args:
    diff: The diff to render.  If not specified, then the diff between `old` and
      `new` will be computed and rendered.
    old: The structure modified by the diff.  If not specified, then use a
      minimal config that can be used as the source for the diff.
    new: The result of the diff.  May not be specified if `diff` is specified.

  Returns:
    A `graphviz.Graph` object containing the resulting rendering of the diff.
    Standard `graphviz` methods can then be used to export this to a file.
  """
  if ((diff is None and (old is None or new is None)) or
      (diff is not None and new is not None)):
    raise TypeError(
        'render_diff must be called with one of the following signatures:\n'
        '  * render_diff(diff=...)\n'
        '  * render_diff(diff=..., old=...)\n'
        '  * render_diff(old=..., new=...)')
  if diff is None:
    diff = fdl_diff.build_diff(old, new)
  if old is None:
    old = fdl_diff.skeleton_from_diff(diff)
  config = _record_changed_values_from_diff(diff, old)
  old_value_ids = _find_old_value_ids(config)
  new_value_ids = _find_new_value_ids(config)
  fill_color = functools.partial(
      _diff_color,
      added_value_ids=new_value_ids - old_value_ids,
      deleted_value_ids=old_value_ids - new_value_ids)
  return _GraphvizRenderer(
      instance_colors=fill_color, curved_edges=True).render(config)


# TODO If memoized_traverse is updated to allow access to the
# memo dict, then we could refactor this code to not need this NamedTuple.
class _OldAndNewSharedValues(NamedTuple):
  """A NamedTuple that pairs an `old` structure with a diff's new shared values.

  This is used as a top-level node to traverse all the values that are relevant
  to rendering a diff.
  """
  old: Any
  new_shared_values: List[Any]


def _record_changed_values_from_diff(diff: fdl_diff.Diff, old: Any) -> Any:
  """Returns a copy of `old`, with `_ChangedValue` nodes used to show changes.

  Args:
    diff: A `Diff` describing changes to `old`.
    old: A nested structure.

  Returns:
    A copy of `old`, where any value that is changed by the diff is replaced
    by a `_ChangedValue` object, which points to both the new and the old value.
    Note: the nested structure returned by `record_changes` may contain cycles
    if you traverse through `_ChangedValue` objects.
  """
  # Update `diff` to replace any references with the objects they point to.
  diff = fdl_diff.resolve_diff_references(diff, old)

  # Index changes by their parent node.
  changes_by_parent = collections.defaultdict(list)
  for change in diff.changes:
    changes_by_parent[change.target[:-1]].append(change)

  # Traverse `old`, replacing any target of a `diff.change` with a
  # `_ChangedValue` object.  We do not fill in the `_ChangedValue.new_value`
  # fields yet, because we need to map new_values from original values (the
  # input to memoized_traverse) to transformed values (the output of
  # memoized_traverse).
  original_to_transformed = {}
  changed_values = []

  def record_change(paths, original_value):
    transformed_value: Any
    transformed_value = yield

    # Changes only apply to `old`, not to `new_shared_values`:
    paths = [p[1:] for p in paths if p and p[0].name == 'old']

    # If the value is a Buildable, then convert it to a _ChangedBuildable.
    if isinstance(original_value, fdl.Buildable):
      transformed_value = _ChangedBuildable(
          buildable_type=type(transformed_value),
          old_callable=transformed_value.__fn_or_cls__,
          new_callable=transformed_value.__fn_or_cls__,
          arguments=transformed_value.__arguments__,
          tags=copy.deepcopy(transformed_value.__argument_tags__))

    # If the value is a tuple, then temporarily convert it to a list so we
    # can modify it. If it's a namedtuple, then convert it to a SimpleNamespace.
    if daglish_legacy.is_namedtuple_instance(original_value):
      transformed_value = types.SimpleNamespace(**transformed_value._asdict())
    elif isinstance(original_value, tuple):
      transformed_value = list(transformed_value)

    # Record any changes to the children of this object.
    for path in paths:
      for change in changes_by_parent.get(path, ()):
        path_elt = change.target[-1]
        if (isinstance(change, fdl_diff.ModifyValue) and
            isinstance(path_elt, daglish.BuildableFnOrCls)):
          transformed_value.new_callable = change.new_value
          continue

        if isinstance(change, fdl_diff.AddTag):
          tags = transformed_value.tags.setdefault(path_elt.name, set())
          tags.add(_AddedTag(change.tag))
          continue

        elif isinstance(change, fdl_diff.RemoveTag) and change.target:
          tags = transformed_value.tags.setdefault(path_elt.name, set())
          tags.difference_update([change.tag])
          tags.add(_RemovedTag(change.tag))
          continue

        if isinstance(change, fdl_diff.SetValue):
          old_child = _NoValue()
        else:
          if isinstance(transformed_value, _ChangedBuildable):
            old_child = transformed_value.arguments[path_elt.name]
          else:
            old_child = path_elt.follow(transformed_value)
        child = _ChangedValue(old_child, _NoValue())
        changed_values.append((child, change))

        if isinstance(path_elt, daglish.Index):
          transformed_value[path_elt.index] = child
        elif isinstance(path_elt, daglish.Key):
          transformed_value[path_elt.key] = child
        elif isinstance(path_elt, daglish.Attr):
          if isinstance(transformed_value, _ChangedBuildable):
            transformed_value.arguments[path_elt.name] = child
          else:
            setattr(transformed_value, path_elt.name, child)
        else:
          raise ValueError(f'Unexpected PathElement {path_elt}')

    # Convert transformed_value back to a tuple or NamedTuple, if necessary.
    if daglish_legacy.is_namedtuple_instance(original_value):
      transformed_value = type(original_value)(**transformed_value.__dict__)
    elif isinstance(original_value, tuple):
      transformed_value = type(original_value)(transformed_value)

    # Record the mapping from the original to transformed value, so we can
    # substitute it in later.
    original_to_transformed[id(original_value)] = transformed_value

    return transformed_value

  new_values = [getattr(change, 'new_value', None) for change in diff.changes]
  old_and_new = _OldAndNewSharedValues(old, new_values)
  result = daglish_legacy.memoized_traverse(record_change, old_and_new).old

  # Set the `_ChangedValue.new_value` values.  We need to do this in a second
  # pass, because the graph can contain cycles, and we need to make sure that
  # we use the transformed version of each new_value.
  for changed_value, change in changed_values:
    if isinstance(change, (fdl_diff.SetValue, fdl_diff.ModifyValue)):
      if daglish.is_memoizable(changed_value.new_value):
        changed_value.new_value = original_to_transformed[id(change.new_value)]

  return result


def _find_new_value_ids(structure_with_changed_values: Any) -> Set[int]:
  """Returns ids of all objects reachable via _ChangedValue.new_value."""
  new_value_ids = set()

  def visit(path, value):
    del path  # Unused.
    if id(value) in new_value_ids:
      return
    if isinstance(value, _ChangedValue):
      daglish_legacy.traverse_with_path(visit, value.new_value)
    elif isinstance(value, _ChangedBuildable):
      daglish_legacy.traverse_with_path(visit, value.arguments)
    elif daglish.is_memoizable(value):
      new_value_ids.add(id(value))
    return (yield)

  daglish_legacy.traverse_with_path(visit, structure_with_changed_values)
  return new_value_ids


def _find_old_value_ids(structure_with_changed_values: Any) -> Set[int]:
  """Returns ids of all objects reachable via _ChangedValue.old_value."""
  old_value_ids = set()

  def visit(path, value):
    del path  # Unused.
    if id(value) in old_value_ids:
      return
    if isinstance(value, _ChangedValue):
      daglish_legacy.traverse_with_path(visit, value.old_value)
    elif isinstance(value, _ChangedBuildable):
      daglish_legacy.traverse_with_path(visit, value.arguments)
    elif daglish.is_memoizable(value):
      old_value_ids.add(id(value))
    return (yield)

  daglish_legacy.traverse_with_path(visit, structure_with_changed_values)
  return old_value_ids


def _diff_color(value: Any, added_value_ids: Set[int],
                deleted_value_ids: Set[int]):
  """Returns the color to use for `value` when rendering a diff."""
  if id(value) in added_value_ids:
    return _DIFF_FILL_COLORS['add'], _DIFF_EDGE_COLORS['add']
  elif id(value) in deleted_value_ids:
    return _DIFF_FILL_COLORS['del'], _DIFF_EDGE_COLORS['del']
  else:
    return _DIFF_FILL_COLORS[None], _DIFF_EDGE_COLORS[None]
