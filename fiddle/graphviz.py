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
import html
import itertools

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set

from fiddle import config as fdl
from fiddle import tagging
from fiddle.codegen import formatting_utilities
from fiddle.experimental import daglish
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

_DEFAULT_HEADER_COLOR = '#eeeeee'


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


class _GraphvizRenderer:
  """Encapsulates state maintained while rendering a `Config` to Graphviz."""

  def __init__(self,
               instance_colors: Optional[List[str]] = None,
               max_sequence_elements_per_row: int = 10):
    """Initializes the render.

    Args:
      instance_colors: Optional list of HTML hexadecimal color codes (e.g.
        '#a1b2c3') to override the default set used to assign colors to
        `fdl.Buildable` instances encountered during rendering. Colors are
        assigned in order (and repeated if there are more instances than
        provided colors).
      max_sequence_elements_per_row: When rendering sequences, up to this many
        elements will be included in a single row of the output table, with
        additional rows added to render remaining elements.
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
    self._clear()

  def _clear(self):
    """Resets all state associated with this renderer."""
    self._dot = graphviz.Graph(
        graph_attr={
            'overlap': 'false',
        },
        node_attr={
            'fontname': 'Courier',
            'fontsize': '10',
            'shape': 'none',
            'margin': '0',
        },
        edge_attr={
            'color': '#00000030',
            'penwidth': '3',
        })
    self._node_id_by_value_id = {}
    # The id of the config currently being rendered. This is obtained from
    # _config_counter and used in _add_node_for_value and _render_nested_value.
    self._current_id: Optional[int] = None
    self._config_counter = itertools.count()
    # Used to assign unique port names to table cells for nested configs.
    self._port_counter = itertools.count()
    # Ids of values reachable via multiple paths.
    self._shared_value_ids: Set[int] = set()

  def _color(self, value):
    node_id = self._node_id_by_value_id[id(value)]
    return self._instance_colors[node_id % len(self._instance_colors)]

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
      visited_ids.add(id(value))
      return (yield)

    daglish.traverse_with_path(visit, root)

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
    if not (isinstance(value, (fdl.Buildable, dict, list, tuple)) or
            daglish.is_namedtuple_instance(value)):
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
    fn_or_cls_name = config.__fn_or_cls__.__name__
    title = (
        type_font(html.escape(f'{type_name}:')) + '&nbsp;' +
        html.escape(fn_or_cls_name))
    header = self._header_row(title, bgcolor=bgcolor, style=style)

    # Generate the arguments table.
    if config.__arguments__:
      label = self._render_dict(
          config.__arguments__, header=header, key_format_fn=str)
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
    elif isinstance(value, tagging.TaggedValue):
      return self._render_tagged_value(value, color)
    elif isinstance(value, fdl.Buildable):
      return self._render_config(value, color)
    elif isinstance(value, dict):
      return self._render_dict(
          value, header=self._header_row(type(value).__name__, bgcolor=color))
    elif daglish.is_namedtuple_instance(value):
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
    if not (id(value) in self._shared_value_ids or
            (isinstance(value, fdl.Buildable) and
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
    self._dot.edge(f'{self._current_id}:{port}:c', f'{node_id}:c')

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

    if not sequence:
      return '[]' if isinstance(sequence, list) else '()'

    cells, indices = [], []
    for i, value in enumerate(sequence):
      cells.append(td(self._render_nested_value(value)))
      indices.append(index_td(index_font(i)))
    row_stride = self._max_sequence_elements_per_row
    colspan = min(len(sequence), row_stride)
    rows = [
        self._header_row(
            type(sequence).__name__, colspan=colspan, bgcolor=color)
    ]
    for i in range(0, len(sequence), row_stride):
      rows.extend([tr(cells[i:i + row_stride]), tr(indices[i:i + row_stride])])
    return table(rows)

  def _render_dict(self,
                   dict_: Dict[str, Any],
                   header: Optional[str] = None,
                   key_format_fn: Callable[[Any], str] = repr) -> str:
    """Renders the given `dict_` as an HTML table.

    Args:
      dict_: The `dict` to render.
      header: A table row containing a header row to include. The table row's
        table cell should have colspan="2" to render properly in the table.
      key_format_fn: A function to apply to dictionary keys to conver them into
        a string representation. Defaults to `repr`.

    Returns:
      The HTML markup for the resulting table representing `dict_`.
    """
    table = self.tag('table')
    tr = self.tag('tr')
    key_td = self.tag('td', align='right', bgcolor=_DEFAULT_HEADER_COLOR)
    value_td = self.tag('td', align='left')

    rows = [header] if header is not None else []
    for key, value in dict_.items():
      key = key_format_fn(key)
      rows.append(tr([key_td(key), value_td(self._render_nested_value(value))]))
    return table(rows)

  def _render_leaf(self, value: Any) -> str:
    """Renders `value` as its `__repr__` string."""
    value = formatting_utilities.pretty_print(value)
    return html.escape(repr(value))


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
