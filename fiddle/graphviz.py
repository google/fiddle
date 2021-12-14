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

import html
import itertools

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from fiddle import config as fdl
from fiddle import placeholders
from fiddle.codegen import codegen
from fiddle.codegen import special_value_codegen
import graphviz

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


class LazyImportManager(special_value_codegen.ImportManagerApi):
  """Import manager for GraphViz.

  We create an instance of the codegen's import manager on every method call,
  which effectively will alias imports like 'jax.numpy' to 'jnp', but will not
  create new import aliases when two modules of the same name are referenced.
  """

  def add_by_name(self, module_name: str) -> str:
    namespace = codegen.Namespace()
    import_manager = codegen.ImportManager(namespace)
    return import_manager.add_by_name(module_name)


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
    self._configs = {}
    # The id of the config currently being rendered. This is obtained from
    # _config_counter and used in _render_config and _render_nested_config.
    self._current_id = None
    self._config_counter = itertools.count()
    # Used to assign unique port names to table cells for nested configs.
    self._port_counter = itertools.count()

  def _color(self, index: int):
    return self._instance_colors[index % len(self._instance_colors)]

  def _tag(self, tag: str, **kwargs) -> Callable[[Any], str]:
    """Returns a function that creates HTML tags of type `tag`.

    Example:

        td = self._tag('td', colspan=2)
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

  def _header_row(self, header, colspan=2, bgcolor='#eeeeee', style='solid'):
    """Constructs a header table row."""
    tr = self._tag('tr')
    header_td = self._tag('td', colspan=colspan, bgcolor=bgcolor, style=style)
    return tr(header_td(header))

  def render(self, config: fdl.Buildable) -> graphviz.Graph:
    """Renders the given `config`, recursively rendering any nested configs."""
    self._render_config(config)
    dot = self._dot
    self._clear()
    return dot

  def _render_config(self, config):
    """Adds `config` to the graph and renders it as a table."""
    if id(config) in self._configs:
      return  # Don't do anything if `config` has already been rendered.

    last_id = self._current_id
    self._current_id = next(self._config_counter)
    self._configs[id(config)] = self._current_id

    # Generate the header row.
    bgcolor = self._color(self._current_id)
    style = 'dashed' if isinstance(config, fdl.Partial) else 'solid'
    type_font = self._tag('font', point_size=8)
    type_name = config.__class__.__name__
    if isinstance(config, placeholders.Placeholder):
      title = type_font(f'{type_name}: {config.key.name!r}') + '&nbsp;'
      header = self._header_row(title, colspan=1, bgcolor=bgcolor, style=style)
    else:
      fn_or_cls_name = config.__fn_or_cls__.__name__
      title = type_font(f'{type_name}:') + '&nbsp;' + fn_or_cls_name
      header = self._header_row(title, bgcolor=bgcolor, style=style)

    # Generate the arguments table.
    if isinstance(config, placeholders.Placeholder):
      table = self._tag('table')
      tr = self._tag('tr')
      value_td = self._tag('td', align='left')
      label = table([header, tr([value_td(self._render_value(config.value))])])
    elif config.__arguments__:
      label = self._render_dict(
          config.__arguments__, header=header, key_format_fn=str)
    else:
      table = self._tag('table')
      italics = self._tag('i')
      label = table([header, self._header_row(italics('no arguments'))])
    self._dot.node(str(self._current_id), f'<{label}>')

    self._current_id = last_id

  def _render_value(self, value: Any):
    """Renders an arbitrary value inside a `Config` rendering."""
    if value is placeholders.NO_VALUE:
      return self._tag('i')('placeholders.NO_VALUE')
    elif isinstance(value, fdl.Buildable):
      return self._render_nested_config(value)
    elif isinstance(value, dict):
      return self._render_dict(
          value, header=self._header_row(type(value).__name__))
    elif isinstance(value, tuple) and hasattr(value, '_asdict'):
      # If it has an "_asdict" attribute it's almost certainly a `NamedTuple`.
      # Unfortunately there is no direct way of testing whether an object is a
      # `NamedTuple`, since all `NamedTuple`s are direct subclasses of `tuple`.
      return self._render_dict(
          value._asdict(),
          header=self._header_row(type(value).__name__),
          key_format_fn=str)
    elif isinstance(value, (list, tuple)):
      return self._render_sequence(value)
    else:
      return self._render_leaf(value)

  def _render_nested_config(self, config: fdl.Buildable) -> str:
    """Renders a nested config as a rectangle, linked to its parent config.

    This function renders the appearance of a config when it is nested inside
    another config (and potentially inside a list/dictionary within the config).
    First, `config` is rendered as its own separate unnested graph node (if it
    hasn't been already). The nested rendering is then just a single-celled
    table whose color matches the header color of the separate node rendering.
    The nested cell is then connected via an edge to the separate rendering.

    Args:
      config: The nested `fdl.Buildable` to render.

    Returns:
      The HTML markup for the resulting table representing `config`.
    """
    # First, add the config to the graph. This will add a separate node and
    # render it as a separate table (but is a no-op if it's already added).
    self._render_config(config)
    # Look up the config id for the config, and get a new unique port name to
    # use for the nested cell (see below).  The port allows the edge to go
    # "inside" the parent config and connect directly to the table cell.
    config_id = self._configs[id(config)]
    port = next(self._port_counter)
    # Now add an edge in the graph to the parent config. The direction here
    # determines the order in which the graph is layed out when using the
    # default "dot" layout engine, so putting the parent config first lays the
    # graph out from root to children.
    self._dot.edge(f'{self._current_id}:{port}:c', f'{config_id}:c')

    # Return a table with a single colored cell, using the port name from above.
    style = 'dashed' if isinstance(config, fdl.Partial) else 'solid'
    table = self._tag('table', style=style)
    tr = self._tag('tr')
    td = self._tag('td', port=port, bgcolor=self._color(config_id), style=style)
    return table(tr(td('')))

  def _render_sequence(self, sequence: Union[List[Any], Tuple[Any]]) -> str:
    """Renders the given sequence (list or tuple) as an HTML table."""
    table = self._tag('table')
    tr = self._tag('tr')
    td = self._tag('td')
    index_td = self._tag('td', cellpadding=0, bgcolor='#eeeeee')
    index_font = self._tag('font', point_size=6)

    if not sequence:
      return '[]' if isinstance(sequence, list) else '()'

    cells, indices = [], []
    for i, value in enumerate(sequence):
      cells.append(td(self._render_value(value)))
      indices.append(index_td(index_font(i)))
    row_stride = self._max_sequence_elements_per_row
    colspan = min(len(sequence), row_stride)
    rows = [self._header_row(type(sequence).__name__, colspan=colspan)]
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
    table = self._tag('table')
    tr = self._tag('tr')
    key_td = self._tag('td', align='right', bgcolor='#eeeeee')
    value_td = self._tag('td', align='left')

    rows = [header] if header is not None else []
    for key, value in dict_.items():
      key = key_format_fn(key)
      rows.append(tr([key_td(key), value_td(self._render_value(value))]))
    return table(rows)

  def _render_leaf(self, value: Any) -> str:
    """Renders `value` as its `__repr__` string."""
    value = special_value_codegen.transform_py_value(value, LazyImportManager())
    return html.escape(repr(value))


def render(config: fdl.Buildable) -> graphviz.Graph:
  """Renders the given `config` as a `graphviz.Graph`.

  Each config is rendered as a table of keys and values (with a header row).
  Any nested configs get their own separate table, with an edge pointing to
  their location in their parent config. If a config instance is present in
  multiple parent configs, it is only rendered once but will have multiple
  edges to parent configs.

  Args:
    config: The config to render.

  Returns:
    A `graphviz.Graph` object containing the resulting rendering of `config`.
    Standard `graphviz` methods can then be used to export this to a file.
  """
  return _GraphvizRenderer().render(config)
