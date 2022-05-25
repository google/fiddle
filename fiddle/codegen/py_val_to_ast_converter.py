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

"""Library for converting Python values to `ast` expressions."""

import ast
import builtins
import inspect
import types
from typing import Any, Callable, Union, Type, List, Optional, NamedTuple, Sequence

from fiddle import config
from fiddle import tagging
from fiddle.experimental import daglish

# A function that takes any python value, and returns an ast node.
PyValToAstFunc = Callable[[Any], ast.AST]

# A ValueConverterFunc is a function that takes a value and a PyValToAstFunc,
# and returns an AST node for the value.  The PyValToAstFunc should be used
# to convert nested values.  ValueConverterFunc may also return None,
# indicating that it could not convert the value.
ValueConverterFunc = Callable[[Any, PyValToAstFunc], Optional[ast.AST]]

# A ValueMatcher is a boolean predicate or a type, used to decide when to use a
# ValueConverterFunc.
ValueMatcher = Union[Callable[[Any], bool], Type]


class ValueConverter(NamedTuple):
  """A ValueConverterFunc triggered by a ValueMatcher and a priority.

  Attributes:
    matcher: A type or boolean predicate, used to indicate when the converter
      should be used.  If `matcher` is a type, then the converter is used for
      values of that type (but not for subtypes).  If `matcher` is a predicate,
      then the converter is used for values where the predicate returns True.
    priority: An int used to determine which converters take precedence. If two
      converters match a value, then the converter with the higher priority will
      be used.  (If both have the same priority, then the order is undefined.)
    converter: A function that takes two arguments (the value to convert and a
      `PyValToAstFunc` that can be used to convert nested values), and returns
      an `ast.AST` node (if the value can be converted) or `None` (otherwise).
  """
  matcher: ValueMatcher
  priority: int
  converter: ValueConverterFunc


def convert_py_val_to_ast(value: Any,
                          additional_converters: Sequence[ValueConverter] = ()):
  """Converts a Python value to an equivalent `ast` expression.

  I.e., `eval(ast.unparse(convert_py_val_to_ast(v))) == v`.

  By default, value converters are registered for the following types:

  * Builtin value types (`int`, `float`, `complex`, `bool`, `str`, `bytes`,
    `None`, and `Ellipsis`): returns an `ast.Constant`.
  * `list`: returns an `ast.List`.
  * `tuple`: returns an `ast.Tuple`.
  * `set`: returns an `ast.Set`
  * `dict`: Returns an `ast.Dict`
  * Fiddle buildable types (`fdl.Config`, `fdl.Partial`, `fdl.TaggedValue`):
    Returns an `ast.Call` that constructs the value.
  * Modules: Returns an `ast.Name` or `ast.Attribute` containing the full
    dotted name for the module.
  * Classes and functions (and anything else for which `is_importable` returns
    True): Returns an `ast.Name` or `ast.Attribute` containing the full dotted
    name for the object.
  * NamedTuples: Returns an `ast.Call` that constructs the value.

  Additional converters may be registered using the decorator
  `@register_py_val_to_ast_converter`.

  Args:
    value: The Python value that should be converted to ast.
    additional_converters: A list of `ValueConverter`s that should be added to
      the default list of converters.  If any converter has the same matcher as
      a default converter, then it will replace that converter.

  Returns:
    An AST node for an expression that evaluates to `value`.

  Raises:
    ValueError: If there is no registered value converter that can handle
    `value` (or some object nested in `value`).
  """
  converter = _py_val_to_ast_converter
  if additional_converters:
    converter = _py_val_to_ast_converter.copy()
    for additional_converter in additional_converters:
      converter.add_converter(additional_converter)
  return converter.convert(value)


ValueConverterDecorator = Callable[[ValueConverterFunc], ValueConverterFunc]


def register_py_val_to_ast_converter(matchers: Union[ValueMatcher,
                                                     Sequence[ValueMatcher]],
                                     priority=None) -> ValueConverterDecorator:
  """Decorator used to register ValueConverters for `convert_py_val_to_ast`.

  The decorated function should take two arguments (the value and a
  `PyValToAstFunc` that can be used to convert nested values), and should return
  an `ast.AST` node (if the value can be converted) or `None` (otherwise).

  Example usage:

  >>> @register_py_val_to_ast_converter(MyType)
  ... def convert_my_type(value, conversion_fn):
  ...   return ast.Call(func=conversion_fn(type(value)),
  ...                   args=[conversion_fn(value.x)],
  ...                   keywords=[])

  Args:
    matchers: A type or a predicate function (or a list of types or predicate
      functions), used to specify when this converter should be used.  If a type
      is given, then the converter is only used for that exact type (not
      subclasses).  If a predicate is given, then the converter is used whenever
      that predicate returns True.
    priority: A priority level for the converter.  Converters with higher
      priority take precedence over converters with lower priority.  The order
      for converters with the same priority is undefined.  If not specified,
      then priority defaults to 100 for type-based converters and 50 for
      predicate-based converters.

  Returns:
    A decorator function.
  """
  if not isinstance(matchers, list):
    matchers = [matchers]

  def decorator(converter: ValueConverterFunc) -> ValueConverterFunc:
    for matcher in matchers:
      if priority is None:
        matcher_priority = 100 if isinstance(matcher, type) else 50
      else:
        matcher_priority = priority
      _py_val_to_ast_converter.add_converter(
          ValueConverter(matcher, matcher_priority, converter))
    return converter

  return decorator


class _PyValToAstConverter:
  """Class that converts Python values to equivalent `ast.expr`s.

  I.e., `eval(ast.unparse(convert(v))) == v`.

  `_PyValToAstConverter` owns a collection of `ValueConverter`s,
  which it uses to convert values to `ast.AST` nodes.
  """

  def __init__(self, converters: List[ValueConverter]):
    # Dispatchers are stored as a flat list (sorted by priority), which we
    # need to scan through for each value we convert.
    #
    # If efficiency becomes a concern here, then we could split type-based
    # dispatchers into a separate dict (keyed by type).  We could quickly check
    # if a type-based dispatcher applies, and if so, then only check other
    # converters whose priority is higher than that type-based dispatcher.
    self._converters: List[ValueConverter] = converters

  def convert(self, value: Any) -> ast.AST:
    """Returns an AST node for an expression that evaluates to `value`."""
    for registered_converter in self._converters:
      if isinstance(registered_converter.matcher, type):
        if type(value) is registered_converter.matcher:  # pylint: disable=unidiomatic-typecheck
          result = registered_converter.converter(value, self.convert)
          if result is not None:
            return result
      else:
        if registered_converter.matcher(value):
          result = registered_converter.converter(value, self.convert)
          if result is not None:
            return result

    raise ValueError(f'{type(self)} has no registered converter ' +
                     f'for {type(value)}')

  def add_converter(self, converter: ValueConverter):
    """Adds a new ValueConverter to this _PyValToAstConverter."""
    # Replace matcher if it's already registered.
    self._converters = [
        dispatcher for dispatcher in self._converters
        if dispatcher.matcher is not converter.matcher
    ]
    self._converters.append(converter)
    # Sort by priority (high to low).
    self._converters.sort(key=lambda elt: -elt.priority)

  def copy(self):
    """Creates a shallow copy of this object."""
    return _PyValToAstConverter(list(self._converters))


# The "default" _PyValToAstConverter.
_py_val_to_ast_converter = _PyValToAstConverter([])


@register_py_val_to_ast_converter(
    [int, float, complex, bool, str, bytes,
     type(None),
     type(Ellipsis)])
def _convert_constant(value: Any, conversion_fn: PyValToAstFunc) -> ast.AST:
  """Converts a constant (such as an int) to AST."""
  del conversion_fn  # Not used.
  return ast.Constant(value)


@register_py_val_to_ast_converter(list)
def _convert_list(value: Any, conversion_fn: PyValToAstFunc) -> ast.AST:
  """Converts a list to AST."""
  return ast.List(elts=[conversion_fn(v) for v in value], ctx=ast.Load())


@register_py_val_to_ast_converter(tuple)
def _convert_tuple(value: Any, conversion_fn: PyValToAstFunc) -> ast.AST:
  """Converts a tuple to AST."""
  return ast.Tuple(elts=[conversion_fn(v) for v in value], ctx=ast.Load())


@register_py_val_to_ast_converter(dict)
def _convert_dict(value: Any, conversion_fn: PyValToAstFunc) -> ast.AST:
  """Converts a dict to AST."""
  return ast.Dict(
      keys=[conversion_fn(k) for k in value.keys()],
      values=[conversion_fn(v) for v in value.values()])


@register_py_val_to_ast_converter(set)
def _convert_set(value: Any, conversion_fn: PyValToAstFunc) -> ast.AST:
  """Converts a set to AST."""
  return ast.Set(elts=[conversion_fn(v) for v in value])


@register_py_val_to_ast_converter(daglish.is_namedtuple_instance)
def _convert_namedtuple(value: Any, conversion_fn: PyValToAstFunc) -> ast.AST:
  """Converts an instance of a named tuple to AST."""
  return ast.Call(
      func=conversion_fn(type(value)),
      args=[],
      keywords=[
          ast.keyword(arg=name, value=conversion_fn(value))
          for (name, value) in value._asdict().items()
      ])


@register_py_val_to_ast_converter([config.Config, config.Partial])
def _convert_buildable(value: Any, conversion_fn: PyValToAstFunc) -> ast.AST:
  """Converts a fdl.Config or fdl.Partial to AST."""
  return ast.Call(
      func=conversion_fn(type(value)),
      args=[conversion_fn(value.__fn_or_cls__)],
      keywords=[
          ast.keyword(arg=k, value=conversion_fn(v))
          for (k, v) in value.__arguments__.items()
      ])


@register_py_val_to_ast_converter(tagging.TaggedValue)
def _convert_tagged_value(value: Any, conversion_fn: PyValToAstFunc) -> ast.AST:
  """Converts a fdl.TaggedValue to AST."""
  node = conversion_fn(value.value)
  for tag in value.tags:
    node = ast.Call(
        func=ast.Attribute(
            value=conversion_fn(tag), attr='new', ctx=ast.Load()),
        args=[node],
        keywords=[])
  return node


@register_py_val_to_ast_converter(types.ModuleType)
def _convert_module(value: Any, conversion_fn: PyValToAstFunc) -> ast.AST:
  """Converts a module to AST."""
  del conversion_fn  # Unused.
  return dotted_name_to_ast(f'{value.__name__}')


def dotted_name_to_ast(dotted_name: str) -> ast.AST:
  """Converts a dotted name to an ast.Attribute."""
  pieces = dotted_name.split('.')
  result = ast.Name(pieces[0], ctx=ast.Load())
  for piece in pieces[1:]:
    result = ast.Attribute(value=result, attr=piece, ctx=ast.Load())
  return result


def is_importable(value: Any) -> bool:
  """Returns true if `value` has a module and a __qualname__."""
  return inspect.getmodule(value) is not None and hasattr(value, '__qualname__')


@register_py_val_to_ast_converter(is_importable)
def _convert_importable(value: Any, conversion_fn: PyValToAstFunc) -> ast.AST:
  """Converts an importable value to the AST for `<module_name>.<qualname>`."""
  module = inspect.getmodule(value)
  if module.__name__ == '__main__' or module is builtins:
    return dotted_name_to_ast(value.__qualname__)
  else:
    result = conversion_fn(inspect.getmodule(value))
    for piece in value.__qualname__.split('.'):
      result = ast.Attribute(value=result, attr=piece, ctx=ast.Load())
    return result
