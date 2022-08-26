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

"""Library for converting Python values to `cst` expressions."""

import builtins
import functools
import inspect
import types
from typing import Any, Callable, Union, Type, List, Optional, NamedTuple, Sequence

from fiddle import config
from fiddle import tagging
from fiddle.experimental import daglish_legacy

import libcst as cst

# A function that takes any python value, and returns a cst node.
PyValToCstFunc = Callable[[Any], cst.CSTNode]

# A ValueConverterFunc is a function that takes a value and a PyValToCstFunc,
# and returns a CST node for the value.  The PyValToCstFunc should be used
# to convert nested values.  ValueConverterFunc may also return None,
# indicating that it could not convert the value.
ValueConverterFunc = Callable[[Any, PyValToCstFunc], Optional[cst.CSTNode]]

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
      `PyValToCstFunc` that can be used to convert nested values), and returns
      an `cst.CSTNode` node (if the value can be converted) or `None`
      (otherwise).
  """
  matcher: ValueMatcher
  priority: int
  converter: ValueConverterFunc


def convert_py_val_to_cst(
    value: Any, additional_converters: Sequence[ValueConverter] = ()
) -> cst.CSTNode:
  """Converts a Python value to an equivalent `cst` expression.

  I.e., `eval(convert_py_val_to_cst(v).code) == v`.

  By default, value converters are registered for the following types:

  * `int` returns a `cst.Integer`
  * `float` returns a `cst.Float`
  * `bool` returns a `cst.Name` containing "True" or "False".
  * `None` returns a `cst.Name` containing "None".
  * `str` and `bytes` return a `cst.SimpleString`.
  * `Ellipsis` returns a `cst.Ellipsis`.
  * `complex` returns a `cst.Imaginary` or a `cst.BinaryOperation`.
  * `list`: returns an `cst.List`.
  * `tuple`: returns an `cst.Tuple`.
  * `set`: returns an `cst.Set` or a `cst.Call` for `set()`.
  * `dict`: Returns an `cst.Dict`.
  * Fiddle buildable types (`fdl.Config`, `fdl.Partial`, `fdl.TaggedValue`):
    Returns a `cst.Call` that constructs the value.
  * Modules: Returns a `cst.Name` or `cst.Attribute` containing the full
    dotted name for the module.
  * Classes and functions (and anything else for which `is_importable` returns
    True): Returns a `cst.Name` or `cst.Attribute` containing the full dotted
    name for the object.
  * NamedTuples: Returns a `cst.Call` that constructs the value.

  Additional converters may be registered using the decorator
  `@register_py_val_to_cst_converter`.

  Args:
    value: The Python value that should be converted to a cst.CSTNode.
    additional_converters: A list of `ValueConverter`s that should be added to
      the default list of converters.  If any converter has the same matcher as
      a default converter, then it will replace that converter.

  Returns:
    A CST node for an expression that evaluates to `value`.

  Raises:
    ValueError: If there is no registered value converter that can handle
    `value` (or some object nested in `value`).
  """
  converter = _py_val_to_cst_converter
  if additional_converters:
    converter = _py_val_to_cst_converter.copy()
    for additional_converter in additional_converters:
      converter.add_converter(additional_converter)
  return converter.convert(value)


ValueConverterDecorator = Callable[[ValueConverterFunc], ValueConverterFunc]


def register_py_val_to_cst_converter(matchers: Union[ValueMatcher,
                                                     Sequence[ValueMatcher]],
                                     priority=None) -> ValueConverterDecorator:
  """Decorator used to register ValueConverters for `convert_py_val_to_cst`.

  The decorated function should take two arguments (the value and a
  `PyValToCstFunc` that can be used to convert nested values), and should return
  an `cst.CSTNode` node (if the value can be converted) or `None` (otherwise).

  Example usage:

  >>> @register_py_val_to_cst_converter(MyType)
  ... def convert_my_type(value, conversion_fn):
  ...   return cst.Call(func=conversion_fn(type(value)),
  ...                   args=[cst.Arg(conversion_fn(value.x))])

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
      _py_val_to_cst_converter.add_converter(
          ValueConverter(matcher, matcher_priority, converter))
    return converter

  return decorator


class _PyValToCstConverter:
  """Class that converts Python values to equivalent `cst` expression nodes.

  I.e., `eval(convert(v).code) == v`.

  `_PyValToCstConverter` owns a collection of `ValueConverter`s,
  which it uses to convert values to `cst.CSTNode` nodes.
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

  def convert(self, value: Any) -> cst.CSTNode:
    """Returns a CST node for an expression that evaluates to `value`."""
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
    """Adds a new ValueConverter to this _PyValToCstConverter."""
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
    return _PyValToCstConverter(list(self._converters))


# The "default" _PyValToCstConverter.
_py_val_to_cst_converter = _PyValToCstConverter([])


def kwarg_to_cst(name: str, value: Any) -> cst.Arg:
  """Returns CST for a keyword argument (formatted w/ google style)."""
  return cst.Arg(
      value,
      cst.Name(name),
      equal=cst.AssignEqual(cst.SimpleWhitespace(''), cst.SimpleWhitespace('')))


@register_py_val_to_cst_converter(int)
def _convert_int(value: Any, conversion_fn: PyValToCstFunc) -> cst.CSTNode:
  """Converts a constant int to CST."""
  del conversion_fn  # Not used.
  return cst.Integer(repr(value))


@register_py_val_to_cst_converter(float)
def _convert_float(value: Any, conversion_fn: PyValToCstFunc) -> cst.CSTNode:
  """Converts a constant float to CST."""
  del conversion_fn  # Not used.
  return cst.Float(repr(value))


@register_py_val_to_cst_converter([bool, type(None)])
def _convert_bool(value: Any, conversion_fn: PyValToCstFunc) -> cst.CSTNode:
  """Converts a constant bool or None to CST."""
  del conversion_fn  # Not used.
  return cst.Name(repr(value))


@register_py_val_to_cst_converter([str, bytes])
def _convert_str(value: Any, conversion_fn: PyValToCstFunc) -> cst.CSTNode:
  """Converts a constant str or bytes to CST."""
  del conversion_fn  # Not used.
  return cst.SimpleString(repr(value))


@register_py_val_to_cst_converter(type(Ellipsis))
def _convert_ellipsis(value: Any, conversion_fn: PyValToCstFunc) -> cst.CSTNode:
  """Converts a constant Ellipsis to CST."""
  del conversion_fn, value  # Not used.
  return cst.Ellipsis()


@register_py_val_to_cst_converter(complex)
def _convert_complex(value: Any, conversion_fn: PyValToCstFunc) -> cst.CSTNode:
  """Converts a constant complex number to CST."""
  del conversion_fn  # Not used.
  if value.real:
    return cst.BinaryOperation(
        left=cst.Float(repr(value.real)),
        operator=cst.Add(cst.SimpleWhitespace(''), cst.SimpleWhitespace('')),
        right=cst.Imaginary(repr(value.imag) + 'j'),
        lpar=[cst.LeftParen(cst.SimpleWhitespace(''))],
        rpar=[cst.RightParen(cst.SimpleWhitespace(''))])
  else:
    return cst.Imaginary(repr(value))


@register_py_val_to_cst_converter(list)
def _convert_list(value: Any, conversion_fn: PyValToCstFunc) -> cst.CSTNode:
  """Converts a list to CST."""
  return cst.List([cst.Element(conversion_fn(v)) for v in value])


@register_py_val_to_cst_converter(tuple)
def _convert_tuple(value: Any, conversion_fn: PyValToCstFunc) -> cst.CSTNode:
  """Converts a tuple to CST."""
  return cst.Tuple([cst.Element(conversion_fn(v)) for v in value])


@register_py_val_to_cst_converter(dict)
def _convert_dict(value: Any, conversion_fn: PyValToCstFunc) -> cst.CSTNode:
  """Converts a dict to CST."""
  return cst.Dict([
      cst.DictElement(conversion_fn(key), conversion_fn(val))
      for (key, val) in value.items()
  ])


@register_py_val_to_cst_converter(set)
def _convert_set(value: Any, conversion_fn: PyValToCstFunc) -> cst.CSTNode:
  """Converts a set to CST."""
  if value:
    return cst.Set([cst.Element(conversion_fn(v)) for v in value])
  else:
    return cst.Call(func=cst.Name('set'))


@register_py_val_to_cst_converter(daglish_legacy.is_namedtuple_instance)
def _convert_namedtuple(value: Any,
                        conversion_fn: PyValToCstFunc) -> cst.CSTNode:
  """Converts an instance of a named tuple to CST."""
  return cst.Call(
      func=conversion_fn(type(value)),
      args=[
          kwarg_to_cst(arg_name, conversion_fn(arg_val))
          for (arg_name, arg_val) in value._asdict().items()
      ])


@register_py_val_to_cst_converter([config.Config, config.Partial])
def _convert_buildable(value: Any,
                       conversion_fn: PyValToCstFunc) -> cst.CSTNode:
  """Converts a fdl.Config or fdl.Partial to CST."""
  args = [cst.Arg(conversion_fn(value.__fn_or_cls__))]
  for (arg_name, arg_val) in value.__arguments__.items():
    if arg_name in value.__argument_tags__:
      for tag in value.__argument_tags__[arg_name]:
        arg_val = tag.new(arg_val)
    args.append(kwarg_to_cst(arg_name, conversion_fn(arg_val)))
  return cst.Call(func=conversion_fn(type(value)), args=args)


@register_py_val_to_cst_converter(tagging.TaggedValueCls)
def _convert_tagged_value(value: Any,
                          conversion_fn: PyValToCstFunc) -> cst.CSTNode:
  """Converts a fdl.TaggedValue to CST."""
  node = conversion_fn(value.value)
  for tag in sorted(value.tags, key=repr, reverse=True):
    node = cst.Call(
        func=cst.Attribute(value=conversion_fn(tag), attr=cst.Name('new')),
        args=[cst.Arg(node)])
  return node


@register_py_val_to_cst_converter(types.ModuleType)
def _convert_module(value: Any, conversion_fn: PyValToCstFunc) -> cst.CSTNode:
  """Converts a module to CST."""
  del conversion_fn  # Unused.
  return dotted_name_to_cst(f'{value.__name__}')


def dotted_name_to_cst(dotted_name: str) -> cst.CSTNode:
  """Converts a dotted name to a cst.Attribute."""
  pieces = dotted_name.split('.')
  result = cst.Name(pieces[0])
  for piece in pieces[1:]:
    result = cst.Attribute(value=result, attr=cst.Name(piece))
  return result


def is_importable(value: Any) -> bool:
  """Returns true if `value` has a module and a __qualname__."""
  return inspect.getmodule(value) is not None and hasattr(value, '__qualname__')


@register_py_val_to_cst_converter(is_importable)
def _convert_importable(value: Any,
                        conversion_fn: PyValToCstFunc) -> cst.CSTNode:
  """Converts an importable value to the CST for `<module_name>.<qualname>`."""
  module = inspect.getmodule(value)
  if module.__name__ == '__main__' or module is builtins:
    return dotted_name_to_cst(value.__qualname__)
  else:
    result = conversion_fn(inspect.getmodule(value))
    for piece in value.__qualname__.split('.'):
      result = cst.Attribute(value=result, attr=cst.Name(piece))
    return result


@register_py_val_to_cst_converter(functools.partial)
def _convert_partial(value: functools.partial,
                     conversion_fn: PyValToCstFunc) -> cst.CSTNode:
  """Converts a functools.partial to CST."""
  return cst.Call(
      func=conversion_fn(functools.partial),
      args=([cst.Arg(conversion_fn(value.func))] +
            [cst.Arg(conversion_fn(arg)) for arg in value.args] + [
                kwarg_to_cst(arg_name, conversion_fn(arg_val))
                for (arg_name, arg_val) in value.keywords.items()
            ]))
